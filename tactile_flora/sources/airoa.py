"""Source adapter for AIRoA MoMa (`airoa-org/airoa-moma`, LeRobot v3-style, Toyota HSR).

Owns: the AIRoA MoMa on-disk layout and its conversion to the episode clock.
Must never: filter episodes in discover(), write to disk, or decide success semantics (raw
fields go to ``meta`` for labels/mapping.py).

Resolved open questions (verified 2026-10-21, docs/research/datasets.md; repo is gated, so some
items could not be confirmed):
- Layout: ``<root>/{data/chunk-000/file-*.parquet, meta/{info.json,stats.json,tasks.parquet,
  episodes/}}`` (LeRobot v3-style, NOT v2.1 as the spec assumed). Read via ``sources/_lerobot.py``.
  Episode id is the episode_index as a string. 10 Hz. 23,762 eps, 97.4% successful.
- Episode metadata columns (meta/episodes, as on the dataset card): episode_index, tasks, length,
  uuid, task_type, task_success, short_horizon_task, primitive_action, label, hsr_id,
  location_name, calib, version, git_hash. Missing columns become None.
- Wrist wrench: key UNVERIFIED in this release. The adapter reads ``observation.wrist.wrench`` if
  present, else the first feature whose name tokens (split on ``._``) contain ``ft`` or whose
  name contains ``wrench``. A 600-wide vector (the 5k release: 100-sample history x 6) is
  reshaped (100, 6) and the sample at ``WRENCH_HISTORY_ROW`` (-1, latest; order UNVERIFIED) is
  used. Columns fx fy fz tx ty tz, arm "R", channel wrist, raw units. 4,608 heiwajima episodes
  (5k release) tile one sample 100x.
- Gripper: ``observation.state`` index 5 (hand_motor_joint; state is arm_lift, arm_flex, arm_roll,
  wrist_flex, wrist_roll, hand_motor_joint, head_pan, head_tilt), min-max normalized to 0..1
  with meta/stats.json min/max (episode min/max when stats are absent). Polarity unverified.
- Cameras: ``observation.image.head`` = scene, ``observation.image.hand`` = wrist_right; 480x640.
  v3 mp4 files are concatenated: in-file start in ``meta["video_from_s"][role]``.
- Session: ``<hsr_id>_<date>`` where date is the first yyyy-mm-dd or yyyymmdd found in ``uuid``
  then ``version``; ``undated`` if none (the date source is UNVERIFIED).
- License: CC-BY-NC-SA-4.0. (airoa-moma-5k is under different research-only terms and is not
  this adapter's target.)
- NOT verified: the per-primitive-action O/X and the failure-cause schema; only the raw columns
  above are recorded. native_labels() is [] and labels/mapping.py reads ``meta``.

meta keys: task_success, primitive_action, label, task_type, hsr_id, uuid, version,
wrench_key, clock_offset_s, video_from_s.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Episode,
    ForceStream,
    GripperStream,
    Label,
    VideoRef,
    WristArmForce,
)
from tactile_flora.sources._common import assign_splits, py
from tactile_flora.sources._lerobot import LeRobotV3, column_array
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "CC-BY-NC-SA-4.0"
WRENCH_KEY = "observation.wrist.wrench"
WRENCH_HISTORY_ROW = -1
STATE_KEY = "observation.state"
HAND_MOTOR_INDEX = 5
CAMERA_ROLES = {"observation.image.head": "scene", "observation.image.hand": "wrist_right"}
DATE_RE = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})")


def wrench_key(feature_keys: list[str]) -> str | None:
    """Pick the wrist wrench feature: the known key, else any with 'wrench' or an 'ft' token."""
    if WRENCH_KEY in feature_keys:
        return WRENCH_KEY
    for k in feature_keys:
        tokens = re.split(r"[._]", k.lower())
        if "wrench" in k.lower() or "ft" in tokens:
            return k
    return None


@register_source("airoa")
class AiroaAdapter:
    """Adapter for AIRoA MoMa in LeRobot v3-style layout."""

    name = "airoa"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._ds: LeRobotV3 | None = None

    def discover(self, root: Path) -> list[str]:
        """Return every episode index under ``root`` as a string."""
        self._root = root
        self._ds = LeRobotV3(root)
        return [str(i) for i in self._ds.episode_indices()]

    def _dataset(self) -> LeRobotV3:
        if self._ds is None:
            self.discover(self._root)
        assert self._ds is not None
        return self._ds

    @staticmethod
    def _session(rec: dict[str, Any]) -> str:
        hsr = str(rec.get("hsr_id", "hsr"))
        for field in ("uuid", "version"):
            m = DATE_RE.search(str(rec.get(field) or ""))
            if m:
                return f"{hsr}_{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return f"{hsr}_undated"

    def load(self, episode_id: str) -> Episode:
        """Load one episode on the episode clock (zero at the first scene frame)."""
        ds = self._dataset()
        idx = int(episode_id)
        rec = ds.record(idx)
        df = ds.frames(idx)
        t_raw = np.asarray(df["timestamp"], dtype=np.float64)
        offset = float(t_raw[0])
        t = t_raw - offset
        n = len(df)

        feats = list(ds.info.get("features", {}))
        wkey = wrench_key(feats)
        if wkey is None:
            raise KeyError("no wrist wrench column (observation.wrist.wrench / *wrench* / *ft*)")
        w = column_array(df, wkey)
        if w.shape[1] % 6 != 0:
            raise ValueError(f"{wkey} has width {w.shape[1]}, expected a multiple of 6")
        if w.shape[1] > 6:
            w = w.reshape(n, -1, 6)[:, WRENCH_HISTORY_ROW, :]
        force = ForceStream(
            channel="wrist",
            arms={
                "R": WristArmForce(
                    fx=w[:, 0].tolist(),
                    fy=w[:, 1].tolist(),
                    fz=w[:, 2].tolist(),
                    tx=w[:, 3].tolist(),
                    ty=w[:, 4].tolist(),
                    tz=w[:, 5].tolist(),
                )  # fmt: skip
            },
            rate_hz=ds.fps,
            t=t.tolist(),
        )

        q = column_array(df, STATE_KEY)[:, HAND_MOTOR_INDEX]
        st = ds.stats().get(STATE_KEY, {})
        lo = float(np.asarray(st["min"])[HAND_MOTOR_INDEX]) if "min" in st else float(q.min())
        hi = float(np.asarray(st["max"])[HAND_MOTOR_INDEX]) if "max" in st else float(q.max())
        g = np.clip((q - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(q)
        gripper = GripperStream(arms={"R": g.tolist()}, rate_hz=ds.fps, t=t.tolist())

        cameras: dict[str, VideoRef] = {}
        video_from: dict[str, float] = {}
        for key in ds.video_keys():
            role = CAMERA_ROLES.get(key)
            if role is None:
                continue
            path, from_ts = ds.video(idx, key)
            cameras[role] = VideoRef(path=str(path), fps=ds.fps, n_frames=n)
            video_from[role] = from_ts

        tasks = py(rec.get("tasks")) or []
        meta: dict[str, Any] = {
            "task_success": rec.get("task_success"),
            "primitive_action": rec.get("primitive_action"),
            "label": rec.get("label"),
            "task_type": rec.get("task_type"),
            "hsr_id": rec.get("hsr_id"),
            "uuid": rec.get("uuid"),
            "version": rec.get("version"),
            "wrench_key": wkey,
            "clock_offset_s": offset,
            "video_from_s": video_from,
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=self._session(rec),
            task=strip_outcome_words(str(tasks[0]) if tasks else ""),
            duration_s=float(t[-1]) if n else 0.0,
            cameras=cameras,
            force=force,
            gripper=gripper,
            labels=[],
            license=LICENSE,
            meta=meta,
        )

    def splits(self) -> dict[str, list[str]]:
        """Return train/val/test by robot-id + date session."""
        ds = self._dataset()
        return assign_splits(
            {str(i): self._session(ds.record(i)) for i in ds.episode_indices()}, self.cfg
        )

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return no labels: raw success fields are in meta for labels/mapping.py."""
        return []
