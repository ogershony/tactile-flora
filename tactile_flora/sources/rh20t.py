"""Source adapter for the RH20T community LeRobot v3 ports (`robot-lev/rh20t_cfg{3,4,5,7}`).

Owns: the RH20T port layout and its conversion to the episode clock.
Must never: filter episodes in discover(), write to disk, or turn ``meta.rating`` into an outcome
label (labels/mapping.py does that from ``meta["rating"]``).

Resolved open questions (verified 2026-10-21, docs/research/datasets.md):
- Layout: ``<root>/rh20t_cfg<N>/`` is one LeRobot v3 dataset each (meta/info.json, meta/episodes,
  data/, videos/, meta/rh20t_episodes.json). Episode id is ``<cfgdir>/<episode_index>``. 10 fps.
  Unofficial ports, RGB only. NO fingertip tactile in any port (the original uSkin data of Cfg 7
  was dropped), so this source feeds the wrist channel only.
- Force: ``observation.robot_ft`` (6) = fx fy fz tx ty tz -> WristArmForce arm "R", channel
  wrist, raw units. Calibration is UNVERIFIED (not known whether the port uses the zeroed or raw
  force_torque_base; cfg3 is clipped to +-300, cfg7 Fz min -571 N). ``observation.force/torque``
  are all zero on cfg5 (Franka) so they are never used.
- Gripper: ``observation.state.gripper`` (1), divided by the per-config max taken from
  meta/stats.json (episode max if stats are absent) and clipped to 0..1, arm "R". Whether larger
  means closed is unverified.
- Cameras: every ``observation.images.cam_<serial>`` video feature (9-10 per config, 360x640),
  or only ``cfg.cameras`` (in that order) when the config sets it, because a partial download
  (data/colab/fetch_rh20t.py: one external and one in-hand camera per config) would otherwise
  reference 8 missing mp4 streams per episode and fail integrity everywhere. The first camera
  kept is role ``scene``; the others are ``extra_<serial>``. A config restriction that leaves a
  port without any camera raises ValueError at load. v3 mp4 files are concatenated: in-file
  start of each camera is ``meta["video_from_s"][role]``.
- Rating: ``meta.rating`` (int64 per frame, range -1..9, -1 presumably unrated, semantics
  undocumented) is read from the first frame into ``meta["rating"]``. No Label is produced here.
- Scene and license: ``meta/rh20t_episodes.json`` gives ``folder`` per episode (list of dicts or
  a dict keyed by episode index; scene = digits after "scene_"). Scenes 1-5: CC-BY-SA-4.0,
  6-10 CC-BY-NC-4.0 (``SA_MAX_SCENE``); stored per episode in ``Episode.license`` and
  ``meta["license"]``. Session = ``scene_<NNNN>``; splits by scene. The ports contain faces and
  voices.
- Unverified: the spec said rating is 0..9; the data says -1..9.

meta keys: rating (int), scene (int or None), license, folder, cfg, episode_index,
clock_offset_s, video_from_s.
"""

from __future__ import annotations

import json
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

LICENSE_SA = "CC-BY-SA-4.0"
LICENSE_NC = "CC-BY-NC-4.0"
SA_MAX_SCENE = 5
FT_KEY = "observation.robot_ft"
GRIPPER_KEY = "observation.state.gripper"
RATING_KEY = "meta.rating"
CAMERA_PREFIX = "observation.images.cam_"
SCENE_RE = re.compile(r"scene_(\d+)")


def scene_license(scene: int | None) -> str:
    """Return the per-scene license (unknown scenes get the stricter non-commercial one)."""
    return LICENSE_SA if scene is not None and scene <= SA_MAX_SCENE else LICENSE_NC


@register_source("rh20t")
class Rh20tAdapter:
    """Adapter for RH20T LeRobot v3 ports."""

    name = "rh20t"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._index: dict[str, tuple[Path, int]] = {}
        self._datasets: dict[Path, LeRobotV3] = {}

    def discover(self, root: Path) -> list[str]:
        """Return every ``<cfgdir>/<episode_index>`` under ``root``."""
        self._root = root
        self._index, self._datasets = {}, {}
        for d in sorted(root.glob("rh20t_cfg*")):
            if not (d / "meta" / "info.json").exists():
                continue
            ds = self._dataset(d)
            for idx in ds.episode_indices():
                self._index[f"{d.name}/{idx}"] = (d, idx)
        return list(self._index)

    def _dataset(self, d: Path) -> LeRobotV3:
        if d not in self._datasets:
            self._datasets[d] = LeRobotV3(d)
        return self._datasets[d]

    def _ensure(self) -> None:
        if not self._index:
            self.discover(self._root)

    @staticmethod
    def _folder(d: Path, idx: int) -> str:
        p = d / "meta" / "rh20t_episodes.json"
        if not p.exists():
            return ""
        data: Any = json.loads(p.read_text())
        entry: Any = None
        if isinstance(data, dict):
            entry = data.get(str(idx))
        elif isinstance(data, list):
            entry = next(
                (e for e in data if isinstance(e, dict) and e.get("episode_index") == idx), None
            )
            if entry is None and 0 <= idx < len(data):
                entry = data[idx]
        return str(entry.get("folder", "")) if isinstance(entry, dict) else ""

    def _cameras(self, ds: LeRobotV3) -> list[str]:
        """Camera feature keys in role order: ``cfg.cameras`` (those this port has) or all."""
        keys = [k for k in ds.video_keys() if k.startswith(CAMERA_PREFIX)]
        if self.cfg.cameras is None:
            return keys
        have = set(keys)
        return [k for k in self.cfg.cameras if k in have]

    def load(self, episode_id: str) -> Episode:
        """Load one episode on the episode clock (zero at the first scene frame)."""
        self._ensure()
        d, idx = self._index[episode_id]
        ds = self._dataset(d)
        rec = ds.record(idx)
        df = ds.frames(idx)
        t_raw = np.asarray(df["timestamp"], dtype=np.float64)
        offset = float(t_raw[0])
        t = t_raw - offset
        n = len(df)

        w = column_array(df, FT_KEY)
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

        g = column_array(df, GRIPPER_KEY)[:, 0]
        stat_max = ds.stats().get(GRIPPER_KEY, {}).get("max")
        gmax = float(np.max(stat_max)) if stat_max is not None else float(np.max(g))
        gnorm = np.clip(g / gmax, 0.0, 1.0) if gmax > 0 else np.zeros_like(g)
        gripper = GripperStream(arms={"R": gnorm.tolist()}, rate_hz=ds.fps, t=t.tolist())

        cameras: dict[str, VideoRef] = {}
        video_from: dict[str, float] = {}
        cams = self._cameras(ds)
        if not cams:
            raise ValueError(f"{d.name}: none of cfg.cameras {self.cfg.cameras} is a video feature")
        for i, key in enumerate(cams):
            role = "scene" if i == 0 else f"extra_{key.removeprefix(CAMERA_PREFIX)}"
            path, from_ts = ds.video(idx, key)
            cameras[role] = VideoRef(path=str(path), fps=ds.fps, n_frames=n)
            video_from[role] = from_ts

        folder = self._folder(d, idx)
        m = SCENE_RE.search(folder)
        scene = int(m.group(1)) if m else None
        lic = scene_license(scene)
        rating = int(np.asarray(df[RATING_KEY].iloc[0]).reshape(-1)[0]) if RATING_KEY in df else -1
        tasks = py(rec.get("tasks")) or []
        task_text = str(tasks[0]) if tasks else folder
        meta: dict[str, Any] = {
            "rating": rating,
            "scene": scene,
            "license": lic,
            "folder": folder,
            "cfg": d.name,
            "episode_index": idx,
            "clock_offset_s": offset,
            "video_from_s": video_from,
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=f"scene_{scene:04d}" if scene is not None else "scene_unknown",
            task=strip_outcome_words(task_text),
            duration_s=float(t[-1]) if n else 0.0,
            cameras=cameras,
            force=force,
            gripper=gripper,
            labels=[],
            license=lic,
            meta=meta,
        )

    def splits(self) -> dict[str, list[str]]:
        """Return train/val/test by scene (a scene never straddles two splits)."""
        self._ensure()
        sessions: dict[str, str] = {}
        for eid, (d, idx) in self._index.items():
            m = SCENE_RE.search(self._folder(d, idx))
            sessions[eid] = f"scene_{int(m.group(1)):04d}" if m else "scene_unknown"
        return assign_splits(sessions, self.cfg)

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return no labels: ``meta["rating"]`` is mapped by labels/mapping.py."""
        return []
