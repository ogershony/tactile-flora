"""Source adapter for SharpaDex v1.0 (`Sharpa-Robotics/SharpaDex-v1.0`, LeRobot v3 export).

Owns: the SharpaDex on-disk layout and its conversion to the episode clock.
Must never: filter episodes in discover(), write to disk, or double count the v2.1 export.

Resolved open questions (verified 2026-10-21, docs/research/datasets.md):
- Layout: ``<root>/<task>/season_<id>_train/lerobot_v3.0/{data,videos,meta}``. Only the
  ``lerobot_v3.0`` export is read; ``lerobot_v2.1`` holds the SAME demos and is ignored. Episode
  id is ``<task>/season_<id>_train/<episode_index>``. 28,993 eps, 59 tasks, 495 seasons, 30 fps.
  Read with pyarrow via ``sources/_lerobot.py`` (no lerobot package).
- Force: ``observation.tactile`` (60 f32) is 10 fingertips x 6 (fx,fy,fz,tx,ty,tz), left hand
  then right, each thumb,index,middle,ring,little; fz of finger k is index 6k+2. The adapter
  takes right thumb fz (index 32) as f_a and right index fz (index 38) as f_b, channel pad, arm
  "R" (the right hand). Units are NOT documented in the dataset card; values are stored raw.
- Gripper: no gripper exists on a dexterous hand. ``GripperStream`` "R" is a CLOSURE PROXY: the
  right-index MCP joint from the 65-D ``observation.state``, min-max normalized per episode to
  0..1 (flat signal -> zeros). The dimension is found by name in info.json (name containing
  ``right``, ``index`` and ``mcp``); if the names are absent it falls back to
  ``RIGHT_INDEX_MCP_FALLBACK_INDEX``, which is UNVERIFIED. Polarity (is larger = more closed?)
  is also unverified.
- Cameras: ``observation.images.head_left`` = scene, ``wrist_left``, ``wrist_right``; 480x480 h264.
  v3 concatenates episodes into one mp4: the in-file start of an episode is
  ``meta["video_from_s"][role]`` (seconds). Tactile/deform/raw video streams and head_right are
  skipped (bandwidth).
- Clock: zero is the first frame (parquet ``timestamp[0]``, normally 0); the offset is stored in
  ``meta["clock_offset_s"]`` and the head_left from_timestamp is in ``meta["video_from_s"]``.
- Subtasks: derived from the per-frame ``subtask_index`` column (run-length segments) and looked
  up in ``meta/subtasks.parquet`` (text and task_skill columns, matched by name). The exact
  parquet schema is UNVERIFIED (spec lists start_step, end_step, text, task_skill, label).
- Native labels: NONE (demonstrations only; there is no success flag), so native_labels() is [].
- License: CC-BY-NC-4.0 stored as given (configs/sources/sharpadex.yaml still lists CC-BY-4.0
  from the spec; the license check in verify/ will therefore flag it until the config is updated).
- Session = season id; splits are by season (hash of session, ``val_fraction``/``test_fraction``
  in ``split_rules``).

meta keys: subtasks (list of {start_step,end_step,text,task_skill}), season, task_dir,
episode_index, clock_offset_s, video_from_s, closure_state_index.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Episode,
    ForceStream,
    GripperStream,
    Label,
    PadArmForce,
    VideoRef,
)
from tactile_flora.sources._common import assign_splits, py
from tactile_flora.sources._lerobot import LeRobotV3, column_array
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "CC-BY-NC-4.0"
EXPORT_DIR = "lerobot_v3.0"
TACTILE_KEY = "observation.tactile"
STATE_KEY = "observation.state"
RIGHT_THUMB_FZ = 32
RIGHT_INDEX_FZ = 38
RIGHT_INDEX_MCP_FALLBACK_INDEX = 40  # UNVERIFIED placeholder; resolved by name when possible
CLOSURE_NAME_HINTS = ("right", "index", "mcp")
CAMERA_ROLES = {
    "observation.images.head_left": "scene",
    "observation.images.wrist_left": "wrist_left",
    "observation.images.wrist_right": "wrist_right",
}
SEASON_RE = re.compile(r"season_(.+?)(?:_train)?$")
SUBTASK_TEXT_COLUMNS = ("text", "subtask", "subtask_text", "description")
FLAT_EPS = 1e-9


def closure_index(ds: LeRobotV3) -> int:
    """Return the observation.state dimension used as the closure proxy (right index MCP)."""
    for i, name in enumerate(ds.feature_names(STATE_KEY)):
        low = name.lower()
        if all(h in low for h in CLOSURE_NAME_HINTS):
            return i
    return RIGHT_INDEX_MCP_FALLBACK_INDEX


@register_source("sharpadex")
class SharpadexAdapter:
    """Adapter for SharpaDex v1.0 seasons in LeRobot v3 layout."""

    name = "sharpadex"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._index: dict[str, tuple[Path, int]] = {}
        self._datasets: dict[Path, LeRobotV3] = {}

    # -- discovery --------------------------------------------------------------------
    def discover(self, root: Path) -> list[str]:
        """Return every ``<task>/season_<id>_train/<episode_index>`` under ``root``."""
        self._root = root
        self._index = {}
        self._datasets = {}
        for export in sorted(root.glob(f"*/season_*/{EXPORT_DIR}")):
            ds = self._dataset(export)
            rel = export.parent.relative_to(root).as_posix()
            for idx in ds.episode_indices():
                self._index[f"{rel}/{idx}"] = (export, idx)
        return list(self._index)

    def _dataset(self, export: Path) -> LeRobotV3:
        if export not in self._datasets:
            self._datasets[export] = LeRobotV3(export)
        return self._datasets[export]

    def _ensure(self) -> None:
        if not self._index:
            self.discover(self._root)

    @staticmethod
    def _season(export: Path) -> str:
        m = SEASON_RE.match(export.parent.name)
        return m.group(1) if m else export.parent.name

    # -- loading ----------------------------------------------------------------------
    def load(self, episode_id: str) -> Episode:
        """Load one episode on the episode clock (zero at the first scene frame)."""
        self._ensure()
        export, idx = self._index[episode_id]
        ds = self._dataset(export)
        rec = ds.record(idx)
        df = ds.frames(idx)
        t_raw = np.asarray(df["timestamp"], dtype=np.float64)
        offset = float(t_raw[0])
        t = t_raw - offset
        n = len(df)

        tactile = column_array(df, TACTILE_KEY)
        pad = PadArmForce(
            f_a=tactile[:, RIGHT_THUMB_FZ].tolist(), f_b=tactile[:, RIGHT_INDEX_FZ].tolist()
        )
        force = ForceStream(channel="pad", arms={"R": pad}, rate_hz=ds.fps, t=t.tolist())

        ci = closure_index(ds)
        q = column_array(df, STATE_KEY)[:, ci]
        span = float(q.max() - q.min())
        closure = (q - q.min()) / span if span > FLAT_EPS else np.zeros_like(q)
        gripper = GripperStream(arms={"R": closure.tolist()}, rate_hz=ds.fps, t=t.tolist())

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
        task_text = str(tasks[0]) if tasks else export.parent.parent.name.replace("_", " ")
        season = self._season(export)
        meta: dict[str, Any] = {
            "subtasks": self._subtasks(ds, df),
            "season": season,
            "task_dir": export.parent.parent.name,
            "episode_index": idx,
            "clock_offset_s": offset,
            "video_from_s": video_from,
            "closure_state_index": ci,
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=season,
            task=strip_outcome_words(task_text),
            duration_s=float(t[-1]) if n else 0.0,
            cameras=cameras,
            force=force,
            gripper=gripper,
            labels=[],
            license=LICENSE,
            meta=meta,
        )

    @staticmethod
    def _subtasks(ds: LeRobotV3, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Run-length segments of the per-frame subtask_index, joined to meta/subtasks.parquet."""
        if "subtask_index" not in df.columns:
            return []
        table = pd.DataFrame()
        path = ds.root / "meta" / "subtasks.parquet"
        if path.exists():
            table = pq.read_table(path).to_pandas()
        text_col = next((c for c in SUBTASK_TEXT_COLUMNS if c in table.columns), None)
        by_index = table.set_index("subtask_index") if "subtask_index" in table.columns else table
        sub = np.asarray(df["subtask_index"], dtype=np.int64)
        out: list[dict[str, Any]] = []
        start = 0
        for i in range(1, len(sub) + 1):
            if i == len(sub) or sub[i] != sub[start]:
                key = int(sub[start])
                row = by_index.loc[key] if key in by_index.index else None
                text = str(row[text_col]) if row is not None and text_col else str(key)
                skill = (
                    str(row["task_skill"])
                    if row is not None and "task_skill" in table.columns
                    else ""
                )
                out.append(
                    {"start_step": start, "end_step": i - 1, "text": text, "task_skill": skill}
                )
                start = i
        return out

    # -- splits and labels ------------------------------------------------------------
    def splits(self) -> dict[str, list[str]]:
        """Return train/val/test by season (a season never straddles two splits)."""
        self._ensure()
        return assign_splits(
            {eid: self._season(e) for eid, (e, _) in self._index.items()}, self.cfg
        )

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return no labels: SharpaDex has no success/failure flag."""
        return []
