"""Source adapter for Baxter policy rollouts published as a LeRobot v3 dataset (eval only).

Owns: how one rollout of ``OG7407/baxter-cube-pick-left-rollouts`` (and datasets laid out the
same way) becomes an Episode. The recorder writes one instrumented gripper (the left one, two
FSR pads ``L_a`` and ``L_b``); its pads become one pad arm whose letter is chosen by
``SourceConfig.baxter_rollouts_arm`` so that adapters fine-tuned on HRDexDB, which only ever
saw ``pad R`` lines, read the block in the form they were trained on.
Must never: filter episodes in discover(), write to disk, be used for training (eval only), or
leak layout details to other modules.

On-disk contract (LeRobot v3 as written by the Baxter rollout recorder):

- ``<root>/meta/info.json``: ``fps`` and ``features`` (the ``observation.state`` names locate
  the gripper column, the ``observation.tactile`` names locate the pad columns, video keys are
  the ``dtype: video`` features).
- ``<root>/meta/episodes/chunk-*/file-*.parquet``: one row per episode with ``episode_index``,
  ``length``, ``data/chunk_index``, ``data/file_index`` and, per video key,
  ``videos/<key>/chunk_index``, ``file_index`` and ``from_timestamp`` (episodes are
  concatenated in one mp4 per chunk; ``from_timestamp`` is the in-file start).
- ``<root>/data/chunk-*/file-*.parquet``: one row per frame with ``timestamp`` (seconds from
  the episode's first frame), ``frame_index``, ``episode_index`` and ``observation.state``
  (joint angles plus the gripper position, 0.04 fully closed .. 0.96 open).
- ``<root>/raw/ep%04d/tactile_200hz.csv``: the native 200 Hz FSR counts, columns ``t`` (seconds
  on the same clock as the frame timestamps), ``teensy_micros`` and ``counts_<pad>``.
- ``<root>/raw/ep%04d/meta.json``: ``task``, ``label`` ("success" / "fail"), ``label_source``,
  ``stop_reason``, ``note``, ``started``, ``policy``.
- ``<root>/raw/index.json``: episode index -> rollout path (``rollouts/<YYYYMMDD>/ep0001``);
  the date is the session.
- ``<root>/labels.json`` (optional override, written by the preparation script from the
  operator's statement): ``{"<episode_id>": {"outcome": 1..5, "note": str}}``. An entry
  replaces the recorder's label and carries origin ``human``.

Clock: frame timestamps and ``t`` both start at the episode's first frame (``clock_offset_s``
is 0). Force in raw ADC counts (the pads are uncalibrated); normalization removes the resting
offset. Gripper closure ``g`` is ``(open - position) / (open - closed)`` clipped to [0, 1], so a
gripper stopped by an object reads about 0.6 and an empty close about 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Arm,
    ArmForce,
    Episode,
    ForceStream,
    GripperStream,
    Label,
    PadArmForce,
    VideoRef,
)
from tactile_flora.sources._eval_common import rate_from_times
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "apache-2.0"
FORCE_RATE_HZ = 200.0
GRIPPER_OPEN = 0.96  # recorder's normalized Baxter gripper position when open
GRIPPER_CLOSED = 0.04  # fully closed on nothing
CAMERA_ROLES = {"head": "scene", "wrist": "wrist_left"}
_CSV_PREFIX = "counts_"


def _index_of(episode_id: str) -> int:
    return int(episode_id.removeprefix("ep"))


def _episode_id(index: int) -> str:
    return f"ep{index:04d}"


@register_source("baxter_rollouts")
class BaxterRolloutsAdapter:
    """Adapter for LeRobot v3 Baxter rollouts with FSR fingertips (eval only)."""

    name = "baxter_rollouts"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._info: dict[str, Any] | None = None
        self._episodes: pd.DataFrame | None = None
        self._frames: dict[tuple[int, int], pd.DataFrame] = {}

    # ---- discovery -------------------------------------------------------------------------

    def discover(self, root: Path) -> list[str]:
        """Return every episode id listed in ``meta/episodes`` (no filtering)."""
        self._root = Path(root)
        self._info = None
        self._episodes = None
        self._frames = {}
        return [_episode_id(int(i)) for i in self._table()["episode_index"]]

    def _load_info(self) -> dict[str, Any]:
        info = self._info
        if info is None:
            info = json.loads((self._root / "meta" / "info.json").read_text())
            self._info = info
        return info

    def _table(self) -> pd.DataFrame:
        if self._episodes is None:
            files = sorted(self._root.glob("meta/episodes/chunk-*/file-*.parquet"))
            if not files:
                raise FileNotFoundError(f"no meta/episodes parquet under {self._root}")
            table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            self._episodes = table.sort_values("episode_index").reset_index(drop=True)
        return self._episodes

    def _row(self, index: int) -> pd.Series:
        table = self._table()
        hits = table[table["episode_index"] == index]
        if hits.empty:
            raise KeyError(f"episode {index} not in meta/episodes")
        return hits.iloc[0]

    def _frame_table(self, chunk: int, file: int) -> pd.DataFrame:
        key = (chunk, file)
        if key not in self._frames:
            path = self._root / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"
            self._frames[key] = pd.read_parquet(path)
        return self._frames[key]

    def _raw_dir(self, index: int) -> Path:
        return self._root / "raw" / _episode_id(index)

    def _raw_meta(self, index: int) -> dict[str, Any]:
        path = self._raw_dir(index) / "meta.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    def _session(self, index: int) -> str:
        path = self._root / "raw" / "index.json"
        if path.is_file():
            entry = json.loads(path.read_text()).get(str(index))
            if entry:
                parts = Path(str(entry)).parts
                if len(parts) >= 2:
                    return parts[-2]
        started = str(self._raw_meta(index).get("started", ""))
        return started[:10].replace("-", "") or "unknown"

    # ---- loading ---------------------------------------------------------------------------

    def _video_keys(self) -> list[str]:
        feats = self._load_info()["features"]
        return [k for k, v in feats.items() if v.get("dtype") == "video"]

    def _names(self, feature: str) -> list[str]:
        names = self._load_info()["features"][feature].get("names") or []
        return [str(n) for n in names]

    def load(self, episode_id: str) -> Episode:
        """Load one rollout on the episode clock (0 at its first frame)."""
        index = _index_of(episode_id)
        info = self._load_info()
        fps = float(info["fps"])
        row = self._row(index)
        n_frames = int(row["length"])
        frames = self._frame_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        frames = frames[frames["episode_index"] == index].sort_values("frame_index")
        if len(frames) != n_frames:
            raise ValueError(
                f"{episode_id}: {len(frames)} frame rows, episodes table says {n_frames}"
            )
        ft = frames["timestamp"].to_numpy(dtype=np.float64)
        ft = ft - ft[0]

        # Gripper closure from the state column named by the recorder.
        state_names = self._names("observation.state")
        gripper_cols = [i for i, n in enumerate(state_names) if n.endswith("_gripper")]
        if not gripper_cols:
            raise ValueError("observation.state has no *_gripper column")
        state = np.stack([np.asarray(v, dtype=np.float64) for v in frames["observation.state"]])
        position = state[:, gripper_cols[0]]
        closure = np.clip((GRIPPER_OPEN - position) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0, 1.0)

        # Native 200 Hz FSR counts; the frame-rate copy in the parquet is the fallback.
        pad_names = self._names("observation.tactile")
        side = state_names[gripper_cols[0]][0].upper()  # "left_gripper" -> pads "L_a", "L_b"
        pads = [n for n in pad_names if n.upper().startswith(f"{side}_")]
        if len(pads) < 2:
            raise ValueError(f"observation.tactile has no two {side}_* pads: {pad_names}")
        csv = self._raw_dir(index) / "tactile_200hz.csv"
        if csv.is_file():
            tdf = pd.read_csv(csv)
            t = tdf["t"].to_numpy(dtype=np.float64)
            f_a = tdf[f"{_CSV_PREFIX}{pads[0]}"].to_numpy(dtype=np.float64)
            f_b = tdf[f"{_CSV_PREFIX}{pads[1]}"].to_numpy(dtype=np.float64)
            rate = rate_from_times(t, FORCE_RATE_HZ)
            force_source = "raw/tactile_200hz.csv"
        else:
            tactile = np.stack(
                [np.asarray(v, dtype=np.float64) for v in frames["observation.tactile"]]
            )
            t = ft
            f_a = tactile[:, pad_names.index(pads[0])]
            f_b = tactile[:, pad_names.index(pads[1])]
            rate = fps
            force_source = "observation.tactile/frame_aligned"
        arm: Arm = self.cfg.baxter_rollouts_arm
        arms: dict[Arm, ArmForce] = {arm: PadArmForce(f_a=f_a.tolist(), f_b=f_b.tolist())}
        gripper = GripperStream(
            arms={arm: closure.tolist()}, rate_hz=rate_from_times(ft, fps), t=ft.tolist()
        )

        cameras: dict[str, VideoRef] = {}
        video_from: dict[str, float] = {}
        wanted = list(self.cfg.baxter_rollouts_cameras)
        for key in self._video_keys():
            short = key.rsplit(".", 1)[-1]
            if short not in wanted:
                continue
            role = CAMERA_ROLES.get(short, short)
            chunk = int(row[f"videos/{key}/chunk_index"])
            file = int(row[f"videos/{key}/file_index"])
            path = self._root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"missing video {path}")
            cameras[role] = VideoRef(path=str(path.resolve()), fps=fps, n_frames=n_frames)
            video_from[role] = float(row[f"videos/{key}/from_timestamp"])
        if "scene" not in cameras:
            raise FileNotFoundError(f"{episode_id}: no scene (head) camera among {wanted}")

        meta = self._raw_meta(index)
        task = str(meta.get("task") or "pick up the object")
        obj = task.removeprefix("pick up the ").strip() or "object"
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=self._session(index),
            task=strip_outcome_words(task),
            duration_s=n_frames / fps,
            cameras=cameras,
            force=ForceStream(channel="pad", arms=arms, rate_hz=rate, t=t.tolist()),
            gripper=gripper,
            labels=self.native_labels(episode_id),
            license=LICENSE,
            meta={
                "clock_offset_s": 0.0,
                "video_from_s": video_from,
                "force_source": force_source,
                "force_units": "raw FSR ADC counts (uncalibrated), resting offset about 92",
                "tactile_channels": pads,
                "instrumented_arm": arm,
                "gripper_convention": "closure = (open - position) / (open - closed)",
                "episode_index": index,
                "object": obj,
                "fps": fps,
                "recorder_label": meta.get("label"),
                "recorder_label_source": meta.get("label_source"),
                "stop_reason": meta.get("stop_reason"),
                "note": meta.get("note"),
                "started": meta.get("started"),
                "policy": meta.get("policy"),
            },
        )

    def splits(self) -> dict[str, list[str]]:
        """Eval only: every episode is in split "benchmark"."""
        return {"benchmark": self.discover(self._root)}

    def native_labels(self, episode_id: str) -> list[Label]:
        """Outcome 5 (success) or 1 (failure); ``labels.json`` overrides the recorder's label."""
        index = _index_of(episode_id)
        override = self._root / "labels.json"
        if override.is_file():
            entry = json.loads(override.read_text()).get(episode_id)
            if entry and entry.get("outcome") is not None:
                return [Label(kind="outcome", origin="human", outcome=int(entry["outcome"]))]
        label = self._raw_meta(index).get("label")
        if label is None:
            return []
        success = str(label).lower().startswith("succ")
        return [Label(kind="outcome", origin="source", outcome=5 if success else 1)]
