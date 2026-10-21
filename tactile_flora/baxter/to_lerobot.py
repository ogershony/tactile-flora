"""Converting recorded Baxter episode directories to a LeRobot v3 dataset.

Owns: converting the directories written by ros/recorder.py into the on-disk contract documented
in sources/baxter.py (data/chunk-000, force/, videos/, meta/episodes.jsonl). The per-frame
``observation.force`` is made by force/resample.py (median in a window around each frame), the
native 200 Hz force is copied unshifted.
Must never: resample force itself, label, or write labels.json (``tf baxter label`` owns it).

Sync gate: after converting, run ``tf verify --config configs/sources/baxter.yaml --source
baxter``. It runs the standard sync check (gripper closure vs force onset, lag under 100 ms; the
source-level median bound is 100 ms) and a Baxter recording only counts once that passes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

from tactile_flora.config import ResampleConfig
from tactile_flora.force.resample import resample
from tactile_flora.schema import Arm, ArmForce, ForceStream, PadArmForce

ROLES = ("scene", "wrist_left", "wrist_right")
FRAME_RATE_HZ = 30.0
CHANNELS = ("L_a", "L_b", "R_a", "R_b")


def _frame_count(path: Path) -> int:
    container = av.open(str(path))
    try:
        return sum(1 for _ in container.decode(video=0))
    finally:
        container.close()


def _nearest_values(t: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.zeros((len(query), values.shape[1]))
    i = np.clip(np.searchsorted(t, query), 1, max(t.size - 1, 1))
    left = np.maximum(i - 1, 0)
    right = np.minimum(i, t.size - 1)
    pick_right = np.abs(t[right] - query) < np.abs(query - t[left])
    return values[np.where(pick_right, right, left)]


def _frame_force(
    t_force: np.ndarray, force: np.ndarray, frame_times: np.ndarray, cfg: ResampleConfig
) -> np.ndarray:
    """(n_frames, 4) newtons: per-channel windowed median via force.resample.resample."""
    out = np.zeros((len(frame_times), 4))
    for half in (0, 1):  # resample's F is the f_a + f_b sum; feed one channel at a time
        arms: dict[Arm, ArmForce] = {}
        for ai, arm in enumerate(("L", "R")):
            col = force[:, 2 * ai + half].tolist()
            arms[arm] = PadArmForce(f_a=col, f_b=[0.0] * len(col))
        stream = ForceStream(channel="pad", arms=arms, rate_hz=200.0, t=t_force.tolist())
        feats = resample(stream, None, frame_times.tolist(), cfg)
        for ai, arm in enumerate(("L", "R")):
            f = np.asarray(feats.arms[arm].F, dtype=np.float64)
            for k in range(1, len(f)):  # empty window: carry the previous value forward
                if np.isnan(f[k]):
                    f[k] = f[k - 1]
            out[:, 2 * ai + half] = np.nan_to_num(f)
    return out


def convert_episode(src: Path, out: Path, cfg: ResampleConfig) -> dict[str, Any]:
    """Convert one episode directory into ``out``; returns its meta/episodes.jsonl row."""
    meta: dict[str, Any] = json.loads((src / "meta.json").read_text())
    idx = int(meta["episode_index"])
    stem = f"episode_{idx:06d}"
    n = _frame_count(src / "scene.mp4")
    t0 = float(meta.get("video_t0_s", 0.0))
    frame_t = np.arange(n) / FRAME_RATE_HZ

    fdf = pd.read_parquet(src / "force.parquet")
    t_force = fdf["t"].to_numpy(dtype=np.float64)
    force = fdf[list(CHANNELS)].to_numpy(dtype=np.float64)
    obs_force = _frame_force(t_force - t0, force, frame_t, cfg)

    sdf = pd.read_parquet(src / "state.parquet")
    st = sdf["t"].to_numpy(dtype=np.float64) - t0
    state_cols = ["L_position", "L_command", "R_position", "R_command"]
    state = _nearest_values(st, sdf[state_cols].to_numpy(dtype=np.float64), frame_t)

    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "force").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": frame_t,
            "frame_index": np.arange(n),
            "episode_index": np.full(n, idx, dtype=int),
            "observation.state": [list(map(float, r)) for r in state],
            "observation.gripper_left": state[:, 0],
            "observation.gripper_right": state[:, 2],
            "observation.force": [list(map(float, r)) for r in obs_force],
        }
    ).to_parquet(out / "data" / "chunk-000" / f"{stem}.parquet")
    shutil.copyfile(src / "force.parquet", out / "force" / f"{stem}.parquet")
    for role in ROLES:
        video = src / f"{role}.mp4"
        if video.is_file():
            dst = out / "videos" / f"observation.images.{role}" / f"{stem}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video, dst)

    keys = [
        "episode_index",
        "object",
        "firmness",
        "policy",
        "occlusion",
        "operator_notes",
        "recording_day",
        "calibration_version",
        "task",
        "video_t0_s",
    ]
    return {k: meta[k] for k in keys if k in meta}


def convert(episodes_dir: Path, out: Path, cfg: ResampleConfig) -> Path:
    """Convert every ``episode_*`` directory of ``episodes_dir`` into the dataset at ``out``.

    Rewrites meta/episodes.jsonl from all episodes present (idempotent). Never touches labels.json.
    """
    rows: dict[int, dict[str, Any]] = {}
    for src in sorted(Path(episodes_dir).glob("episode_*")):
        if (src / "meta.json").is_file():
            row = convert_episode(src, out, cfg)
            rows[int(row["episode_index"])] = row
    if not rows:
        raise RuntimeError(f"no episodes found under {episodes_dir}")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(rows[i]) + "\n" for i in sorted(rows))
    )
    return out
