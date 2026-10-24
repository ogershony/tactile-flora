"""Write a tiny synthetic REASSEMBLE LeRobot v3 port in the real layout (run from repo root).

Three episode rows: 0 (train, one failed segment), 1 (test, an idle segment flagged failed) and
2 (unassigned) whose data parquet is deliberately absent, as in a partial download. Episodes
0 and 1 share one concatenated mp4 per camera, so episode 1 has a non-zero in-file start.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tests.fixtures.lerobot_writer import write_lerobot_v3, write_mp4

HERE = Path(__file__).parent
FPS = 30
TASKS = ["No action.", "Pick square peg 3.", "Insert square peg 3."]
IDLE, PICK, INSERT = 0, 1, 2
VIDEO_KEYS = [
    "observation.images.hand",
    "observation.images.hama1",
    "observation.images.hama2",
]
RECORDINGS: dict[int, tuple[str, str]] = {
    0: ("2025-01-11-14-45-48", "train"),
    1: ("2025-01-13-11-16-17", "test"),
    2: ("2025-01-10-16-17-40", "unassigned"),
}
STATS = {"observation.state.gripper_position": {"min": [0.0, 0.0], "max": [0.04, 0.04]}}
N0, N1, N2 = 30, 12, 8
# (segment.index, task_index, segment.success, n_frames) in frame order
SEG0 = [(0, IDLE, True, 6), (1, PICK, True, 9), (2, INSERT, False, 9), (3, IDLE, True, 6)]
SEG1 = [(0, IDLE, False, 3), (1, PICK, True, 9)]


RAW_BIAS = np.array([2.0, -0.5, -13.5])  # tool weight in the raw sensor reading, N
BASE_TORQUE_OFFSET = np.array([0.0, 6.0, 0.0])  # standing offset of observation.torque.base, N m


def columns(
    segs: list[tuple[int, int, bool, int]],
    grasp: tuple[int, int],
    ramp: int,
    noise: float = 0.0,
    seed: int = 3,
    compensated: bool = True,
) -> dict[str, np.ndarray]:
    """Per-frame columns: a 10 N fz plateau and a closed gripper over frames grasp[0]..grasp[1].

    ``observation.force.base`` is the clean plateau; ``observation.force`` adds ``RAW_BIAS`` as
    the port does. With ``compensated`` False the ``.base`` columns are omitted (fallback path).
    """
    n = sum(k for _, _, _, k in segs)
    seg_index = np.concatenate([np.full(k, i) for i, _, _, k in segs]).astype(np.int64)
    task = np.concatenate([np.full(k, ti) for _, ti, _, k in segs]).astype(np.int64)
    ok = np.concatenate([np.full(k, s) for _, _, s, k in segs]).astype(bool)
    f = np.arange(n, dtype=np.float64)
    a, b = grasp
    frac = np.interp(f, [a - ramp, a, b, b + ramp], [0, 1, 1, 0])
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, noise, (n, 2)) if noise > 0 else np.zeros((n, 2))
    force = np.stack(
        [frac * 2 + jitter[:, 0], np.zeros(n), frac * 10 + jitter[:, 1]], axis=1
    ).astype(np.float32)
    width = (0.04 * (1 - frac)).astype(np.float32)
    out = {
        "observation.force": (force + RAW_BIAS).astype(np.float32),
        "observation.torque": np.zeros((n, 3), dtype=np.float32),
        "observation.state.gripper_position": np.stack([width, width], axis=1),
        "segment.success": ok,
        "segment.index": seg_index,
        "task_index": task,
    }
    if compensated:
        out["observation.force.base"] = force
        out["observation.torque.base"] = np.tile(BASE_TORQUE_OFFSET, (n, 1)).astype(np.float32)
    return out


def _append_episode(
    root: Path, idx: int, cols: dict[str, np.ndarray] | None, n: int, from_frame: int
) -> None:
    """Add episode ``idx`` to the episodes table (and its data file when ``cols`` is given)."""
    if cols is not None:
        data: dict[str, Any] = {
            "episode_index": [idx] * n,
            "frame_index": list(range(n)),
            "timestamp": [i / FPS for i in range(n)],
            "index": [from_frame + i for i in range(n)],
        }
        for name, arr in cols.items():
            data[name] = arr.tolist()
        pq.write_table(
            pa.table({k: pa.array(v) for k, v in data.items()}),
            root / "data" / "chunk-000" / f"file-{idx:03d}.parquet",
        )
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    rows = pq.read_table(ep_path).to_pylist()
    row = dict(rows[0])
    row.update(
        {
            "episode_index": idx,
            "length": n,
            "data/file_index": idx,
            "dataset_from_index": from_frame,
            "dataset_to_index": from_frame + n,
        }
    )
    for key in VIDEO_KEYS:
        row[f"videos/{key}/from_timestamp"] = from_frame / FPS
        row[f"videos/{key}/to_timestamp"] = (from_frame + n) / FPS
    rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), ep_path)


def _write_meta(root: Path, recordings: dict[int, tuple[str, str]], total_frames: int) -> None:
    pd.DataFrame(
        {"task_index": list(range(len(TASKS)))}, index=pd.Index(TASKS, name="task")
    ).to_parquet(root / "meta" / "tasks.parquet")
    (root / "meta" / "splits.json").write_text(
        json.dumps(
            {str(i): {"recording": r, "split": s} for i, (r, s) in recordings.items()}, indent=1
        )
    )
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.update(
        {
            "total_episodes": len(recordings),
            "total_frames": total_frames,
            "total_tasks": len(TASKS),
            "robot_type": "franka",
        }
    )
    for name, dtype in (
        ("segment.success", "bool"),
        ("segment.index", "int64"),
        ("task_index", "int64"),
    ):
        info["features"].setdefault(name, {"shape": [1], "names": None})["dtype"] = dtype
    info_path.write_text(json.dumps(info, indent=1))


def main() -> None:
    for d in ("data", "meta", "videos"):
        shutil.rmtree(HERE / d, ignore_errors=True)
    tasks = [TASKS[PICK], TASKS[INSERT], TASKS[IDLE]]
    write_lerobot_v3(
        HERE,
        fps=FPS,
        columns=columns(SEG0, (6, 23), ramp=3),
        episode_meta={"tasks": tasks},
        video_keys=VIDEO_KEYS,
        stats=STATS,
        n_frames=N0,
    )
    _append_episode(HERE, 1, columns(SEG1, (3, 11), ramp=3), N1, from_frame=N0)
    _append_episode(HERE, 2, None, N2, from_frame=N0 + N1)
    for k, key in enumerate(VIDEO_KEYS):
        write_mp4(HERE / "videos" / key / "chunk-000" / "file-000.mp4", FPS, N0 + N1, shade=40 * k)
    _write_meta(HERE, RECORDINGS, N0 + N1 + N2)


def write_long(root: Path, compensated: bool = True, recording: str = RECORDINGS[0][0]) -> None:
    """A 16 s single-recording port (30 Hz) with one failed segment under ``root``."""
    shutil.rmtree(root, ignore_errors=True)
    n = 480
    segs = [(0, IDLE, True, 60), (1, PICK, True, 180), (2, INSERT, False, 180), (3, IDLE, True, 60)]
    write_lerobot_v3(
        root,
        fps=FPS,
        columns=columns(segs, (90, 330), ramp=30, noise=0.04, compensated=compensated),
        episode_meta={"tasks": [TASKS[PICK], TASKS[INSERT], TASKS[IDLE]]},
        video_keys=VIDEO_KEYS,
        stats=STATS,
        n_frames=n,
    )
    _write_meta(root, {0: (recording, RECORDINGS[0][1])}, n)


if __name__ == "__main__":
    main()
