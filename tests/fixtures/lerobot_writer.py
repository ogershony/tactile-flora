"""Shared helper that writes a tiny LeRobot v3 dataset directory for fixture generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_FRAMES = 10
VIDEO_SIZE = 32


def write_mp4(path: Path, fps: int, n_frames: int = N_FRAMES, shade: int = 0) -> None:
    """Write a tiny synthetic mp4 (flat colour that changes per frame)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = VIDEO_SIZE  # type: ignore[attr-defined]
        stream.pix_fmt = "yuv420p"  # type: ignore[attr-defined]
        for i in range(n_frames):
            img = np.full((VIDEO_SIZE, VIDEO_SIZE, 3), (shade + 10 * i) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):  # type: ignore[attr-defined]
                out.mux(packet)
        for packet in stream.encode():  # type: ignore[attr-defined]
            out.mux(packet)


def write_lerobot_v3(
    root: Path,
    *,
    fps: int,
    columns: dict[str, np.ndarray],
    episode_meta: dict[str, Any],
    video_keys: list[str],
    feature_names: dict[str, list[str]] | None = None,
    stats: dict[str, Any] | None = None,
    n_frames: int = N_FRAMES,
) -> None:
    """Write one episode (index 0) as a LeRobot v3 dataset under ``root``.

    ``columns`` maps column name to an (n_frames, D) or (n_frames,) array. ``episode_meta`` is
    merged into the single row of meta/episodes (e.g. tasks, task_success, hsr_id).
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "episode_index": [0] * n_frames,
        "frame_index": list(range(n_frames)),
        "timestamp": [i / fps for i in range(n_frames)],
        "index": list(range(n_frames)),
    }
    features: dict[str, Any] = {}
    for name, arr in columns.items():
        data[name] = arr.tolist()
        dim = 1 if arr.ndim == 1 else arr.shape[1]
        features[name] = {
            "dtype": "float32",
            "shape": [dim],
            "names": (feature_names or {}).get(name),
        }
    table = pa.table({k: pa.array(v) for k, v in data.items()})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    row: dict[str, Any] = {
        "episode_index": 0,
        "length": n_frames,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": n_frames,
        **episode_meta,
    }
    for k, key in enumerate(video_keys):
        write_mp4(root / "videos" / key / "chunk-000" / "file-000.mp4", fps, n_frames, shade=40 * k)
        row[f"videos/{key}/chunk_index"] = 0
        row[f"videos/{key}/file_index"] = 0
        row[f"videos/{key}/from_timestamp"] = 0.0
        row[f"videos/{key}/to_timestamp"] = n_frames / fps
        features[key] = {"dtype": "video", "shape": [VIDEO_SIZE, VIDEO_SIZE, 3], "names": None}
    pq.write_table(
        pa.table({k: pa.array([v]) for k, v in row.items()}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": 1,
        "total_frames": n_frames,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=1))
    if stats is not None:
        (root / "meta" / "stats.json").write_text(json.dumps(stats))
