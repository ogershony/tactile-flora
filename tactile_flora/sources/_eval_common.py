"""Small helpers shared by the eval-only adapters (hrdexdb, softvtbench, baxter).

Owns: probing an mp4 into a VideoRef, nearest-sample alignment of one time base to another and
estimating a sample rate from timestamps.
Must never: know any dataset layout, filter episodes, or write to disk.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from tactile_flora.schema import VideoRef


def probe_video(path: Path) -> VideoRef:
    """Return a VideoRef for ``path`` by reading the container header (decodes if no count)."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        n_frames = int(stream.frames)
        if n_frames <= 0:
            n_frames = sum(1 for _ in container.decode(video=0))
    finally:
        container.close()
    return VideoRef(path=str(path), fps=fps, n_frames=n_frames)


def nearest_index(src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """For every time in ``dst_t`` return the index of the nearest time in sorted ``src_t``."""
    if len(src_t) == 0:
        raise ValueError("cannot align to an empty time base")
    right = np.clip(np.searchsorted(src_t, dst_t), 1, max(len(src_t) - 1, 1))
    left = right - 1
    right = np.minimum(right, len(src_t) - 1)
    pick_right = np.abs(src_t[right] - dst_t) < np.abs(dst_t - src_t[left])
    return np.where(pick_right, right, left)


def rate_from_times(t: np.ndarray, default: float) -> float:
    """Median-based sample rate of ``t`` in Hz; ``default`` when it cannot be estimated."""
    if len(t) < 2:
        return default
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else default
