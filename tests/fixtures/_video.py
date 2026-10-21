"""Tiny mp4 writer shared by the fixture generators (PyAV, generated frames only)."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def write_mp4(path: Path, n: int = 10, fps: int = 30, size: int = 32) -> None:
    """Write ``n`` tiny generated grey-ramp frames to an mp4 at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size  # type: ignore[attr-defined]
        stream.pix_fmt = "yuv420p"  # type: ignore[attr-defined]
        for i in range(n):
            img = np.full((size, size, 3), 20 * i, dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):  # type: ignore[attr-defined]
                out.mux(pkt)
        for pkt in stream.encode():  # type: ignore[attr-defined]
            out.mux(pkt)
