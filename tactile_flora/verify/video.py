"""Video frame readers behind the FrameReader protocol (schema.FrameReader).

Owns: PyAVFrameReader, the production reader that decodes episode videos with PyAV.
Must never: decide whether an episode is accepted, or be imported by anything outside verify/.
Tests use an in-memory reader (tests/fixtures/synthetic_video.py) instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tactile_flora.schema import Episode


class PyAVFrameReader:
    """Decode frames from the video files named by ``Episode.cameras[role].path``.

    Relative paths resolve against ``root``. Frames are decoded sequentially from the start
    and cached for the last opened file so first/middle/last reads of one video stay cheap.
    """

    def __init__(self, root: Path | str = ".") -> None:
        """Remember the directory relative video paths resolve against."""
        self.root = Path(root)
        self._count: dict[Path, int] = {}

    def _path(self, episode: Episode, role: str) -> Path:
        p = Path(episode.cameras[role].path)
        return p if p.is_absolute() else self.root / p

    def frame_count(self, episode: Episode, role: str) -> int:
        """Decode the whole stream and return the number of frames (raises if undecodable)."""
        import av

        path = self._path(episode, role)
        if path in self._count:
            return self._count[path]
        n = 0
        with av.open(str(path)) as container:
            for _ in container.decode(video=0):
                n += 1
        self._count[path] = n
        return n

    def read_frame(self, episode: Episode, role: str, index: int) -> np.ndarray:
        """Return frame ``index`` as an (H, W, 3) uint8 RGB array."""
        import av

        path = self._path(episode, role)
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i == index:
                    return np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
        raise IndexError(f"frame {index} not found in {path}")
