"""The FrameReader implementations: PyAV (mp4), zarr JPEG (PHANTOM), synthetic (tests), auto.

Owns: PyAVReader, ZarrJpegReader, SyntheticReader, AutoReader, SourceRootReader, smooth_image.
Must never: compose mosaics (video/mosaic.py does), decide whether an episode is accepted, or
read anything but the files a ``VideoRef`` names.

Protocol (schema.FrameReader): ``frame_count(episode, role)``, ``read(episode, role, index)``
-> (H, W, 3) uint8 RGB and ``read_many(episode, role, indices)`` -> list of such arrays, in the
order asked. ``index`` is an EPISODE frame index (0 = first frame of the episode clock).

LeRobot v3 mp4s hold many episodes back to back; ``meta["video_from_s"][role]`` is the in-file
start of this episode, so episode frame ``i`` is in-file frame ``round((video_from_s + i / fps)
* fps)``. PHANTOM scene video is a zarr group (``VideoRef.path`` is a directory ending in
``camera_scene_color``) whose ``data`` array holds one JPEG byte string per frame.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tactile_flora.schema import Episode
from tactile_flora.video.zarr_compat import open_group

ZARR_SUFFIX = "camera_scene_color"
_RESEEK_GAP = 64  # decode forward through gaps up to this many frames, otherwise seek again
SIZE = 64


def video_path(root: Path, episode: Episode, role: str) -> Path:
    """Resolve ``episode.cameras[role].path`` (relative paths hang off ``root``)."""
    p = Path(episode.cameras[role].path)
    return p if p.is_absolute() else root / p


def file_offset_frames(episode: Episode, role: str) -> int:
    """In-file index of the episode's frame 0: round(video_from_s * fps), 0 when absent."""
    start = float(episode.meta.get("video_from_s", {}).get(role, 0.0) or 0.0)
    return round(start * episode.cameras[role].fps)


def _file_index(episode: Episode, role: str, i: int) -> int:
    fps = episode.cameras[role].fps
    start = float(episode.meta.get("video_from_s", {}).get(role, 0.0) or 0.0)
    return round((start + i / fps) * fps)


class _Base:
    """read / read_many defaults on top of ``read_many`` implementations."""

    def read_many(self, episode: Episode, role: str, indices: Sequence[int]) -> list[np.ndarray]:
        """Frames at ``indices`` (episode indices), one array per index, order preserved."""
        return [self.read(episode, role, i) for i in indices]

    def read(self, episode: Episode, role: str, index: int) -> np.ndarray:
        """Return episode frame ``index`` as an (H, W, 3) uint8 RGB array."""
        return self.read_many(episode, role, [index])[0]


class PyAVReader(_Base):
    """Decode mp4 frames with PyAV, honouring ``meta["video_from_s"]`` for concatenated files.

    Each ``read_many`` call opens the file once, seeks to the keyframe before the first wanted
    frame and decodes forward, seeking again only across gaps longer than 64 frames.
    """

    def __init__(self, root: Path | str = ".") -> None:
        """Remember the directory relative video paths resolve against."""
        self.root = Path(root)
        self._total: dict[Path, int] = {}

    def frame_count(self, episode: Episode, role: str) -> int:
        """Decode the whole stream and return this episode's frame count (raises if undecodable).

        For a concatenated file the count is the frames after the episode start, capped at the
        declared ``n_frames``.
        """
        import av

        path = video_path(self.root, episode, role)
        if path not in self._total:
            n = 0
            with av.open(str(path)) as container:
                for _ in container.decode(video=0):
                    n += 1
            self._total[path] = n
        total = self._total[path]
        start = file_offset_frames(episode, role)
        if start == 0:
            return total
        return max(0, min(total - start, episode.cameras[role].n_frames))

    def read_many(self, episode: Episode, role: str, indices: Sequence[int]) -> list[np.ndarray]:
        """Decode the requested frames in one pass over the file."""
        import av

        path = video_path(self.root, episode, role)
        wanted = sorted({_file_index(episode, role, int(i)) for i in indices})
        found: dict[int, np.ndarray] = {}
        if wanted:
            with av.open(str(path)) as c:
                s = c.streams.video[0]
                tb = float(s.time_base or 0)
                rate = float(s.average_rate or episode.cameras[role].fps)
                t0 = int(s.start_time or 0)
                ptr = 0
                while ptr < len(wanted):
                    c.seek(t0 + int(wanted[ptr] / rate / tb), stream=s, backward=True)
                    for frame in c.decode(s):
                        idx = round((int(frame.pts or 0) - t0) * tb * rate)
                        if idx < wanted[ptr]:
                            continue
                        img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
                        while ptr < len(wanted) and wanted[ptr] <= idx:
                            found[wanted[ptr]] = img  # a skipped index gets the next frame
                            ptr += 1
                        if ptr == len(wanted) or wanted[ptr] - idx > _RESEEK_GAP:
                            break
                    else:
                        if ptr < len(wanted):
                            raise IndexError(f"frame {wanted[ptr]} not found in {path}")
        return [found[_file_index(episode, role, int(i))] for i in indices]


class ZarrJpegReader(_Base):
    """PHANTOM scene video: a zarr group whose ``data`` holds one JPEG per frame."""

    def __init__(self, root: Path | str = ".") -> None:
        """Remember the directory relative group paths resolve against."""
        self.root = Path(root)
        self._groups: dict[Path, Any] = {}

    def _data(self, episode: Episode, role: str) -> Any:
        path = video_path(self.root, episode, role)
        if path not in self._groups:
            group: Any = open_group(path)
            self._groups[path] = group["data"]
        return self._groups[path]

    def frame_count(self, episode: Episode, role: str) -> int:
        """Return the stored frame count; decode frame 0 to prove the JPEGs are readable."""
        data = self._data(episode, role)
        n = int(data.shape[0])
        if n:
            self.read(episode, role, 0)
        return n

    def read_many(self, episode: Episode, role: str, indices: Sequence[int]) -> list[np.ndarray]:
        """Decode the JPEG at each index with pillow."""
        from PIL import Image

        data = self._data(episode, role)
        n = int(data.shape[0])
        out: list[np.ndarray] = []
        for i in indices:
            if not 0 <= int(i) < n:
                raise IndexError(f"frame {i} outside 0..{n - 1}")
            raw = data[int(i)]
            raw = raw.item() if isinstance(raw, np.ndarray) else raw  # 0-d object array
            img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
            out.append(np.asarray(img, dtype=np.uint8))
        return out


def smooth_image(seed_key: str, size: int = SIZE) -> np.ndarray:
    """Return a smooth random gray image (float, 0..255) determined by ``seed_key``."""
    seed = int.from_bytes(hashlib.sha1(seed_key.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    k = 12
    coef = rng.normal(size=(k, k)) / (1 + np.add.outer(np.arange(k), np.arange(k))) ** 0.5
    n = np.arange(size)
    basis = np.cos(np.pi * (2 * n[None, :] + 1) * np.arange(k)[:, None] / (2 * size))  # (k, size)
    img = basis.T @ coef @ basis
    img = (img - img.min()) / (img.max() - img.min())
    return img * 255


class SyntheticReader(_Base):
    """In-memory frames derived from the episode id (tests); no video files.

    ``alias`` makes one episode look like another; ``fail`` lists episode ids whose video does
    not decode; ``counts`` overrides decoded counts.
    """

    def __init__(
        self,
        alias: dict[str, str] | None = None,
        fail: set[str] | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        self.alias = alias or {}
        self.fail = fail or set()
        self.counts = counts or {}

    def frame_count(self, episode: Episode, role: str) -> int:
        """Return the declared count unless overridden; raise for ``fail`` episodes."""
        if episode.episode_id in self.fail:
            raise OSError("corrupt moov atom")
        return self.counts.get(episode.episode_id, episode.cameras[role].n_frames)

    def read(self, episode: Episode, role: str, index: int) -> np.ndarray:
        """Return the (aliased) episode's image for ``role``; the same for every index."""
        key = self.alias.get(episode.episode_id, episode.episode_id)
        gray = smooth_image(f"{key}/{role}")
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        return np.clip(rgb, 0, 255).astype(np.uint8)


class AutoReader(_Base):
    """Dispatch on the VideoRef path: a zarr group directory goes to ZarrJpegReader, else PyAV."""

    def __init__(self, root: Path | str = ".") -> None:
        """Both underlying readers resolve relative paths against ``root``."""
        self.root = Path(root)
        self._av = PyAVReader(self.root)
        self._zarr = ZarrJpegReader(self.root)

    def _pick(self, episode: Episode, role: str) -> PyAVReader | ZarrJpegReader:
        p = video_path(self.root, episode, role)
        if p.is_dir() or p.name.endswith(ZARR_SUFFIX):
            return self._zarr
        return self._av

    def frame_count(self, episode: Episode, role: str) -> int:
        """Delegate to the reader for this path."""
        return self._pick(episode, role).frame_count(episode, role)

    def read_many(self, episode: Episode, role: str, indices: Sequence[int]) -> list[np.ndarray]:
        """Delegate to the reader for this path."""
        return self._pick(episode, role).read_many(episode, role, indices)


class SourceRootReader(_Base):
    """AutoReader routed by ``episode.source`` to the ``root`` in ``<sources_dir>/<source>.yaml``.

    Sources without a config (e.g. roboreward, whose cached video paths are absolute) resolve
    against the working directory.
    """

    def __init__(self, sources_dir: Path | str = "configs/sources") -> None:
        """Remember where the per-source configs live."""
        self.sources_dir = Path(sources_dir)
        self._readers: dict[str, AutoReader] = {}

    def _reader(self, episode: Episode) -> AutoReader:
        from tactile_flora.config import SourceConfig, load_config

        if episode.source not in self._readers:
            cfg_path = self.sources_dir / f"{episode.source}.yaml"
            root = load_config(cfg_path, SourceConfig).root if cfg_path.exists() else "."
            self._readers[episode.source] = AutoReader(root)
        return self._readers[episode.source]

    def frame_count(self, episode: Episode, role: str) -> int:
        """Delegate to the source's reader."""
        return self._reader(episode).frame_count(episode, role)

    def read_many(self, episode: Episode, role: str, indices: Sequence[int]) -> list[np.ndarray]:
        """Delegate to the source's reader."""
        return self._reader(episode).read_many(episode, role, indices)
