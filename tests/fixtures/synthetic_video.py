"""In-memory FrameReader for tests: deterministic smooth frames per episode, no video files."""

from __future__ import annotations

import hashlib

import numpy as np

from tactile_flora.schema import Episode

SIZE = 64


def smooth_image(seed_key: str, size: int = SIZE) -> np.ndarray:
    """A smooth random gray image (float, 0..255) determined by ``seed_key``."""
    seed = int.from_bytes(hashlib.sha1(seed_key.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    k = 12
    coef = rng.normal(size=(k, k)) / (1 + np.add.outer(np.arange(k), np.arange(k))) ** 0.5
    n = np.arange(size)
    basis = np.cos(np.pi * (2 * n[None, :] + 1) * np.arange(k)[:, None] / (2 * size))  # (k, size)
    img = basis.T @ coef @ basis
    img = (img - img.min()) / (img.max() - img.min())
    return img * 255


class SyntheticFrameReader:
    """Frames derived from the episode id; ``alias`` makes one episode look like another.

    ``fail`` lists episode ids whose video does not decode; ``counts`` overrides decoded counts.
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
        """Declared frame count unless overridden; raises for ``fail`` episodes."""
        if episode.episode_id in self.fail:
            raise OSError("corrupt moov atom")
        return self.counts.get(episode.episode_id, episode.cameras[role].n_frames)

    def read_frame(self, episode: Episode, role: str, index: int) -> np.ndarray:
        """Scene image of the (aliased) episode, as RGB uint8."""
        key = self.alias.get(episode.episode_id, episode.episode_id)
        gray = smooth_image(f"{key}/{role}")
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        return np.clip(rgb, 0, 255).astype(np.uint8)
