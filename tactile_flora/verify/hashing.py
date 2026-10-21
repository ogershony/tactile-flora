"""Perceptual frame hashes, the per-episode hash cache and the benchmark hash index.

Owns: the 64-bit DCT pHash, Hamming distance, the benchmark index parquet
(data/verified/benchmark_hashes.parquet: source, episode_id, h_first, h_mid, h_last) and the
per-source cache (data/verified/<source>/hashes.parquet).
Must never: decide which episodes to drop (checks.check_overlap does) or decode video itself.

pHash: grayscale frame -> 32x32 -> 2-D DCT-II -> top-left 8x8 block -> bit = coefficient above
the block median, packed row-major into 16 hex characters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tactile_flora.schema import BenchmarkHashRow, Episode, FrameReader

HASH_SIZE = 32
HASH_BLOCK = 8
SCENE_ROLE = "scene"
_N = np.arange(HASH_SIZE)
_DCT = np.cos(np.pi * (2 * _N[None, :] + 1) * _N[:, None] / (2 * HASH_SIZE))  # (k, n)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an (H, W) or (H, W, 3) frame to float64 luma."""
    a = np.asarray(frame, dtype=np.float64)
    if a.ndim == 3:
        return a @ np.array([0.299, 0.587, 0.114])
    return a


def phash64(gray: np.ndarray) -> str:
    """Return the 64-bit DCT perceptual hash of a 2-D gray frame as 16 hex characters."""
    a = to_gray(gray)
    if a.shape != (HASH_SIZE, HASH_SIZE):
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        a = np.asarray(img.resize((HASH_SIZE, HASH_SIZE), Image.Resampling.BOX), dtype=np.float64)
    coeffs = _DCT @ a @ _DCT.T
    block = coeffs[:HASH_BLOCK, :HASH_BLOCK]
    bits = (block > np.median(block)).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    """Return the number of differing bits between two hex hashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def episode_hashes(episode: Episode, reader: FrameReader, role: str = SCENE_ROLE) -> list[str]:
    """Return pHashes of the first, middle and last frame of the scene camera."""
    n = reader.frame_count(episode, role)
    if n < 1:
        raise ValueError(f"{episode.episode_id}: no frames to hash")
    return [phash64(to_gray(reader.read(episode, role, i))) for i in (0, n // 2, n - 1)]


def cached_hashes(
    episode: Episode, reader: FrameReader, cache: dict[str, list[str]] | None
) -> list[str]:
    """Return ``episode_hashes`` using (and filling) ``cache`` keyed by episode id."""
    if cache is not None and episode.episode_id in cache:
        return cache[episode.episode_id]
    hashes = episode_hashes(episode, reader)
    if cache is not None:
        cache[episode.episode_id] = hashes
    return hashes


def load_hash_cache(path: Path | str) -> dict[str, list[str]]:
    """Read data/verified/<source>/hashes.parquet; a missing file gives an empty cache."""
    p = Path(path)
    if not p.exists():
        return {}
    rows = pq.read_table(p).to_pylist()
    return {r["episode_id"]: [r["h_first"], r["h_mid"], r["h_last"]] for r in rows}


def save_hash_cache(path: Path | str, cache: dict[str, list[str]]) -> Path:
    """Write the per-episode cache, rows sorted by episode id."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(cache)
    table = pa.table(
        {
            "episode_id": pa.array(ids, pa.string()),
            "h_first": pa.array([cache[i][0] for i in ids], pa.string()),
            "h_mid": pa.array([cache[i][1] for i in ids], pa.string()),
            "h_last": pa.array([cache[i][2] for i in ids], pa.string()),
        }
    )
    pq.write_table(table, p)
    return p


def build_benchmark_index(
    episodes: Iterable[Episode], reader: FrameReader, path: Path | str
) -> Path:
    """Hash every benchmark episode once and write the index parquet to ``path``."""
    rows = [(e.source, e.episode_id, *episode_hashes(e, reader)) for e in episodes]
    rows.sort()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ("source", "episode_id", "h_first", "h_mid", "h_last")
    table = pa.table({c: pa.array([r[i] for r in rows], pa.string()) for i, c in enumerate(cols)})
    pq.write_table(table, p)
    return p


def load_index(path: Path | str) -> list[BenchmarkHashRow]:
    """Read the benchmark hash index; a missing file gives an empty index."""
    p = Path(path)
    if not p.exists():
        return []
    return [BenchmarkHashRow.model_validate(r) for r in pq.read_table(p).to_pylist()]


def matching_row(
    hashes: Sequence[str],
    index: Iterable[BenchmarkHashRow],
    max_distance: int = 6,
) -> BenchmarkHashRow | None:
    """Return the first benchmark row whose three hashes are all under ``max_distance`` bits."""
    for row in index:
        pairs = zip(hashes, (row.h_first, row.h_mid, row.h_last), strict=True)
        if all(hamming(a, b) < max_distance for a, b in pairs):
            return row
    return None


def is_overlap(
    hashes: Sequence[str], index: Iterable[BenchmarkHashRow], max_distance: int = 6
) -> bool:
    """Return True when all three hashes are under ``max_distance`` bits of one benchmark row."""
    return matching_row(hashes, index, max_distance) is not None
