"""Small pure helpers shared by the source adapters (time alignment and session splits).

Owns: nearest-sample alignment, rate estimation and deterministic session-level split
assignment. Must never: know a dataset layout, touch the disk, or filter episodes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from tactile_flora.config import SourceConfig

DEFAULT_VAL_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1
SPLIT_NAMES = ("train", "val", "test")


def median_rate(t: np.ndarray, default: float = 1.0) -> float:
    """Return the median sampling rate (Hz) of a timestamp array, ``default`` if undefined."""
    if t.size < 2:
        return default
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else default


def nearest_index(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Index of the nearest ``src_t`` sample for every ``query_t`` (``src_t`` ascending)."""
    if src_t.size == 0:
        raise ValueError("cannot align to an empty stream")
    right = np.clip(np.searchsorted(src_t, query_t), 1, max(src_t.size - 1, 1))
    left = right - 1
    right = np.minimum(right, src_t.size - 1)
    pick_right = np.abs(src_t[right] - query_t) < np.abs(query_t - src_t[left])
    return np.where(pick_right, right, left)


def session_split(session: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    """Deterministically assign a whole session to train, val or test by hashing it."""
    digest = hashlib.sha1(f"{seed}:{session}".encode()).hexdigest()
    u = int(digest[:8], 16) / 16**8
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def assign_splits(session_by_episode: Mapping[str, str], cfg: SourceConfig) -> dict[str, list[str]]:
    """Group episode ids into train/val/test so a session never straddles two splits.

    ``cfg.split_rules`` may carry ``val_fraction`` and ``test_fraction``.
    """
    rules: dict[str, Any] = cfg.split_rules
    val_f = float(rules.get("val_fraction", DEFAULT_VAL_FRACTION))
    test_f = float(rules.get("test_fraction", DEFAULT_TEST_FRACTION))
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for ep, session in session_by_episode.items():
        out[session_split(session, cfg.seed, val_f, test_f)].append(ep)
    return out


def py(value: Any) -> Any:
    """Convert numpy scalars/arrays (and bytes) to plain Python values for ``meta``."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [py(v) for v in value.tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
