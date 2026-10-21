"""Perceptual frame hashes and the benchmark hash index.

Owns: perceptual frame hashes and the benchmark hash index.
Must never: deciding which episodes to drop.
"""

from __future__ import annotations

from pathlib import Path


def build_benchmark_index(out: Path) -> Path:
    """Public entry point of this module (milestone 6)."""
    raise NotImplementedError("milestone 6")
