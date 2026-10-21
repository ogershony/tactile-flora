"""Fitting and applying per-source normalization constants (norm.json, norm_version).

Owns: fitting and applying per-source normalization constants (norm.json, norm_version).
Must never: fitting eval-only sources on training data, or resampling.
"""

from __future__ import annotations

from pathlib import Path


def fit_norm(source: str, features: dict[str, object], out: Path) -> str:
    """Public entry point of this module (milestone 2)."""
    raise NotImplementedError("milestone 2")
