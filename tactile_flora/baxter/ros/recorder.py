"""Capturing one Baxter episode directory (videos, parquet, meta.json).

Owns: capturing one Baxter episode directory (videos, parquet, meta.json).
Must never: starting an episode while any channel is flat or saturated.
"""

from __future__ import annotations

from pathlib import Path


def record_episode(out_dir: Path) -> Path:
    """Public entry point of this module (milestone 11)."""
    raise NotImplementedError("milestone 11")
