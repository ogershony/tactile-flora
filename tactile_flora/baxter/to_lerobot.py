"""Converting recorded Baxter episode directories to a LeRobot v3 dataset.

Owns: converting recorded Baxter episode directories to a LeRobot v3 dataset.
Must never: resampling force itself (uses force/resample.py) or labeling.
"""

from __future__ import annotations

from pathlib import Path


def convert(episodes_dir: Path, out: Path) -> Path:
    """Public entry point of this module (milestone 11)."""
    raise NotImplementedError("milestone 11")
