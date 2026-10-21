"""Per-pad weight-to-newton calibration written to the calibration JSON.

Owns: per-pad weight-to-newton calibration written to the calibration JSON.
Must never: touching ROS or recording episodes.
"""

from __future__ import annotations

from pathlib import Path


def calibrate(out: Path) -> Path:
    """Public entry point of this module (milestone 11)."""
    raise NotImplementedError("milestone 11")
