"""Running checkpoint x benchmark x force_mode to predictions.parquet and metrics.json.

Owns: running checkpoint x benchmark x force_mode to predictions.parquet and metrics.json.
Must never: reading raw datasets or comparing predictions with different manifest hashes.
"""

from __future__ import annotations

from pathlib import Path


def run_eval(checkpoint: str, benchmark: str, force_mode: str, out: Path) -> Path:
    """Public entry point of this module (milestone 10)."""
    raise NotImplementedError("milestone 10")
