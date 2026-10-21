"""Per-source markdown verification reports.

Owns: per-source markdown verification reports.
Must never: computing checks or writing manifests.
"""

from __future__ import annotations

from pathlib import Path


def write_report(source: str, out_dir: Path) -> Path:
    """Public entry point of this module (milestone 6)."""
    raise NotImplementedError("milestone 6")
