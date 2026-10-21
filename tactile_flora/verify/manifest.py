"""Writing and reading the accepted manifest and the frozen holdout id file.

Owns: writing and reading the accepted manifest and the frozen holdout id file.
Must never: changing the split of a frozen row or reading raw datasets.
"""

from __future__ import annotations

from pathlib import Path


def write_manifest(name: str, out_dir: Path) -> Path:
    """Public entry point of this module (milestone 6)."""
    raise NotImplementedError("milestone 6")
