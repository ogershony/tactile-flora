"""The keystroke labeling tool for Baxter episodes (labels.json).

Owns: the keystroke labeling tool for Baxter episodes (labels.json).
Must never: showing force plots or deriving labels from force thresholds.
"""

from __future__ import annotations

from pathlib import Path


def run_labeler(episodes_dir: Path) -> Path:
    """Public entry point of this module (milestone 11)."""
    raise NotImplementedError("milestone 11")
