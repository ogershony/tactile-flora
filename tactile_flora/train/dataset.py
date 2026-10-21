"""Turning a manifest into tokenized samples with answer-only loss masks.

Owns: turning a manifest into tokenized samples with answer-only loss masks.
Must never: reading raw datasets.
"""

from __future__ import annotations

from pathlib import Path


def build_dataset(manifest: Path, force_mode: str) -> list[dict[str, object]]:
    """Public entry point of this module (milestone 9)."""
    raise NotImplementedError("milestone 9")
