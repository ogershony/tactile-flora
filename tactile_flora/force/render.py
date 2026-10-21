"""Rendering normalized features to the force text block, one line per frame.

Owns: rendering normalized features to the force text block, one line per frame.
Must never: resampling, normalizing, or deciding the ablation mode.
"""

from __future__ import annotations


def render_block(features: dict[str, object]) -> list[str]:
    """Public entry point of this module (milestone 2)."""
    raise NotImplementedError("milestone 2")
