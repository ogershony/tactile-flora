"""FLORA window geometry, dense label stitching and target-string construction.

Owns: fLORA window geometry, dense label stitching and target-string construction.
Must never: rendering force or building prompts.
"""

from __future__ import annotations

from tactile_flora.schema import Episode, Window


def make_windows(episode: Episode) -> list[Window]:
    """Public entry point of this module (milestone 3)."""
    raise NotImplementedError("milestone 3")
