"""Vision-only FLORA pseudo-labeling of SharpaDex windows (origin=pseudo).

Owns: vision-only FLORA pseudo-labeling of SharpaDex windows (origin=pseudo).
Must never: running on eval_only sources or on ids in holdout_ids.json.
"""

from __future__ import annotations

from tactile_flora.schema import Label


def run_pseudo(source: str) -> list[Label]:
    """Public entry point of this module (milestone 8)."""
    raise NotImplementedError("milestone 8")
