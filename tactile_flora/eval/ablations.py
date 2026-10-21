"""Force-mode transforms (real, omitted, zero, shuffled, video_blank) on resampled features.

Owns: force-mode transforms (real, omitted, zero, shuffled, video_blank) on resampled features.
Must never: rendering text or loading models.
"""

from __future__ import annotations

from tactile_flora.config import ForceMode


def apply_ablation(features: dict[str, object], mode: ForceMode, seed: int) -> dict[str, object]:
    """Public entry point of this module (milestone 10)."""
    raise NotImplementedError("milestone 10")
