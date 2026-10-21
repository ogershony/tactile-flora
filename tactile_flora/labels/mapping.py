"""Converting native source labels to FLORA Label objects.

Owns: converting native source labels to FLORA Label objects.
Must never: reading raw datasets or using force thresholds for Baxter labels.
"""

from __future__ import annotations

from tactile_flora.schema import Label


def map_native_labels(source: str, native: list[Label]) -> list[Label]:
    """Public entry point of this module (milestone 4)."""
    raise NotImplementedError("milestone 4")
