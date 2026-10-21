"""Resample a ForceStream onto frame times (median F, D, dF, contact flag).

Owns: the per-frame features plus the raw 100 Hz slice per 0.5 s group.
Must never: normalizing, rendering text, or reading datasets.
"""

from __future__ import annotations

from tactile_flora.schema import ForceStream


def resample(stream: ForceStream, frame_times: list[float]) -> dict[str, object]:
    """Public entry point of this module (milestone 2)."""
    raise NotImplementedError("milestone 2")
