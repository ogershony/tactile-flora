"""Programmatic force-event detectors (slip, contact_loss, regrasp, crush, off_axis).

Owns: programmatic force-event detectors (slip, contact_loss, regrasp, crush, off_axis).
Must never: producing labels for eval-only Baxter data or reading raw datasets.
"""

from __future__ import annotations

from tactile_flora.schema import Episode, Label


def detect_events(episode: Episode) -> list[Label]:
    """Public entry point of this module (milestone 4)."""
    raise NotImplementedError("milestone 4")
