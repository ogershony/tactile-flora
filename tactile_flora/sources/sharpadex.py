"""Source adapter for SharpaDex v1.0.

Owns: how SharpaDex v1.0 is laid out on disk. Reads: LeRobot v3 parquet; observation.tactile 60-D,
    right thumb and index fz as f_a, f_b, channel pad.
Split key is the season directory.
Must never: filter episodes in discover(), write to disk, or leak layout details to other modules.
"""

from __future__ import annotations

from pathlib import Path

from tactile_flora.schema import Episode, Label
from tactile_flora.sources.base import register_source


@register_source("sharpadex")
class SharpadexAdapter:
    """Adapter for SharpaDex v1.0; every method is a stub until milestone 5."""

    name = "sharpadex"

    def discover(self, root: Path) -> list[str]:
        """Return every episode id under ``root``."""
        raise NotImplementedError("milestone 5")

    def load(self, episode_id: str) -> Episode:
        """Load a full episode on the episode clock."""
        raise NotImplementedError("milestone 5")

    def splits(self) -> dict[str, list[str]]:
        """Return the session-level split assignment."""
        raise NotImplementedError("milestone 5")

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return the source's own labels."""
        raise NotImplementedError("milestone 5")
