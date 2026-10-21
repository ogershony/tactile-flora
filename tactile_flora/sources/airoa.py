"""Source adapter for AIRoA MoMa.

Owns: how AIRoA MoMa is laid out on disk. Reads: LeRobot v2.1; wrist F/T 6-D, channel wrist.
Split key is robot id plus date.
Must never: filter episodes in discover(), write to disk, or leak layout details to other modules.
"""

from __future__ import annotations

from pathlib import Path

from tactile_flora.schema import Episode, Label
from tactile_flora.sources.base import register_source


@register_source("airoa")
class AiroaAdapter:
    """Adapter for AIRoA MoMa; every method is a stub until milestone 5."""

    name = "airoa"

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
