"""Source adapter for PHANTOM.

Owns: how PHANTOM is laid out on disk. Reads: LeRobot episodes + manifests/all.jsonl; DM-Tac
    fingertip force, channel pad.
Open question: license and exact manifest fields (success, tactile_contact, peak force units).
Must never: filter episodes in discover(), write to disk, or leak layout details to other modules.
"""

from __future__ import annotations

from pathlib import Path

from tactile_flora.schema import Episode, Label
from tactile_flora.sources.base import register_source


@register_source("phantom")
class PhantomAdapter:
    """Adapter for PHANTOM; every method is a stub until milestone 5."""

    name = "phantom"

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
