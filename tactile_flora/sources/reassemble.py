"""Source adapter for REASSEMBLE.

Owns: how REASSEMBLE is laid out on disk. Reads: HDF5 per demonstration; wrist F/T 6-D, channel
    wrist.
Open question: anomaly annotation format (span times vs frame indices).
Must never: filter episodes in discover(), write to disk, or leak layout details to other modules.
"""

from __future__ import annotations

from pathlib import Path

from tactile_flora.schema import Episode, Label
from tactile_flora.sources.base import register_source


@register_source("reassemble")
class ReassembleAdapter:
    """Adapter for REASSEMBLE; every method is a stub until milestone 5."""

    name = "reassemble"

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
