"""Source adapter for HRDexDB (eval only).

Owns: how HRDexDB (eval only) is laid out on disk. Reads: Per-trial archives; fingertip tactile
    mapped to pad using the two most-loaded fingers.
Open question: tactile layout per hand type.
Must never: filter episodes in discover(), write to disk, or leak layout details to other modules.
"""

from __future__ import annotations

from pathlib import Path

from tactile_flora.schema import Episode, Label
from tactile_flora.sources.base import register_source


@register_source("hrdexdb")
class HrdexdbAdapter:
    """Adapter for HRDexDB (eval only); every method is a stub until milestone 5."""

    name = "hrdexdb"

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
