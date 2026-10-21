"""Source adapter for SoftVTBench (eval only).

Owns: how SoftVTBench (eval only) is laid out on disk. Reads: HDF5 + MP4; FEM contact state to
    scalar force proxy per finger, channel pad.
Open question: which FEM array gives per-finger contact force.
Must never: filter episodes in discover(), write to disk, or leak layout details to other modules.
"""

from __future__ import annotations

from pathlib import Path

from tactile_flora.schema import Episode, Label
from tactile_flora.sources.base import register_source


@register_source("softvtbench")
class SoftvtbenchAdapter:
    """Adapter for SoftVTBench (eval only); every method is a stub until milestone 5."""

    name = "softvtbench"

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
