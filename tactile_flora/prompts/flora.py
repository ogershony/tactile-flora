"""FLORA dense and outcome prompt builders with the force block inserted in one place.

Owns: fLORA dense and outcome prompt builders with the force block inserted in one place.
Must never: parsing outputs or loading models.
"""

from __future__ import annotations

from tactile_flora.config import ForceMode
from tactile_flora.schema import Window


def build_prompt(window: Window, force_mode: ForceMode) -> str:
    """Public entry point of this module (milestone 3)."""
    raise NotImplementedError("milestone 3")
