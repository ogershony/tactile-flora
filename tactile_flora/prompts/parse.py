"""Parsing model output (Worst/Quality/Rationales, Score) into Label objects.

Owns: parsing model output (Worst/Quality/Rationales, Score) into Label objects.
Must never: building prompts or accepting a Worst that disagrees with the minimum digit.
"""

from __future__ import annotations

from tactile_flora.schema import Label


def parse_output(text: str, prompt_kind: str) -> Label:
    """Public entry point of this module (milestone 3)."""
    raise NotImplementedError("milestone 3")
