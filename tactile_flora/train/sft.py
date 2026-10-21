"""The LoRA SFT loop and the run directory it writes.

Owns: the LoRA SFT loop and the run directory it writes.
Must never: starting without holdout_ids.json or with a stale report_version; reading raw datasets.
"""

from __future__ import annotations

from tactile_flora.config import TrainConfig


def run_sft(cfg: TrainConfig, dry_run: bool = False) -> None:
    """Public entry point of this module (milestone 9)."""
    raise NotImplementedError("milestone 9")
