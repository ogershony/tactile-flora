"""Vision-only FLORA pseudo-labeling of SharpaDex windows (origin=pseudo).

Owns: the gating around pseudo-labeling (eval-only and holdout refusal) and parsing of model
answers into Labels. The model inference itself is injected as ``runner`` (wired in milestone 8).
Must never: run on eval_only sources or on ids in holdout_ids.json.
"""

from __future__ import annotations

from collections.abc import Callable

from tactile_flora.config import SourceConfig
from tactile_flora.prompts.parse import parse_dense, parse_outcome, parsed_to_label
from tactile_flora.schema import Label, Window


class PseudoRefused(RuntimeError):
    """Raised when pseudo-labeling is not allowed for the source or the windows."""


def _episode_id(window: Window) -> str:
    return window.episode_ref.split("/", 1)[-1]


def pseudo_label(
    source_cfg: SourceConfig,
    windows: list[Window],
    holdout_ids: set[str],
    runner: Callable[[Window], str],
) -> list[Label]:
    """Run ``runner`` on each window and parse its answer into a Label(origin="pseudo").

    Raises PseudoRefused if the source is eval_only or any window's episode id (the part of
    ``episode_ref`` after ``source/``, or the full ref) is in ``holdout_ids``; the check runs
    before any inference. Dense answers go through parse_dense (a malformed answer becomes
    all-normal), outcome answers through parse_outcome.
    """
    if source_cfg.eval_only:
        raise PseudoRefused(f"source {source_cfg.name} is eval_only")
    bad = sorted(
        w.episode_ref
        for w in windows
        if _episode_id(w) in holdout_ids or w.episode_ref in holdout_ids
    )
    if bad:
        raise PseudoRefused(f"windows from holdout episodes: {bad[:5]}")
    labels: list[Label] = []
    for w in windows:
        text = runner(w)
        parsed = parse_dense(text) if w.prompt_kind == "dense" else parse_outcome(text)
        labels.append(parsed_to_label(parsed, w.anchor_s, "pseudo"))
    return labels


def run_pseudo(source: str) -> list[Label]:
    """CLI entry point (milestone 8: wires the real model into ``pseudo_label``)."""
    raise NotImplementedError("milestone 8")
