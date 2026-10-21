"""Vision-only FLORA pseudo-labeling of SharpaDex windows (origin=pseudo).

Owns: the gating around pseudo-labeling (eval-only and holdout refusal), parsing of model
answers into Labels, and ``run_pseudo``, which wires a ``Backend`` (eval/inference.py) into
``pseudo_label`` for a source and writes ``data/verified/<source>/pseudo.parquet``.
Must never: run on eval_only sources or on ids in holdout_ids.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tactile_flora.config import SourceConfig
from tactile_flora.labels import windows as W
from tactile_flora.prompts.parse import parse_dense, parse_outcome, parsed_to_label
from tactile_flora.schema import Episode, Label, Window

if TYPE_CHECKING:
    from tactile_flora.eval.inference import Backend


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


def run_pseudo(
    cfg: SourceConfig,
    data_root: Path,
    backend: Backend,
    *,
    seed: int = 0,
    limit: int | None = None,
) -> Path:
    """Label every dense window of a source's accepted episodes with the vision-only model.

    Reads data/verified/<source>/checks.parquet (run ``tf verify`` first; episodes failing a
    drop_episode check are skipped), skips held-out episodes of data/manifest/holdout_ids.json
    (pseudo labels may never enter an eval split), prompts WITHOUT a force block, and writes
    ``pseudo.parquet`` (episode_id, anchor_s, raw_text, parse_failed, label_json, origin) and a
    JSONL log next to it. ``limit`` caps the number of episodes (smoke runs). Returns the
    parquet path. Refuses eval-only sources (PseudoRefused) before loading anything.
    """
    import pandas as pd

    from tactile_flora.eval.inference import DENSE_MAX_NEW_TOKENS
    from tactile_flora.prompts.flora import build_prompt, to_messages
    from tactile_flora.prompts.parse import dense_regex, parse_dense
    from tactile_flora.sources.base import make_adapter
    from tactile_flora.verify import holdout_path, manifest, report, verified_dir

    if cfg.eval_only:
        raise PseudoRefused(f"source {cfg.name} is eval_only")
    out_dir = verified_dir(data_root, cfg.name)
    if not (out_dir / "checks.parquet").exists():
        raise RuntimeError(f"no verification results under {out_dir}; run tf verify first")
    results = report.read_checks(out_dir)
    ok = sorted(
        e
        for e, per in results.items()
        if not any(not r.passed and r.action == "drop_episode" for r in per.values())
    )
    hp = holdout_path(data_root)
    held = manifest.load_holdout(hp).held_out_ids if hp.exists() else frozenset[str]()
    adapter = make_adapter(cfg.name, cfg)
    adapter.discover(Path(cfg.root))
    splits = adapter.splits()
    eps: dict[str, Episode] = {}
    windows: list[Window] = []
    for eid in ok:
        if f"{cfg.name}/{eid}" in held or len(eps) == limit:
            continue
        ep = adapter.load(eid)
        eps[eid] = ep
        split = manifest.assign_split(ep, splits)
        windows += [
            Window(
                episode_ref=f"{cfg.name}/{eid}", prompt_kind="dense", anchor_s=a,
                frame_times=tuple(W.dense_frame_times(a)), force_lines=(), target="", split=split,
            )
            for a in W.dense_anchors(ep.duration_s)
        ]  # fmt: skip
    raw: dict[tuple[str, float], str] = {}

    def runner(w: Window) -> str:
        ep = eps[w.episode_ref.split("/", 1)[1]]
        cam = ep.cameras.get("scene") or next(iter(ep.cameras.values()))
        idx = [min(max(int(t * cam.fps + 0.5), 0), cam.n_frames - 1) for t in w.frame_times]
        bundle = build_prompt(
            w, None, task=ep.task, camera_roles=tuple(ep.cameras), fps=cam.fps, frame_indices=idx
        )
        res = backend.generate(
            to_messages(bundle), bundle.video, regex=dense_regex(),
            max_new_tokens=DENSE_MAX_NEW_TOKENS, top_logprobs=0, episode=ep,
        )  # fmt: skip
        raw[(w.episode_ref, w.anchor_s)] = res.text
        return res.text

    labels = pseudo_label(cfg, windows, set(held), runner)
    recs: list[dict[str, Any]] = []
    for w, lab in zip(windows, labels, strict=True):
        text = raw[(w.episode_ref, w.anchor_s)]
        recs.append(
            {
                "episode_id": w.episode_ref.split("/", 1)[1],
                "anchor_s": w.anchor_s,
                "raw_text": text,
                "parse_failed": parse_dense(text).parse_failed,
                "label_json": lab.model_dump_json(),
                "origin": "pseudo",
                "seed": seed,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pseudo.parquet"
    pd.DataFrame(recs).to_parquet(path, index=False)
    with open(out_dir / "pseudo.log.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(
                json.dumps({k: r[k] for k in ("episode_id", "anchor_s", "parse_failed")}) + "\n"
            )
    return path
