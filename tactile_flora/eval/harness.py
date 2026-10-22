"""Running checkpoint x benchmark x force_mode to predictions.parquet and metrics.json.

Owns: run_eval (one cell), run_grid (the condition matrix), write_report (markdown tables),
the window providers that turn manifest rows back into prompts, and ManifestMismatchError.
Must never: read raw datasets except through the registered source adapters (to rebuild the
windows of manifest rows), or compare predictions whose manifest hashes differ.

Windows are rebuilt on the fly: manifest rows carry ids and anchors only, so for each episode
the adapter is loaded, ``episode_windows`` re-derives the windows (control: no force lines),
and the force features are resampled and normalized with the source's ``norm.json`` before
the force mode is applied and the block rendered. A benchmark with a
``<manifest>.windows.parquet`` sidecar (eval/cache.py, used by roboreward_500) is read from
that instead. Output layout::

    <data_root>/eval/<checkpoint tag>/<benchmark>/<mode>/{predictions.parquet, metrics.json,
                                                            config.json, log.jsonl}

where the tag is the last component of the checkpoint path, except that a path ending in
``adapter`` uses the run directory name (``data/runs/force/adapter`` -> ``force``) so that every
run's ``adapter`` directory does not collide ("released" stays as is). Two different checkpoints
that map to one tag are rejected (``CheckpointTagCollisionError``, checked when the grid is
expanded).
Stitching of dense windows to 1 Hz uses labels/windows.py for predictions and truth alike, and
truth is the window target string. Dense predictions, truth and objects are keyed by
``source/episode_id``, never the bare episode id. An unparsable outcome answer counts in
``parse_failed`` and is replaced by score 3 (``metrics.OUTCOME_PARSE_FAIL_SCORE``, the middle of
the scale; the dense replacement is all-normal, as in the paper), so it stays in MAE, accuracy
and AUROC (with E[s] = 3). Binary outcome labels and the predicted-success cut share one
constant, ``metrics.SUCCESS_THRESHOLD`` (>= 4.0). When a backend returns no score-token logprobs
``expected_score`` is NaN (never silently the integer score), ``no_logprobs`` counts them in
metrics.json, and accuracy / AUROC / mae_vs_dsr then use the integer score with a
``warnings`` entry.
Each episode is loaded and windowed once per run (the provider output is cached in memory,
keyed by ``source/episode_id``). A grid cell counts as done only when its ``metrics.json`` (the
last file written) exists and records the current manifest hash, windows-cache hash and config
hash; otherwise the cell is rerun and the reason logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from tactile_flora.config import (
    EvalConfig,
    ForceConfig,
    ForceMode,
    GridConfig,
    SourceConfig,
    load_config,
)
from tactile_flora.eval import cache as wcache
from tactile_flora.eval import metrics as M
from tactile_flora.eval.ablations import PoolEntry, apply_force_mode
from tactile_flora.eval.inference import (
    DENSE_MAX_NEW_TOKENS,
    RATIONALE_MAX_NEW_TOKENS,
    SCORE_ONLY_MAX_NEW_TOKENS,
    TOP_LOGPROBS,
    Backend,
    expected_score,
)
from tactile_flora.force import normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.labels import windows as W
from tactile_flora.prompts.flora import build_prompt, to_messages
from tactile_flora.prompts.parse import (
    ParsedDense,
    ParsedOutcome,
    dense_regex,
    outcome_regex,
    parse_dense,
    parse_outcome,
)
from tactile_flora.schema import Episode, ManifestRow, ResampledForce, Window
from tactile_flora.sources.base import make_adapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.video.readers import SourceRootReader as SourceRootReader

log = logging.getLogger(__name__)

OMITTED_ONLY = frozenset({"roboreward_500"})  # no force exists for these benchmarks
DEFAULT_SOURCES_DIR = Path("configs/sources")


class ManifestMismatchError(RuntimeError):
    """Raised when a report would compare predictions built from different manifests."""


class CheckpointTagCollisionError(ValueError):
    """Raised when two different checkpoints of a grid map to the same directory tag."""


class EpisodeBatch(NamedTuple):
    """One episode with its force-free windows and the manifest window id of each."""

    episode: Episode
    windows: list[Window]
    window_ids: list[str]
    shift_s: float = 0.0  # force clock correction from the manifest rows (0.0 when absent)


Provider = Callable[[], Iterator[EpisodeBatch]]


def checkpoint_tag(checkpoint: str) -> str:
    """Directory name for a checkpoint.

    The last path component, except that a path ending in ``adapter`` uses its run directory
    name (``data/runs/force/adapter`` -> ``force``). "released" is unchanged.
    """
    p = Path(checkpoint)
    if p.name == "adapter" and p.parent.name:
        return p.parent.name
    return p.name or checkpoint


def cell_dir(data_root: Path, checkpoint: str, benchmark: str, mode: str) -> Path:
    """Return data/eval/<tag>/<benchmark>/<mode>."""
    return data_root / "eval" / checkpoint_tag(checkpoint) / benchmark / mode


def check_mode(benchmark: str, mode: str) -> None:
    """Raise ValueError if ``mode`` is not allowed for ``benchmark``."""
    if benchmark in OMITTED_ONLY and mode != "omitted":
        raise ValueError(f"benchmark {benchmark} accepts only force mode 'omitted', got {mode!r}")


def _manifest_path(spec_manifest: str, data_root: Path) -> Path:
    p = Path(spec_manifest)
    if p.exists() or p.is_absolute():
        return p
    return data_root / "manifest" / p.name


def _load_source_cfgs(
    sources: set[str], sources_dir: Path, given: Mapping[str, SourceConfig] | None
) -> dict[str, SourceConfig]:
    out: dict[str, SourceConfig] = dict(given or {})
    for s in sources - set(out):
        out[s] = load_config(sources_dir / f"{s}.yaml", SourceConfig)
    return out


def adapter_provider(
    rows: list[ManifestRow],
    source_cfgs: Mapping[str, SourceConfig],
    force_cfg: ForceConfig,
    data_root: Path | None = None,
) -> Provider:
    """Return a provider that reloads manifest episodes via their adapter and re-derives windows."""
    by_ep: dict[tuple[str, str], list[ManifestRow]] = {}
    for r in rows:
        by_ep.setdefault((r.source, r.episode_id), []).append(r)

    def _iter() -> Iterator[EpisodeBatch]:
        adapters: dict[str, Any] = {}
        inputs: dict[str, Any] = {}
        consts_by_source: dict[str, normalize.NormConstants] = {}
        for (src, eid), rs in sorted(by_ep.items()):
            if src not in adapters:
                a = make_adapter(src, source_cfgs[src])
                a.discover(Path(source_cfgs[src].root))
                adapters[src] = a
                if data_root is not None:
                    from tactile_flora.verify import report
                    from tactile_flora.verify.windowing import build_source_inputs

                    vdir = data_root / "verified" / src
                    consts = normalize.load(vdir / "norm.json")
                    consts_by_source[src] = consts
                    results = report.read_checks(vdir) if (vdir / "checks.parquet").exists() else {}
                    inputs[src] = build_source_inputs(
                        a, force_cfg, consts, vdir, sorted(results), results
                    )
            ep: Episode = adapters[src].load(eid)
            inp = inputs.get(src)
            wins = {
                manifest.window_id(src, eid, w.prompt_kind, w.anchor_s): w
                for w in _episode_windows(
                    ep, force_cfg, rs[0].split, inp, consts_by_source.get(src)
                )
            }
            missing = [r.window_id for r in rs if r.window_id not in wins]
            if missing:
                raise RuntimeError(f"{src}/{eid}: manifest windows not reproducible: {missing[:3]}")
            ids = [r.window_id for r in rs]
            shift = float(
                inp.shift_of(eid) if inp is not None else getattr(rs[0], "shift_s", 0.0) or 0.0
            )
            yield EpisodeBatch(ep, [wins[i] for i in ids], ids, shift)

    return _iter


def _episode_windows(
    ep: Episode,
    force_cfg: ForceConfig,
    split: str,
    inputs: Any | None = None,
    consts: normalize.NormConstants | None = None,
) -> list[Window]:
    from tactile_flora.verify.windowing import episode_windows

    return episode_windows(ep, force_cfg, consts, split, with_force=False, inputs=inputs)  # type: ignore[arg-type]


def cached_provider(items: list[wcache.CachedWindow]) -> Provider:
    """Return a provider over a windows sidecar; each window is a one-window episode batch."""

    def _iter() -> Iterator[EpisodeBatch]:
        for it in items:
            ep, win = wcache.to_episode_window(it)
            yield EpisodeBatch(ep, [win], [it.window_id])

    return _iter


def make_provider(
    manifest_path: Path,
    kinds: set[str],
    force_cfg: ForceConfig,
    sources_dir: Path,
    source_cfgs: Mapping[str, SourceConfig] | None,
    data_root: Path | None = None,
) -> Provider:
    """Windows sidecar if one exists next to the manifest, else the adapter rebuild."""
    sidecar = wcache.cache_path(manifest_path)
    if sidecar.exists():
        return cached_provider(
            [i for i in wcache.read_windows_cache(sidecar) if i.prompt_kind in kinds]
        )
    rows = [r for r in manifest.read_manifest(manifest_path) if r.prompt_kind in kinds]
    cfgs = _load_source_cfgs({r.source for r in rows}, sources_dir, source_cfgs)
    return adapter_provider(rows, cfgs, force_cfg, data_root)


def window_features(
    ep: Episode,
    w: Window,
    consts: normalize.NormConstants,
    force_cfg: ForceConfig,
    shift_s: float = 0.0,
) -> ResampledForce:
    """Resample and normalize the episode force at the window's frame times.

    A non-zero ``shift_s`` (ManifestRow.shift_s) is applied through
    ``verify.windowing.resample_shifted`` when that function exists.
    """
    from tactile_flora.verify import windowing

    shifted = getattr(windowing, "resample_shifted", None)
    if shift_s and shifted is not None:
        feats = shifted(ep, w.frame_times, force_cfg.resample, shift_s)
    else:
        if shift_s:
            log.warning("manifest shift_s=%s ignored: resample_shifted is unavailable", shift_s)
        feats = resample(ep.force, ep.gripper, w.frame_times, force_cfg.resample)
    return normalize.apply(feats, consts, force_cfg.render)


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _clean(x: float) -> float | None:
    return None if isinstance(x, float) and math.isnan(x) else x


def _frame_indices(w: Window, fps: float, n_frames: int) -> list[int]:
    return [min(max(int(t * fps + 0.5), 0), n_frames - 1) for t in w.frame_times]


def _read_consts(data_root: Path, sources: set[str]) -> dict[str, normalize.NormConstants]:
    return {s: normalize.load(data_root / "verified" / s / "norm.json") for s in sources}


def _stitched(per_ep: dict[str, list[tuple[float, list[int]]]]) -> pd.DataFrame:
    """1 Hz frame keyed by ``source/episode_id`` (the dict keys)."""
    recs = [
        {"episode_id": e, "t": t, "quality": q}
        for e, wins in per_ep.items()
        for t, q in W.to_1hz(W.stitch_dense(wins))
    ]
    return pd.DataFrame(recs, columns=["episode_id", "t", "quality"])


def _has_over_compression(ep: Episode) -> bool:
    """Return whether the episode carries source over-compression event spans."""
    return any(
        lb.origin == "source" and sp.rationale == M.OVERCOMPRESS_RATIONALE
        for lb in ep.labels
        if lb.kind != "outcome"
        for sp in lb.spans
    )


def _dsr(ep: Episode, success: bool) -> float:
    """DSR: 1 if success and peak_fem_kabsch_max_pct <= tolerance, else 0; NaN if unknown."""
    peak, tol = ep.meta.get("peak_fem_kabsch_max_pct"), ep.meta.get("dsr_tolerance_pct")
    if peak is None or tol is None:
        return math.nan
    return float(success and float(peak) <= float(tol))


def _load_batches(prov: Provider) -> dict[str, EpisodeBatch]:
    """Run the provider once; episodes keyed by ``source/episode_id`` (duplicates merge)."""
    out: dict[str, EpisodeBatch] = {}
    for b in prov():
        key = f"{b.episode.source}/{b.episode.episode_id}"
        if key in out:
            o = out[key]
            out[key] = EpisodeBatch(
                o.episode, o.windows + b.windows, o.window_ids + b.window_ids, o.shift_s
            )
        else:
            out[key] = b
    return out


def _metric_block(
    names: list[str],
    dense: tuple[pd.DataFrame, pd.DataFrame] | None,
    outcome: tuple[pd.DataFrame, pd.DataFrame] | None,
) -> dict[str, float | None]:
    return {k: _clean(v) for k, v in M.compute_metrics(names, dense, outcome).items()}


def _cell_config(
    checkpoint: str,
    benchmark: str,
    force_mode: str,
    seed: int,
    eval_cfg: EvalConfig,
    force_cfg: ForceConfig,
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "benchmark": benchmark,
        "force_mode": force_mode,
        "seed": seed,
        "eval": eval_cfg.model_dump(mode="json"),
        "force": force_cfg.model_dump(mode="json"),
    }


def _config_hash(config: Mapping[str, Any]) -> str:
    """Hash of the cell config without the backend name (unknown before a backend exists)."""
    return _sha({k: v for k, v in config.items() if k != "backend"})


def stale_reason(
    cell: Path,
    mpath: Path,
    config: Mapping[str, Any],
) -> str | None:
    """Why ``cell`` must be rerun, or None when its metrics.json matches the current inputs."""
    f = cell / "metrics.json"
    if not f.exists():
        return "metrics.json missing"
    try:
        doc = json.loads(f.read_text("utf-8"))
    except (OSError, ValueError):
        return "metrics.json unreadable"
    sidecar = wcache.cache_path(mpath)
    cur_manifest = manifest.manifest_hash(mpath) if mpath.exists() else "none"
    cur_cache = manifest.manifest_hash(sidecar) if sidecar.exists() else None
    if doc.get("manifest_hash") != cur_manifest or doc.get("windows_cache_hash") != cur_cache:
        return "manifest hash changed"
    if doc.get("config_hash") != _config_hash(config):
        return "config hash changed"
    return None


def run_eval(
    checkpoint: str,
    benchmark: str,
    force_mode: ForceMode,
    eval_cfg: EvalConfig,
    force_cfg: ForceConfig,
    data_root: Path,
    backend: Backend,
    *,
    seed: int,
    sources_dir: Path = DEFAULT_SOURCES_DIR,
    source_cfgs: Mapping[str, SourceConfig] | None = None,
    provider: Provider | None = None,
) -> Path:
    """Evaluate one checkpoint x benchmark x force-mode cell and return its directory.

    Writes predictions.parquet, metrics.json, config.json and log.jsonl there. ``provider``
    overrides how windows are produced (tests); by default they come from the manifest.
    """
    check_mode(benchmark, force_mode)
    spec = eval_cfg.benchmarks[benchmark]
    mpath = _manifest_path(spec.manifest, data_root)
    out = cell_dir(data_root, checkpoint, benchmark, force_mode)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").unlink(missing_ok=True)  # last file written: absent while running
    prov = provider or make_provider(
        mpath, set(spec.prompts), force_cfg, sources_dir, source_cfgs, data_root
    )
    batches = _load_batches(prov)  # every episode is loaded and windowed exactly once

    need_force = force_mode != "omitted"
    consts: dict[str, normalize.NormConstants] = {}
    if need_force:
        consts = _read_consts(data_root, {b.episode.source for b in batches.values()})
    feat_of: dict[str, ResampledForce] = {}
    pool: list[PoolEntry] = []
    if need_force:
        for b in batches.values():
            for wid, w in zip(b.window_ids, b.windows, strict=True):
                feat_of[wid] = window_features(
                    b.episode, w, consts[b.episode.source], force_cfg, b.shift_s
                )
                if force_mode == "shuffled":
                    pool.append(
                        PoolEntry(
                            features=feat_of[wid],
                            source=b.episode.source,
                            episode_id=b.episode.episode_id,
                            prompt_kind=w.prompt_kind,
                        )
                    )

    recs: list[dict[str, Any]] = []
    dense_pred: dict[str, list[tuple[float, list[int]]]] = {}
    dense_truth: dict[str, list[tuple[float, list[int]]]] = {}
    out_pred: list[dict[str, Any]] = []
    out_truth: list[dict[str, Any]] = []
    obj_of: dict[str, str] = {}
    over_comp: dict[str, bool] = {}
    no_logprobs = 0
    k = 0
    with open(out / "log.jsonl", "w", encoding="utf-8") as logf:
        for ekey, b in batches.items():
            ep = b.episode
            cam = ep.cameras.get("scene") or next(iter(ep.cameras.values()))
            obj = str(ep.meta.get("object", ""))
            obj_of[ekey] = obj
            over_comp[ekey] = _has_over_compression(ep)
            for wid, w in zip(b.window_ids, b.windows, strict=True):
                block: str | None = None
                blank = False
                if need_force:
                    ident = PoolEntry(
                        features=feat_of[wid],
                        source=ep.source,
                        episode_id=ep.episode_id,
                        prompt_kind=w.prompt_kind,
                    )
                    abl = apply_force_mode(
                        feat_of[wid], force_mode, pool=pool, seed=seed + k, identity=ident
                    )
                    blank = abl.video_blank
                    if abl.features is not None:
                        lines = render.render_lines(abl.features, w.frame_times, ep.force.channel)
                        block = render.render_block(lines, force_cfg.render) or None
                k += 1
                rationale = w.prompt_kind == "outcome" and not spec.score_only
                bundle = build_prompt(
                    w,
                    block,
                    video_blank=blank,
                    with_rationale=rationale,
                    task=ep.task,
                    camera_roles=tuple(ep.cameras),
                    fps=cam.fps,
                    frame_indices=_frame_indices(w, cam.fps, cam.n_frames),
                )
                dense = w.prompt_kind == "dense"
                regex = dense_regex() if dense else outcome_regex(rationale)
                budget = (
                    DENSE_MAX_NEW_TOKENS
                    if dense
                    else (RATIONALE_MAX_NEW_TOKENS if rationale else SCORE_ONLY_MAX_NEW_TOKENS)
                )
                res = backend.generate(
                    to_messages(bundle),
                    bundle.video,
                    regex=regex,
                    max_new_tokens=budget,
                    top_logprobs=0 if dense else TOP_LOGPROBS,
                    episode=ep,
                )
                parsed: ParsedDense | ParsedOutcome = (
                    parse_dense(res.text) if dense else parse_outcome(res.text)
                )
                es = math.nan
                score = math.nan
                if isinstance(parsed, ParsedDense):
                    dense_pred.setdefault(ekey, []).append((w.anchor_s, parsed.digits))
                    truth = parse_dense(w.target)
                    dense_truth.setdefault(ekey, []).append((w.anchor_s, truth.digits))
                else:
                    t_out = parse_outcome(w.target)
                    if parsed.score is not None:
                        score = float(parsed.score)
                        if res.score_token_logprobs:
                            es = expected_score(res.score_token_logprobs)
                        else:
                            no_logprobs += 1  # NaN, never silently the integer score
                    else:
                        # unparsable: counted in parse_failed AND replaced by the middle score
                        score = float(M.OUTCOME_PARSE_FAIL_SCORE)
                        es = score
                    out_pred.append({"episode_id": ekey, "score": score, "expected_score": es})
                    t_score = float(t_out.score) if t_out.score is not None else math.nan
                    success = bool(t_score >= M.SUCCESS_THRESHOLD)
                    out_truth.append(
                        {
                            "episode_id": ekey,
                            "score": t_score,
                            "label": int(success),
                            "dsr": _dsr(ep, success),
                            "object": obj,
                        }
                    )
                recs.append(
                    {
                        "window_id": wid,
                        "source": ep.source,
                        "episode_id": ep.episode_id,
                        "prompt_kind": w.prompt_kind,
                        "anchor_s": w.anchor_s,
                        "object": obj,
                        "raw_text": res.text,
                        "parsed": parsed.model_dump_json(),
                        "score": score,
                        "expected_score": es,
                        "parse_failed": parsed.parse_failed,
                        "latency_s": res.latency_s,
                        "truth_target": w.target,
                    }
                )
                logf.write(
                    json.dumps(
                        {
                            "window_id": wid,
                            "episode_id": ep.episode_id,
                            "prompt_kind": w.prompt_kind,
                            "parse_failed": parsed.parse_failed,
                            "latency_s": res.latency_s,
                        }
                    )
                    + "\n"
                )
                if parsed.parse_failed:
                    log.warning("parse failed: %s", wid)

    preds = pd.DataFrame(recs)
    preds.to_parquet(out / "predictions.parquet", index=False)

    d_pair = None
    if dense_pred:
        d_pred, d_truth = _stitched(dense_pred), _stitched(dense_truth)
        d_truth["object"] = d_truth["episode_id"].map(obj_of)
        d_truth["over_compression"] = d_truth["episode_id"].map(over_comp).astype(bool)
        d_pair = (d_pred, d_truth)
    o_pair = (pd.DataFrame(out_pred), pd.DataFrame(out_truth)) if out_pred else None
    warnings: list[str] = []
    if o_pair is not None and no_logprobs:
        warnings.append(
            f"{no_logprobs} outcome answers had no score-token logprobs: expected_score is NaN "
            "in predictions; accuracy, AUROC and mae_vs_dsr use the integer score instead"
        )
        log.warning(warnings[-1])
        o_pair = (o_pair[0].assign(expected_score=o_pair[0]["score"]), o_pair[1])
    block_metrics = _metric_block(spec.metrics, d_pair, o_pair)
    per_obj: dict[str, dict[str, float | None]] = {}
    for name in spec.metrics:
        pair, keys = (d_pair, M.DENSE_KEYS) if name in M.DENSE_METRICS else (o_pair, M.OUTCOME_KEYS)
        if pair is None:
            continue
        for o, v in M.per_object(M.METRICS[name], pair[0], pair[1], keys).items():
            per_obj.setdefault(o, {})[name] = _clean(v)
    config = _cell_config(checkpoint, benchmark, force_mode, seed, eval_cfg, force_cfg)
    config["backend"] = backend.name
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), "utf-8")
    sidecar = wcache.cache_path(mpath)
    doc: dict[str, Any] = {
        "checkpoint": checkpoint,
        "benchmark": benchmark,
        "force_mode": force_mode,
        "backend": backend.name,
        "n_windows": len(recs),
        "n_episodes": int(preds["episode_id"].nunique()) if len(preds) else 0,
        "parse_failed": M.parse_failed_count(preds),
        "metrics": block_metrics,
        "per_object": {o: v for o, v in sorted(per_obj.items()) if o},
        "manifest_hash": manifest.manifest_hash(mpath) if mpath.exists() else "none",
        "windows_cache_hash": manifest.manifest_hash(sidecar) if sidecar.exists() else None,
        "git_hash": full_git_hash(),
        "config_hash": _config_hash(config),
        "no_logprobs": no_logprobs,
        "warnings": warnings,
        "seed": seed,
    }
    (out / "metrics.json").write_text(json.dumps(doc, indent=2, sort_keys=True), "utf-8")
    return out


def expand_grid(grid_cfg: GridConfig, eval_cfg: EvalConfig) -> list[tuple[str, str, ForceMode]]:
    """Expand checkpoints x benchmarks x modes, honoring checkpoint_modes and omitted-only."""
    cells: list[tuple[str, str, ForceMode]] = []
    seen: dict[str, str] = {}
    for ckpt in grid_cfg.checkpoints:
        tag = checkpoint_tag(ckpt)
        if seen.setdefault(tag, ckpt) != ckpt:
            raise CheckpointTagCollisionError(
                f"checkpoints {seen[tag]!r} and {ckpt!r} both map to directory tag {tag!r}"
            )
    for ckpt in grid_cfg.checkpoints:
        cap = grid_cfg.checkpoint_modes.get(ckpt)
        for bench in eval_cfg.benchmarks:
            for mode in grid_cfg.modes.get(bench, []):
                check_mode(bench, mode)
                if cap is None or mode in cap:
                    cells.append((ckpt, bench, mode))
    return cells


def run_grid(
    grid_cfg: GridConfig,
    eval_cfg: EvalConfig,
    force_cfg: ForceConfig,
    data_root: Path,
    backend_factory: Callable[[str], Backend],
    *,
    sources_dir: Path = DEFAULT_SOURCES_DIR,
    source_cfgs: Mapping[str, SourceConfig] | None = None,
    provider: Provider | None = None,
) -> list[Path]:
    """Run every grid cell without a current metrics.json; return the cells run.

    A cell is skipped only if its metrics.json (written last) exists with the current manifest,
    windows-cache and config hashes; otherwise it is rerun and the reason logged.
    """
    done: list[Path] = []
    backends: dict[str, Backend] = {}
    for ckpt, bench, mode in expand_grid(grid_cfg, eval_cfg):
        d = cell_dir(data_root, ckpt, bench, mode)
        mpath = _manifest_path(eval_cfg.benchmarks[bench].manifest, data_root)
        cfg_now = _cell_config(ckpt, bench, mode, grid_cfg.seed, eval_cfg, force_cfg)
        why = stale_reason(d, mpath, cfg_now)
        if why is None:
            log.info("skip up-to-date cell %s", d)
            continue
        if (d / "predictions.parquet").exists() or (d / "metrics.json").exists():
            log.warning("rerun cell %s: %s", d, why)
        if ckpt not in backends:
            backends[ckpt] = backend_factory(ckpt)
        done.append(
            run_eval(
                ckpt,
                bench,
                mode,
                eval_cfg,
                force_cfg,
                data_root,
                backends[ckpt],
                seed=grid_cfg.seed,
                sources_dir=sources_dir,
                source_cfgs=source_cfgs,
                provider=provider,
            )
        )
    return done


MODE_ORDER = ("omitted", "zero", "real", "shuffled", "video_blank")


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _headline(names: list[str]) -> str:
    return "failure_recall" if "failure_recall" in names else names[0]


def write_report(eval_cfg: EvalConfig, data_root: Path, out: Path | None = None) -> Path:
    """Render one markdown table per benchmark from the metrics.json files under data/eval.

    Rows are conditions (checkpoint / mode), columns the benchmark's metrics plus parse_failed
    a shuffle-delta column (real minus shuffled on the headline metric: failure_recall when
    present) and an ``omitted_minus_control`` column, filled on the omitted row of each force
    checkpoint (any checkpoint other than released / control) as its omitted headline metric
    minus the control's omitted one (the spec's "must match the control within noise" check).
    Baxter also gets a per-object table. Raises ManifestMismatchError if the cells
    of one benchmark were built from different manifests.
    """
    root = data_root / "eval"
    cells: dict[str, list[dict[str, Any]]] = {b: [] for b in eval_cfg.benchmarks}
    for f in sorted(root.glob("*/*/*/metrics.json")):
        doc = json.loads(f.read_text("utf-8"))
        if doc["benchmark"] in cells:
            cells[doc["benchmark"]].append(doc)
    lines = ["# Evaluation report", ""]
    for bench, docs in cells.items():
        if not docs:
            continue
        hashes = {(d["manifest_hash"], d.get("windows_cache_hash")) for d in docs}
        if len(hashes) > 1:
            raise ManifestMismatchError(
                f"{bench}: predictions come from {len(hashes)} different manifests; rerun the cells"
            )
        names = eval_cfg.benchmarks[bench].metrics
        head = _headline(names)
        docs.sort(
            key=lambda d: (checkpoint_tag(d["checkpoint"]), MODE_ORDER.index(d["force_mode"]))
        )
        by = {(checkpoint_tag(d["checkpoint"]), d["force_mode"]): d for d in docs}
        lines += [
            f"## {bench}",
            "",
            f"manifest hash: `{docs[0]['manifest_hash'][:12]}` (headline metric: {head})",
            "",
        ]
        lines.append(
            "| condition | "
            + " | ".join(
                [*names, "parse_failed", f"shuffle delta ({head})", "omitted_minus_control"]
            )
            + " |"
        )
        lines.append("|" + "---|" * (len(names) + 4))
        for d in docs:
            tag = checkpoint_tag(d["checkpoint"])
            delta = "-"
            if d["force_mode"] == "real" and (tag, "shuffled") in by:
                r, s = d["metrics"].get(head), by[(tag, "shuffled")]["metrics"].get(head)
                if r is not None and s is not None:
                    delta = f"{r - s:+.3f}"
            omit = "-"
            if d["force_mode"] == "omitted" and tag not in ("released", "control"):
                o, c = d["metrics"].get(head), by.get(("control", "omitted"), {}).get("metrics", {})
                if o is not None and c.get(head) is not None:
                    omit = f"{o - c[head]:+.3f}"
            cols = [_fmt(d["metrics"].get(n)) for n in names]
            lines.append(
                f"| {tag} / {d['force_mode']} | "
                + " | ".join([*cols, str(d["parse_failed"]), delta, omit])
                + " |"
            )
        objects = sorted({o for d in docs for o in d.get("per_object", {})})
        if bench == "baxter" and objects:
            lines += ["", f"### baxter per object ({head})", ""]
            lines.append("| condition | " + " | ".join(objects) + " |")
            lines.append("|" + "---|" * (len(objects) + 1))
            for d in docs:
                vals = [_fmt(d.get("per_object", {}).get(o, {}).get(head)) for o in objects]
                tag = checkpoint_tag(d["checkpoint"])
                lines.append(f"| {tag} / {d['force_mode']} | " + " | ".join(vals) + " |")
        lines.append("")
    path = out or (root / "report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), "utf-8")
    return path
