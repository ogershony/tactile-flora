"""Verification pipeline: run every check over a source, report, and build the manifest.

Owns: run_verify, record_spot_pass and run_manifest (the entry points behind ``tf verify``,
``tf verify --spot-pass N --record-only`` and ``tf manifest``), VerifySummary, and the on-disk
layout under data/verified/<source>/ and data/manifest/.
Must never: filter episodes inside adapters, or let training read anything but the manifest.

Layout (``data_root`` is normally ``data``): data/verified/<source>/{norm.json, report.md,
checks.parquet, hashes.parquet, spot/}, data/verified/benchmark_hashes.parquet and
data/manifest/<name>.parquet. Episodes are loaded lazily and more than once (fit, windows,
checks), trading time for memory.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tactile_flora.config import ForceConfig, SourceConfig
from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.schema import (
    CheckResult,
    Episode,
    FrameReader,
    HoldoutFile,
    ResampledForce,
    VerifyContext,
    VerifyThresholds,
    Window,
)
from tactile_flora.sources.base import SourceAdapter, make_adapter
from tactile_flora.verify import checks, hashing, manifest, report
from tactile_flora.verify.windowing import assemble_labels, build_source_inputs, episode_windows

WindowsFn = Callable[[Episode, str], list[Window]]
# Checks a ``verify_from`` child copies from its parent recording instead of re-running: they
# need the whole video decoded (integrity, overlap), the recording's free-space baseline and
# session statistics (sensor_health, normalization) or its full-length force/closure structure
# (sync). Everything else (label_sanity, spot_check, prompt_diff, holdout, license) is cheap
# and is evaluated on the child itself.
INHERITED_CHECKS: frozenset[str] = frozenset(
    {"integrity", "sensor_health", "sync", "normalization", "overlap"}
)


class VerifySummary(BaseModel):
    """Outcome of one ``run_verify``."""

    source: str
    n_episodes: int
    episode_ok: dict[str, bool]
    episode_failures: dict[str, list[str]]
    source_verdicts: dict[str, bool]
    source_ok: bool
    shifts: dict[str, float] = Field(default_factory=lambda: dict[str, float]())
    sync_median_s: float | None = None
    report_path: str
    norm_path: str | None = None
    norm_error: str | None = None
    # episodes that fed the normalization fit (those without meta.exclude_from_norm_fit)
    norm_fit_episodes: int = 0


def full_git_hash() -> str:
    """Return ``git rev-parse HEAD`` or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def verified_dir(data_root: Path, source: str) -> Path:
    """Return data/verified/<source>."""
    return data_root / "verified" / source


def holdout_path(data_root: Path) -> Path:
    """Return data/manifest/holdout_ids.json."""
    return data_root / "manifest" / "holdout_ids.json"


def _adapter(cfg: SourceConfig) -> SourceAdapter:
    return make_adapter(cfg.name, cfg)


def _frame_times(ep: Episode, period: float) -> list[float]:
    return [k * period for k in range(int(ep.duration_s / period) + 1)]


def _load_holdout(data_root: Path) -> HoldoutFile | None:
    p = holdout_path(data_root)
    return manifest.load_holdout(p) if p.exists() else None


def _thresholds(cfg: SourceConfig) -> VerifyThresholds:
    """Spec thresholds with the source's own ``sync_median_s`` / ``norm_p99_max`` overrides."""
    over = {"sync_median_s": cfg.sync_median_s, "norm_p99_max": cfg.norm_p99_max}
    return VerifyThresholds().model_copy(update={k: v for k, v in over.items() if v is not None})


def norm_fit_items(
    items: Sequence[tuple[str, ResampledForce, bool]],
) -> list[tuple[str, ResampledForce]]:
    """Select the (group, features) pairs that feed the normalization fit.

    ``items`` are (group, features, excluded) per discovered episode, ``excluded`` being the
    adapter's ``meta["exclude_from_norm_fit"]`` (PHANTOM failure demos: their over-squeeze or
    under-grasp must not define the "typical grasp"). The constants are still applied to every
    episode, so a group whose every episode is excluded has nothing to normalize against and
    raises NormError (the source then fails the normalization check) rather than being fitted on
    its failure demos.
    """
    kept = [(g, f) for g, f, excluded in items if not excluded]
    fitted = {g for g, _ in kept}
    orphans = sorted({g for g, _, excluded in items if excluded and g not in fitted})
    if orphans:
        raise normalize.NormError(
            f"groups {orphans} have only episodes excluded from the normalization fit"
        )
    return kept


class _DeclaredCountReader:
    """FrameReader whose ``frame_count`` trusts ``VideoRef.n_frames``; decoding is delegated.

    Used when a source inherits its integrity results from ``verify_from``: the parent already
    decoded every file in full, and the spot strips only need a few frames of each clip.
    """

    def __init__(self, inner: FrameReader) -> None:
        self.inner = inner

    def frame_count(self, episode: Episode, role: str) -> int:
        """Return the declared count (verified on the parent recording)."""
        return episode.cameras[role].n_frames

    def read(self, episode: Episode, role: str, index: int) -> np.ndarray:
        """Delegate."""
        return self.inner.read(episode, role, index)

    def read_many(self, episode: Episode, role: str, indices: Sequence[int]) -> list[np.ndarray]:
        """Delegate."""
        return self.inner.read_many(episode, role, indices)


@dataclass
class _Parent:
    """What a ``verify_from`` child reads from data/verified/<parent>/."""

    source: str
    results: dict[str, dict[str, CheckResult]]
    consts: normalize.NormConstants
    normalization: CheckResult  # the parent's POOLED stats, given to every child episode


def load_parent(cfg: SourceConfig, data_root: Path) -> _Parent:
    """Read the parent's checks.parquet and norm.json for a ``verify_from`` source.

    The child's normalization result is the parent's pooled (median over its fit episodes)
    grasp / free-space / p99 stats as one-element lists, so ``report.source_verdicts`` pools
    them back to the parent's verdict whatever the number of child episodes per recording.
    """
    if not cfg.verify_from:
        raise ValueError("load_parent needs cfg.verify_from")
    pdir = verified_dir(data_root, cfg.verify_from)
    if not (pdir / "checks.parquet").exists() or not (pdir / "norm.json").exists():
        raise RuntimeError(f"verify_from {cfg.verify_from!r}: run tf verify on it first ({pdir})")
    results = report.read_checks(pdir)
    seen = [r["normalization"] for r in results.values() if "normalization" in r]
    stats = report.pooled_normalization_stats(seen) or {}
    doc: dict[str, object] = {k: [stats[k]] for k in ("grasp", "free", "p99") if k in stats}
    doc["problems"] = []
    doc["inherited_from"] = cfg.verify_from
    return _Parent(
        source=cfg.verify_from,
        results=results,
        consts=normalize.load(pdir / "norm.json"),
        normalization=CheckResult(
            passed=True, value=stats.get("grasp"), detail=json.dumps(doc, separators=(",", ":"))
        ),
    )


def inherit_results(
    cfg: SourceConfig, parent: _Parent, ep: Episode, own: dict[str, CheckResult]
) -> dict[str, CheckResult]:
    """Merge the child's own cheap results with the parent's INHERITED_CHECKS, in check order.

    The parent episode is ``ep.meta["parent_episode_id"]``; a child without a verified parent
    fails integrity (drop_episode). With ``cfg.inherited_drops == "record"`` an inherited
    failure keeps its ``passed=False`` and detail but its action becomes "keep".
    """
    pid = str(ep.meta.get("parent_episode_id", ""))
    pres = parent.results.get(pid)
    out: dict[str, CheckResult] = {}
    for name in checks.CHECK_NAMES:
        if name not in INHERITED_CHECKS:
            out[name] = own[name]
            continue
        if pres is None or (name != "normalization" and name not in pres):
            why = f"no {name} result for parent {parent.source}/{pid or '?'} to inherit"
            out[name] = CheckResult(passed=False, action="drop_episode", detail=why)
            continue
        if name == "normalization":
            out[name] = parent.normalization
            continue
        r = pres[name]
        detail = f"inherited from {parent.source}/{pid}" + (f": {r.detail}" if r.detail else "")
        action = r.action
        if (
            not r.passed
            and action in ("drop_episode", "drop_channel")
            and cfg.inherited_drops == "record"
        ):
            action = "keep"
            detail += " (recorded, not applied: inherited_drops=record)"
        out[name] = CheckResult(passed=r.passed, value=r.value, action=action, detail=detail)
    return out


def run_verify(
    cfg: SourceConfig,
    force_cfg: ForceConfig,
    data_root: Path,
    reader: FrameReader,
    *,
    windows_fn: WindowsFn | None = None,
    spot_pass: int | None = None,
) -> VerifySummary:
    """Run every check over every discovered episode of ``cfg`` and write the report.

    ``windows_fn(episode, split)`` supplies candidate windows (default: episode_windows with
    force rendered from the fitted constants). ``spot_pass`` records the human reviewer's pass
    count into the spot-check index before the checks run.

    With ``cfg.verify_from`` (a verified parent source, e.g. reassemble_lerobot for
    reassemble_seg) pass 1 is skipped: the parent's norm.json is copied, INHERITED_CHECKS are
    read from its checks.parquet per ``meta["parent_episode_id"]`` (``inherit_results``), no
    video is decoded except for the spot strips, and frame hashes are not computed.
    """
    adapter = _adapter(cfg)
    out = verified_dir(data_root, cfg.name)
    thresholds = _thresholds(cfg)
    th_ctx = VerifyContext(source=cfg.name, thresholds=thresholds)
    ids = sorted(adapter.discover(Path(cfg.root)))
    splits = adapter.splits()
    parent = load_parent(cfg, data_root) if cfg.verify_from else None
    if parent is not None:
        reader = _DeclaredCountReader(reader)

    # pass 1: fit normalization (per norm_group) over every discovered episode that the
    # adapter did not exclude from the fit, gather label and baseline stats
    feats: list[tuple[str, ResampledForce, bool]] = []
    labeled: list[bool] = []
    base_by_key: dict[str, list[float]] = {}
    for eid in ids if parent is None else []:
        ep = adapter.load(eid)
        feats.append(
            (
                normalize.norm_group(ep),
                resample(
                    ep.force,
                    ep.gripper,
                    _frame_times(ep, force_cfg.resample.frame_period_s),
                    force_cfg.resample,
                ),
                bool(ep.meta.get("exclude_from_norm_fit", False)),
            )
        )
        if ep.labels:
            labeled.append(checks.has_bad_label(ep))
        for k, v in checks.episode_baselines(ep, th_ctx.thresholds.baseline_window_s).items():
            base_by_key.setdefault(f"{ep.session}/{k}", []).append(v)
    consts: normalize.NormConstants | None = None
    norm_error: str | None = None
    norm_path: Path | None = None
    norm_fit_n = 0
    if parent is not None:
        consts = parent.consts
        norm_path = out / "norm.json"
        normalize.save(consts, norm_path)
        feats = []
    try:
        fit_items = norm_fit_items(feats) if parent is None else []
        norm_fit_n = len(fit_items)
        if parent is None:
            consts = normalize.fit(
                fit_items,
                force_cfg.resample,
                source=cfg.name,
                grasp_g=force_cfg.normalize.grasp_g,
                min_group_frac=force_cfg.normalize.min_group_frac,
            )
            norm_path = out / "norm.json"
            normalize.save(consts, norm_path)
    except normalize.NormError as exc:
        norm_error = str(exc)
    del feats
    baseline_stats = {
        k: (float(np.mean(v)), float(np.std(v))) for k, v in base_by_key.items() if len(v) >= 3
    }
    obs_rate = sum(labeled) / len(labeled) if labeled else None

    hash_path = out / "hashes.parquet"
    cache = hashing.load_hash_cache(hash_path)
    spot_dir = out / "spot"
    if spot_pass is not None:
        if checks.read_spot_index(spot_dir)[0] == 0:
            raise RuntimeError("no spot strips yet; run verify once, review, then pass --spot-pass")
        checks.record_spot_review(spot_dir, spot_pass)
    n_spot, reviewer = checks.read_spot_index(spot_dir)
    holdout = _load_holdout(data_root)
    ctx = VerifyContext(
        source=cfg.name,
        holdout_ids=holdout.held_out_ids if holdout else frozenset(),
        license_allowed=cfg.license_allowed,
        seed=cfg.seed,
        reader=reader,
        norm=consts,
        thresholds=thresholds,
        resample_cfg=force_cfg.resample,
        render_cfg=force_cfg.render,
        benchmark_index=hashing.load_index(data_root / "verified" / "benchmark_hashes.parquet"),
        hash_cache=cache,
        expected_label_rate=cfg.expected_label_rate,
        observed_label_rate=obs_rate,
        spot_n=n_spot,
        spot_reviewer_pass=reviewer,
        baseline_stats=baseline_stats,
    )
    # label inputs (PHANTOM per-object p99, pseudo labels); windows built here are unshifted
    # because the sync check has not run yet (the manifest and training paths apply the shift)
    inputs = build_source_inputs(adapter, force_cfg, consts, out, ids)
    to_windows = windows_fn or (
        lambda ep, split: episode_windows(ep, force_cfg, consts, split, inputs=inputs)
    )
    # the observed label rate must count the ASSEMBLED labels (native + mapped + events +
    # pseudo), not only what load() returned, which is empty for the real adapters
    if windows_fn is None:
        labeled = []
        for eid in ids:
            ep = adapter.load(eid)
            labs = assemble_labels(ep, force_cfg, consts, inputs)
            if labs:
                labeled.append(checks.has_bad_label(ep.model_copy(update={"labels": labs})))
        obs_rate = sum(labeled) / len(labeled) if labeled else None
        ctx = ctx.model_copy(
            update={"observed_label_rate": obs_rate, "labeled_episode_count": len(labeled)},
            deep=False,
        )

    # pass 2: windows and checks per episode
    results: dict[str, dict[str, CheckResult]] = {}
    pairs: list[tuple[str, Window]] = []
    for eid in ids:
        ep = adapter.load(eid)
        split = manifest.assign_split(ep, splits)
        wins = to_windows(ep, split)
        ep_ctx = ctx.model_copy(
            update={"windows": {eid: wins}, "split_of": {eid: split}}, deep=False
        )
        if parent is None:
            # cache first/middle/last hashes for the manifest even without a benchmark index;
            # an undecodable video is reported by the integrity check instead
            with contextlib.suppress(Exception):
                hashing.cached_hashes(ep, reader, cache)
            results[eid] = checks.run_checks(ep, ep_ctx)
        else:
            own = {
                name: fn(ep, ep_ctx)
                for name, fn in zip(checks.CHECK_NAMES, checks.ALL_CHECKS, strict=True)
                if name not in INHERITED_CHECKS
            }
            results[eid] = inherit_results(cfg, parent, ep, own)
        # spot strips decode frames without a guard, so only episodes whose video passed the
        # integrity check are candidates (a missing mp4 killed a 41-minute RH20T run otherwise)
        integrity = results[eid].get("integrity")
        if integrity is None or integrity.passed:
            pairs.extend((eid, w) for w in wins)
    # spot strips are source-level: sample once, then re-evaluate spot_check for every episode
    if n_spot == 0 and pairs:
        picked = _pick(pairs, cfg.seed, ctx.thresholds.spot_n)
        # episode ids such as "rh20t_cfg3/0" contain a slash; the strip file name must not
        items = [
            (adapter.load(eid).model_copy(update={"episode_id": eid.replace("/", "_")}), w)
            for eid, w in picked
        ]
        checks.write_spot_strips(cfg.name, items, reader, spot_dir, ctx.thresholds.spot_n, cfg.seed)
        n_spot, reviewer = checks.read_spot_index(spot_dir)
        sctx = ctx.model_copy(update={"spot_n": n_spot, "spot_reviewer_pass": reviewer})
        for eid in ids:
            results[eid]["spot_check"] = checks.check_spot_check(adapter.load(eid), sctx)
    if cache:
        hashing.save_hash_cache(hash_path, cache)
    return _finish(cfg, results, out, thresholds, consts, norm_error, norm_fit_n)


def _finish(
    cfg: SourceConfig,
    results: dict[str, dict[str, CheckResult]],
    out: Path,
    thresholds: VerifyThresholds,
    consts: normalize.NormConstants | None,
    norm_error: str | None,
    norm_fit_n: int,
) -> VerifySummary:
    """Write report.md and checks.parquet from ``results`` and derive the source verdicts."""
    report_path = report.write_report(
        cfg.name,
        results,
        out,
        sync_median_s=thresholds.sync_median_s,
        norm_p99_max=thresholds.norm_p99_max,
        norm=consts,
        norm_grasp_range=cfg.norm_grasp_range,
    )
    verdicts = report.source_verdicts(
        results,
        sync_median_s=thresholds.sync_median_s,
        norm_p99_max=thresholds.norm_p99_max,
        norm_grasp_range=cfg.norm_grasp_range,
    )
    lags = [
        abs(r["sync"].value)
        for r in results.values()
        if "sync" in r and r["sync"].value is not None
    ]
    median_lag = float(np.median(lags)) if lags else None
    if median_lag is not None:
        verdicts["sync_median"] = median_lag < thresholds.sync_median_s
    if norm_error is not None:
        verdicts["normalization"] = False
    failures = {eid: [n for n, r in per.items() if not r.passed] for eid, per in results.items()}
    return VerifySummary(
        source=cfg.name,
        n_episodes=len(results),
        episode_ok={
            eid: not any(not r.passed and r.action == "drop_episode" for r in per.values())
            for eid, per in results.items()
        },
        episode_failures={e: f for e, f in failures.items() if f},
        source_verdicts=verdicts,
        source_ok=all(verdicts.values()),
        shifts={
            eid: r["sync"].value
            for eid, r in results.items()
            if "sync" in r and r["sync"].action == "shift" and r["sync"].value is not None
        },
        sync_median_s=median_lag,
        report_path=str(report_path),
        norm_path=str(out / "norm.json") if consts is not None else None,
        norm_error=norm_error,
        norm_fit_episodes=norm_fit_n,
    )


def record_spot_pass(cfg: SourceConfig, data_root: Path, spot_pass: int) -> VerifySummary:
    """Record the reviewer's spot-check pass count and re-score only ``spot_check``.

    Every other check is read back from checks.parquet exactly as verified (no video is decoded
    and no episode loaded), the count goes to spot/index.json, the spot gate is re-evaluated for
    every episode, and report.md and checks.parquet are rewritten. This is
    ``tf verify --spot-pass N --record-only``: the human verdict is the only new fact, and a full
    re-verify of a large source (rh20t: about 2 M AV1 frames) decodes every video again.
    ``norm_fit_episodes`` is not known here and is reported as 0.
    """
    out = verified_dir(data_root, cfg.name)
    if not (out / "checks.parquet").exists():
        raise RuntimeError(f"{out / 'checks.parquet'} does not exist; run tf verify first")
    results = report.read_checks(out)
    spot_dir = out / "spot"
    if checks.read_spot_index(spot_dir)[0] == 0:
        raise RuntimeError("no spot strips yet; run verify once, review, then pass --spot-pass")
    checks.record_spot_review(spot_dir, spot_pass)
    n_spot, reviewer = checks.read_spot_index(spot_dir)
    thresholds = _thresholds(cfg)
    verdict = checks.spot_verdict(
        VerifyContext(
            source=cfg.name, thresholds=thresholds, spot_n=n_spot, spot_reviewer_pass=reviewer
        )
    )
    for per in results.values():
        per["spot_check"] = verdict
    norm_path = out / "norm.json"
    consts = normalize.load(norm_path) if norm_path.exists() else None
    norm_error = None if consts is not None else "no norm.json from the last verify"
    return _finish(cfg, results, out, thresholds, consts, norm_error, 0)


def _pick(pairs: list[tuple[str, Window]], seed: int, n: int) -> list[tuple[str, Window]]:
    import random

    ordered = sorted(pairs, key=lambda p: (p[0], p[1].prompt_kind, p[1].anchor_s))
    return random.Random(seed).sample(ordered, min(n, len(ordered)))


def run_manifest(
    cfg: SourceConfig,
    force_cfg: ForceConfig,
    data_root: Path,
    *,
    name: str = "train",
    report_version: str | None = None,
    windows_fn: WindowsFn | None = None,
) -> Path:
    """Build data/manifest/<name>.parquet for ``cfg`` from its saved checks and norm.json.

    Reads the verify outputs (never re-runs checks), so run ``tf verify`` first. Honors the
    frozen holdout file if present.
    """
    adapter = _adapter(cfg)
    out = verified_dir(data_root, cfg.name)
    results = report.read_checks(out)
    consts = normalize.load(out / "norm.json")
    version = consts.version()
    splits = adapter.splits()
    inputs = build_source_inputs(adapter, force_cfg, consts, out, sorted(results), results)
    to_windows = windows_fn or (
        lambda ep, split: episode_windows(ep, force_cfg, consts, split, inputs=inputs)
    )

    def _pairs():
        for eid in sorted(results):
            ep = adapter.load(eid)
            yield ep, to_windows(ep, manifest.assign_split(ep, splits))

    path = data_root / "manifest" / f"{name}.parquet"
    manifest.build_manifest(
        cfg,
        _pairs(),
        results,
        version,
        report_version or full_git_hash(),
        _load_holdout(data_root),
        splits,
        path,
        hashes=hashing.load_hash_cache(out / "hashes.parquet"),
    )
    return path
