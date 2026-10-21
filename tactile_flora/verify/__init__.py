"""Verification pipeline: run every check over a source, report, and build the manifest.

Owns: run_verify and run_manifest (the entry points behind ``tf verify`` and ``tf manifest``),
VerifySummary, and the on-disk layout under data/verified/<source>/ and data/manifest/.
Must never: filter episodes inside adapters, or let training read anything but the manifest.

Layout (``data_root`` is normally ``data``): data/verified/<source>/{norm.json, report.md,
checks.parquet, hashes.parquet, spot/}, data/verified/benchmark_hashes.parquet and
data/manifest/<name>.parquet. Episodes are loaded lazily and more than once (fit, windows,
checks), trading time for memory.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Callable
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
    Window,
)
from tactile_flora.sources.base import SourceAdapter, get_source
from tactile_flora.verify import checks, hashing, manifest, report
from tactile_flora.verify.windowing import episode_windows

WindowsFn = Callable[[Episode, str], list[Window]]


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
    return get_source(cfg.name)()


def _frame_times(ep: Episode, period: float) -> list[float]:
    return [k * period for k in range(int(ep.duration_s / period) + 1)]


def _load_holdout(data_root: Path) -> HoldoutFile | None:
    p = holdout_path(data_root)
    return manifest.load_holdout(p) if p.exists() else None


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
    """
    adapter = _adapter(cfg)
    out = verified_dir(data_root, cfg.name)
    th_ctx = VerifyContext(source=cfg.name)
    ids = sorted(adapter.discover(Path(cfg.root)))
    splits = adapter.splits()

    # pass 1: fit normalization over every discovered episode, gather label and baseline stats
    feats: list[ResampledForce] = []
    labeled: list[bool] = []
    base_by_key: dict[str, list[float]] = {}
    for eid in ids:
        ep = adapter.load(eid)
        feats.append(
            resample(
                ep.force,
                ep.gripper,
                _frame_times(ep, force_cfg.resample.frame_period_s),
                force_cfg.resample,
            )
        )
        if ep.labels:
            labeled.append(checks.has_bad_label(ep))
        for k, v in checks.episode_baselines(ep, th_ctx.thresholds.baseline_window_s).items():
            base_by_key.setdefault(f"{ep.session}/{k}", []).append(v)
    consts: normalize.NormConstants | None = None
    norm_error: str | None = None
    norm_path: Path | None = None
    try:
        consts = normalize.fit(feats, force_cfg.resample, source=cfg.name)
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
    to_windows = windows_fn or (
        lambda ep, split: episode_windows(ep, force_cfg, consts, split)  # type: ignore[arg-type]
    )

    # pass 2: windows and checks per episode
    results: dict[str, dict[str, CheckResult]] = {}
    pairs: list[tuple[str, Window]] = []
    for eid in ids:
        ep = adapter.load(eid)
        split = manifest.assign_split(ep, splits)
        wins = to_windows(ep, split)
        pairs.extend((eid, w) for w in wins)
        ep_ctx = ctx.model_copy(
            update={"windows": {eid: wins}, "split_of": {eid: split}}, deep=False
        )
        # cache first/middle/last hashes for the manifest even without a benchmark index;
        # an undecodable video is reported by the integrity check instead
        with contextlib.suppress(Exception):
            hashing.cached_hashes(ep, reader, cache)
        results[eid] = checks.run_checks(ep, ep_ctx)
    # spot strips are source-level: sample once, then re-evaluate spot_check for every episode
    if n_spot == 0 and pairs:
        picked = _pick(pairs, cfg.seed, ctx.thresholds.spot_n)
        items = [(adapter.load(eid), w) for eid, w in picked]
        checks.write_spot_strips(cfg.name, items, reader, spot_dir, ctx.thresholds.spot_n, cfg.seed)
        n_spot, reviewer = checks.read_spot_index(spot_dir)
        sctx = ctx.model_copy(update={"spot_n": n_spot, "spot_reviewer_pass": reviewer})
        for eid in ids:
            results[eid]["spot_check"] = checks.check_spot_check(adapter.load(eid), sctx)
    if cache:
        hashing.save_hash_cache(hash_path, cache)

    report_path = report.write_report(cfg.name, results, out)
    verdicts = report.source_verdicts(results)
    lags = [
        abs(r["sync"].value)
        for r in results.values()
        if "sync" in r and r["sync"].value is not None
    ]
    median_lag = float(np.median(lags)) if lags else None
    if median_lag is not None:
        verdicts["sync_median"] = median_lag < ctx.thresholds.sync_median_s
    if norm_error is not None:
        verdicts["normalization"] = False
    failures = {eid: [n for n, r in per.items() if not r.passed] for eid, per in results.items()}
    return VerifySummary(
        source=cfg.name,
        n_episodes=len(ids),
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
        norm_path=str(norm_path) if norm_path else None,
        norm_error=norm_error,
    )


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
    to_windows = windows_fn or (
        lambda ep, split: episode_windows(ep, force_cfg, consts, split)  # type: ignore[arg-type]
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
