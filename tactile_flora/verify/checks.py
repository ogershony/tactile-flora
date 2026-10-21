"""One function per verification check, each returning a CheckResult.

Owns: check_<name>(episode, ctx) for integrity, sensor_health, sync, normalization,
label_sanity, spot_check, prompt_diff, holdout, overlap and license; ALL_CHECKS (in spec order);
run_checks; write_spot_strips and record_spot_review (the human spot-check artifacts).
Must never: mutate an episode, decide the source verdict (report/verify do), or load datasets.

Checks are deterministic given the episode and the context. Source-level checks (normalization,
label_sanity, spot_check, prompt_diff) return action "keep": a failure fails the source, not just
the episode (see SOURCE_CHECKS).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from tactile_flora.config import ResampleConfig
from tactile_flora.force import normalize
from tactile_flora.force.render import TokenBudgetError, budget_check
from tactile_flora.force.resample import resample
from tactile_flora.prompts.flora import prompt_diff
from tactile_flora.schema import (
    CheckResult,
    Episode,
    PadArmForce,
    VerifyContext,
    Window,
    episode_key,
)
from tactile_flora.verify import hashing

Check = Callable[[Episode, VerifyContext], CheckResult]

SPOT_INDEX = "index.json"
EVAL_SPLITS = ("val", "test", "benchmark")
_DEFAULT_RESAMPLE = ResampleConfig(
    half_window_s=0.125,
    frame_period_s=0.25,
    group_period_s=0.5,
    raw_rate_hz=100.0,
    contact_sigma=3.0,
)
_GRIPPER_CLOSED = 0.6


def _ok(value: float | None = None, detail: str = "", action: str = "keep") -> CheckResult:
    return CheckResult(passed=True, value=value, action=action, detail=detail)  # type: ignore[arg-type]


def _fail(action: str, detail: str, value: float | None = None) -> CheckResult:
    return CheckResult(passed=False, value=value, action=action, detail=detail)  # type: ignore[arg-type]


def _frame_times(duration: float, step: float) -> list[float]:
    return [k * step for k in range(int(duration / step) + 1)]


# --- integrity -------------------------------------------------------------------------------


def check_integrity(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Require decodable videos, matching counts, monotonic times and the declared force rate."""
    th = ctx.thresholds
    problems: list[str] = []
    for role, ref in episode.cameras.items():
        if ctx.reader is None:
            problems.append(f"no frame reader to decode {role}")
            continue
        try:
            n = ctx.reader.frame_count(episode, role)
        except Exception as exc:  # decode errors come in many types
            problems.append(f"{role} does not decode ({type(exc).__name__}: {exc})")
            continue
        if n != ref.n_frames:
            problems.append(f"{role} decoded {n} frames, declared {ref.n_frames}")
        if abs(n / ref.fps - episode.duration_s) > th.rate_tol_frames / ref.fps:
            problems.append(
                f"{role} video {n / ref.fps:.2f}s vs duration {episode.duration_s:.2f}s"
            )
    f = episode.force
    t = f.t_array
    if t.size < 2:
        problems.append("fewer than 2 force samples")
    else:
        if not np.all(np.diff(t) > 0):
            problems.append("force timestamps not strictly increasing")
        else:
            dt = float(np.median(np.diff(t)))
            if abs(dt * f.rate_hz - 1) > th.rate_tol_frac:
                problems.append(f"force median dt {dt:.4f}s vs declared {f.rate_hz} Hz")
    if not f.arms:
        problems.append("no force arms")
    for arm, data in f.arms.items():
        if data.n_samples() != t.size:
            problems.append(f"arm {arm} has {data.n_samples()} rows for {t.size} timestamps")
    g = episode.gripper
    if g is not None and np.any(np.diff(np.asarray(g.t)) <= 0):
        problems.append("gripper timestamps not strictly increasing")
    if problems:
        return _fail("drop_episode", "; ".join(problems))
    return _ok()


# --- sensor health -----------------------------------------------------------------------------


def _channels(episode: Episode) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for arm, data in episode.force.arms.items():
        if isinstance(data, PadArmForce):
            out[(arm, "f_a")] = data.f_a_array
            out[(arm, "f_b")] = data.f_b_array
        else:
            for name in ("fx", "fy", "fz", "tx", "ty", "tz"):
                out[(arm, name)] = np.asarray(getattr(data, name), dtype=np.float64)
    return out


def episode_baselines(episode: Episode, window_s: float) -> dict[str, float]:
    """Median of each channel over the first ``window_s`` seconds, keyed "arm/channel"."""
    t = episode.force.t_array
    out: dict[str, float] = {}
    for (arm, name), x in _channels(episode).items():
        if x.size:
            sel = x[t < t[0] + window_s] if t.size else x
            out[f"{arm}/{name}"] = float(np.median(sel if sel.size else x))
    return out


def check_sensor_health(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Per channel: variance above the noise floor, saturation under 1%, baseline within 2 sigma.

    Wrist torque channels may legitimately stay near zero, so for wrist streams only the force
    components fx, fy, fz are required to vary; saturation and baseline still apply to all.
    A failed channel gives drop_channel; if every arm has lost all its channels, drop_episode.
    """
    th = ctx.thresholds
    t = episode.force.t_array
    failed: dict[str, list[str]] = {}
    for (arm, name), x in _channels(episode).items():
        why: list[str] = []
        required = episode.force.channel == "pad" or name in ("fx", "fy", "fz")
        if required and (x.size == 0 or float(np.std(x)) <= th.noise_floor):
            why.append("flat")
        if x.size and float(np.std(x)) > 0:
            top = float(np.mean(x == x.max()))
            bottom = float(np.mean(x == x.min()))
            if max(top, bottom) > th.saturation_frac:
                why.append(f"saturated {max(top, bottom):.1%}")
        if x.size:
            sel = x[t < t[0] + th.baseline_window_s] if t.size else x
            base = float(np.median(sel if sel.size else x))
            stats = ctx.baseline_stats.get(f"{episode.session}/{arm}/{name}")
            if stats is not None:
                mean, std = stats
                if abs(base - mean) > th.baseline_sigma * max(std, th.noise_floor):
                    why.append(f"baseline {base:.3g} off session mean {mean:.3g}")
        if why:
            failed.setdefault(arm, []).append(f"{name}: {', '.join(why)}")
    if not failed:
        return _ok()
    n_ch = {arm: sum(1 for (a, _) in _channels(episode) if a == arm) for arm in episode.force.arms}
    dead = [arm for arm, items in failed.items() if len(items) >= n_ch[arm]]
    detail = "; ".join(f"{arm}: {', '.join(items)}" for arm, items in sorted(failed.items()))
    nfail = float(sum(len(v) for v in failed.values()))
    if len(dead) == len(episode.force.arms):
        return _fail("drop_episode", detail, nfail)
    return _fail("drop_channel", detail, nfail)


# --- sync --------------------------------------------------------------------------------------


def _xcorr_lag(f: np.ndarray, g: np.ndarray, step: float, search_s: float) -> float | None:
    """Lag in seconds of ``f`` behind ``g`` (positive = force trails closure), None if undefined."""
    if f.size < 4 or float(np.std(f)) < 1e-9 or float(np.std(g)) < 1e-9:
        return None
    f = (f - f.mean()) / f.std()
    g = (g - g.mean()) / g.std()
    full = np.correlate(f, g, mode="full")
    lags = np.arange(-(g.size - 1), f.size)
    keep = np.abs(lags * step) <= search_s
    return float(lags[keep][int(np.argmax(full[keep]))] * step)


def check_sync(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Cross-correlate resampled F with gripper closure; value is the signed lag in seconds.

    Positive lag means force trails the gripper. Passes with action "shift" (lag stored in
    ``value``, to be applied by the resampler) when max |lag| over arms is under sync_max_s and
    at least one grid step, "keep" when negligible, and drop_episode otherwise. The source-level
    median bound (sync_median_s) is evaluated by the runner over all episodes.
    """
    th = ctx.thresholds
    if episode.gripper is None:
        return _ok(detail="no gripper stream; sync not checkable")
    cfg = ctx.resample_cfg or _DEFAULT_RESAMPLE
    times = _frame_times(episode.duration_s, th.sync_grid_s)
    feats = resample(episode.force, episode.gripper, times, cfg)
    lags: dict[str, float] = {}
    for arm, af in feats.arms.items():
        if af.g is None:
            continue
        lag = _xcorr_lag(
            np.nan_to_num(np.asarray(af.F)), np.asarray(af.g), th.sync_grid_s, th.sync_search_s
        )
        if lag is not None:
            lags[arm] = lag
    if not lags:
        return _ok(detail="no closure or no force variation; sync not checkable")
    worst = max(lags.values(), key=abs)
    detail = ", ".join(f"{a}={v * 1000:+.0f}ms" for a, v in sorted(lags.items()))
    if abs(worst) >= th.sync_max_s:
        return _fail("drop_episode", f"lag {detail} exceeds {th.sync_max_s * 1000:.0f}ms", worst)
    return _ok(worst, detail, "shift" if abs(worst) >= th.sync_grid_s else "keep")


# --- normalization -----------------------------------------------------------------------------


def check_normalization(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """After applying constants: grasp near 1.0, free space near 0.0, 99th percentile under 3."""
    th = ctx.thresholds
    if ctx.norm is None or ctx.render_cfg is None:
        return _fail("keep", "no normalization constants in context")
    cfg = ctx.resample_cfg or _DEFAULT_RESAMPLE
    times = _frame_times(episode.duration_s, cfg.frame_period_s)
    feats = normalize.apply(
        resample(episode.force, episode.gripper, times, cfg), ctx.norm, ctx.render_cfg
    )
    problems: list[str] = []
    grasp_meds: list[float] = []
    for arm, af in feats.arms.items():
        F = np.asarray(af.F)
        ok = ~np.isnan(F)
        if ok.any() and float(np.percentile(F[ok], 99)) >= th.norm_p99_max:
            problems.append(f"{arm} p99 {np.percentile(F[ok], 99):.2f} >= {th.norm_p99_max}")
        free = ok & (np.asarray(times) < cfg.group_period_s)
        if free.any() and abs(float(np.median(F[free]))) > th.norm_free_tol:
            problems.append(f"{arm} free-space median {np.median(F[free]):.2f}")
        if af.g is not None:
            held = ok & (np.asarray(af.g) > _GRIPPER_CLOSED) & (np.asarray(af.c) == 1)
            if held.any():
                med = float(np.median(F[held]))
                grasp_meds.append(med)
                if abs(med - 1.0) > th.norm_grasp_tol:
                    problems.append(f"{arm} grasp median {med:.2f}")
    value = float(np.median(grasp_meds)) if grasp_meds else None
    if problems:
        return _fail("keep", "; ".join(problems), value)
    return _ok(value, "" if grasp_meds else "no grasp frames; grasp level not checked")


# --- label sanity ------------------------------------------------------------------------------


def has_bad_label(episode: Episode) -> bool:
    """Return True if the episode has an outcome of 1-2 or any span below quality 3."""
    for lab in episode.labels:
        if lab.outcome is not None and lab.outcome <= 2:
            return True
        if any(s.quality < 3 for s in lab.spans):
            return True
    return False


def observed_label_rate(episodes: Iterable[Episode]) -> float | None:
    """Fraction of labeled episodes that carry a bad label; None if no episode is labeled."""
    labeled = [e for e in episodes if e.labels]
    if not labeled:
        return None
    return sum(has_bad_label(e) for e in labeled) / len(labeled)


def check_label_sanity(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Observed bad-label rate within label_rate_factor of the configured expected rate."""
    exp, obs = ctx.expected_label_rate, ctx.observed_label_rate
    if exp is None or obs is None:
        return _ok(obs, "no expected rate or no labels to compare")
    k = ctx.thresholds.label_rate_factor
    good = obs == 0 if exp == 0 else (exp / k <= obs <= exp * k)
    detail = f"observed {obs:.3f} vs expected {exp:.3f} (factor {k})"
    return _ok(obs, detail) if good else _fail("keep", detail, obs)


# --- spot check --------------------------------------------------------------------------------


def write_spot_strips(
    source: str,
    items: Sequence[tuple[Episode, Window]],
    reader: object,
    out_dir: Path,
    n: int = 20,
    seed: int = 0,
) -> Path:
    """Sample ``n`` windows (seeded) and write one PNG label strip each plus index.json.

    A strip is 8 evenly spaced scene frames above the window's target digits. index.json lists
    the files and holds ``reviewer_pass`` (null until a human records it via record_spot_review).
    """
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    ordered = sorted(items, key=lambda it: (it[0].episode_id, it[1].prompt_kind, it[1].anchor_s))
    picked = rng.sample(ordered, min(n, len(ordered)))
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = 96
    files: list[str] = []
    for k, (ep, win) in enumerate(picked):
        ref = ep.cameras.get("scene") or next(iter(ep.cameras.values()))
        role = "scene" if "scene" in ep.cameras else next(iter(ep.cameras))
        sel = np.linspace(0, len(win.frame_times) - 1, 8).round().astype(int)
        strip = Image.new("RGB", (cell * 8, cell + 60), "white")
        n_frames = reader.frame_count(ep, role)  # type: ignore[attr-defined]
        for j, s in enumerate(sel):
            idx = min(max(int(round(win.frame_times[s] * ref.fps)), 0), n_frames - 1)
            frame = np.asarray(reader.read(ep, role, idx), dtype=np.uint8)  # type: ignore[attr-defined]
            strip.paste(Image.fromarray(frame).resize((cell, cell)), (j * cell, 0))
        draw = ImageDraw.Draw(strip)
        draw.text(
            (4, cell + 4), f"{ep.episode_id} {win.prompt_kind} anchor={win.anchor_s:.2f}", "black"
        )
        draw.text((4, cell + 20), win.target[:140], "black")
        name = f"{k:02d}_{ep.episode_id}_{win.prompt_kind}_{win.anchor_s:.2f}.png"
        strip.save(out_dir / name)
        files.append(name)
    index = out_dir / SPOT_INDEX
    index.write_text(
        json.dumps(
            {"source": source, "n_windows": len(files), "files": files, "reviewer_pass": None},
            indent=2,
        ),
        encoding="utf-8",
    )
    return index


def record_spot_review(spot_dir: Path, passed: int) -> Path:
    """Store the human reviewer's pass count in ``spot_dir/index.json``."""
    index = spot_dir / SPOT_INDEX
    doc = json.loads(index.read_text(encoding="utf-8"))
    if not 0 <= passed <= doc["n_windows"]:
        raise ValueError(f"pass count {passed} outside 0..{doc['n_windows']}")
    doc["reviewer_pass"] = passed
    index.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return index


def read_spot_index(spot_dir: Path) -> tuple[int, int | None]:
    """Return (n_windows, reviewer_pass) from ``spot_dir/index.json``; (0, None) if absent."""
    index = spot_dir / SPOT_INDEX
    if not index.exists():
        return 0, None
    doc = json.loads(index.read_text(encoding="utf-8"))
    return int(doc["n_windows"]), doc["reviewer_pass"]


def check_spot_check(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Fail the source unless a human passed at least spot_min_pass of the spot windows.

    The windows are written by write_spot_strips; ``ctx.spot_n`` is how many exist and
    ``ctx.spot_reviewer_pass`` the recorded pass count (None while awaiting review).
    """
    th = ctx.thresholds
    if ctx.spot_n == 0:
        return _fail("keep", "no spot-check strips written")
    if ctx.spot_reviewer_pass is None:
        return _fail("keep", f"awaiting human review of {ctx.spot_n} windows")
    need = (
        th.spot_min_pass
        if ctx.spot_n == th.spot_n
        else math.ceil(th.spot_min_pass / th.spot_n * ctx.spot_n)
    )
    detail = f"{ctx.spot_reviewer_pass}/{ctx.spot_n} passed, need {need}"
    if ctx.spot_reviewer_pass >= need:
        return _ok(float(ctx.spot_reviewer_pass), detail)
    return _fail("keep", detail, float(ctx.spot_reviewer_pass))


# --- prompt diff -------------------------------------------------------------------------------


def check_prompt_diff(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Force and control prompts differ only by the force block; the block is under budget."""
    if ctx.render_cfg is None:
        return _fail("keep", "no render config in context")
    wins = [w for w in ctx.windows.get(episode.episode_id, []) if w.force_lines]
    if not wins:
        return _ok(detail="no force windows to diff")
    roles = list(episode.cameras) or ["scene"]
    problems: list[str] = []
    worst = 0
    for w in wins:
        try:
            prompt_diff(w, header=ctx.render_cfg.header, task=episode.task, camera_roles=roles)
            block = "\n".join([ctx.render_cfg.header, *w.force_lines])
            worst = max(worst, budget_check(block, ctx.render_cfg.max_tokens))
        except (AssertionError, TokenBudgetError) as exc:
            problems.append(f"{w.prompt_kind}@{w.anchor_s}: {exc}")
    if problems:
        return _fail("keep", "; ".join(problems[:3]), float(worst))
    return _ok(float(worst), f"{len(wins)} windows, max {worst} tokens")


# --- holdout, overlap, license -----------------------------------------------------------------


def check_holdout(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """No train episode in holdout_ids.json; no pseudo-labeled window in an eval split.

    An episode frozen into a held-out split may sit in an eval split; it is leakage only when
    it is assigned to train (``ctx.split_of``, default train).
    """
    ids = {episode.episode_id, episode_key(episode.source, episode.episode_id)}
    if ids & ctx.holdout_ids and ctx.split_of.get(episode.episode_id, "train") == "train":
        return _fail("drop_episode", "episode id is in the frozen holdout set")
    pseudo = any(lab.origin == "pseudo" for lab in episode.labels)
    if pseudo and any(w.split in EVAL_SPLITS for w in ctx.windows.get(episode.episode_id, [])):
        return _fail("drop_episode", "pseudo-labeled window in an eval split")
    return _ok()


def check_overlap(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """PHash of first, middle, last frame vs every benchmark episode; a match is all three close."""
    if not ctx.benchmark_index:
        return _ok(detail="empty benchmark index")
    if ctx.reader is None:
        return _fail("drop_episode", "no frame reader; cannot rule out overlap")
    try:
        hashes = hashing.cached_hashes(episode, ctx.reader, ctx.hash_cache)
    except Exception as exc:
        return _fail("drop_episode", f"cannot hash frames ({type(exc).__name__}: {exc})")
    others = [
        r
        for r in ctx.benchmark_index
        if (r.source, r.episode_id) != (episode.source, episode.episode_id)
    ]
    row = hashing.matching_row(hashes, others, ctx.thresholds.overlap_max_distance)
    if row is None:
        return _ok()
    dist = max(
        hashing.hamming(a, b)
        for a, b in zip(hashes, (row.h_first, row.h_mid, row.h_last), strict=True)
    )
    return _fail("drop_episode", f"matches benchmark {row.source}/{row.episode_id}", float(dist))


def check_license(episode: Episode, ctx: VerifyContext) -> CheckResult:
    """Require the license string to be in the source's allowed list."""
    if episode.license in ctx.license_allowed:
        return _ok()
    return _fail("drop_episode", f"license {episode.license!r} not in {ctx.license_allowed}")


ALL_CHECKS: list[Check] = [
    check_integrity,
    check_sensor_health,
    check_sync,
    check_normalization,
    check_label_sanity,
    check_spot_check,
    check_prompt_diff,
    check_holdout,
    check_overlap,
    check_license,
]
CHECK_NAMES: list[str] = [c.__name__.removeprefix("check_") for c in ALL_CHECKS]
SOURCE_CHECKS = frozenset({"normalization", "label_sanity", "spot_check", "prompt_diff"})


def run_checks(episode: Episode, ctx: VerifyContext) -> dict[str, CheckResult]:
    """Run every check in ALL_CHECKS order and return results keyed by check name."""
    return {name: fn(episode, ctx) for name, fn in zip(CHECK_NAMES, ALL_CHECKS, strict=True)}
