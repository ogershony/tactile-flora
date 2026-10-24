"""Per-source markdown verification reports and the checks.parquet table.

Owns: write_report (report.md + checks.parquet), read_checks (the inverse of the parquet) and
source_verdicts (which source-level checks failed).
Must never: compute checks or write manifests.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.force import normalize
from tactile_flora.schema import CheckResult, VerifyThresholds

Results = dict[str, dict[str, CheckResult]]
SOURCE_LEVEL = ("normalization", "label_sanity", "spot_check", "prompt_diff")


NORM_GRASP_RANGE = (0.8, 1.2)
NORM_FREE_TOL = 0.1
NORM_P99_MAX = 3.0


def pooled_normalization_stats(seen: list[CheckResult]) -> dict[str, float] | None:
    """Median over episodes and arms of the grasp, free-space and p99 stats in ``detail``.

    Episodes marked ``"excluded": true`` are skipped; None when a detail is unreadable or no
    stat is left. Keys present only when the pool has that stat.
    """
    pooled: dict[str, list[float]] = {"grasp": [], "free": [], "p99": []}
    for r in seen:
        try:
            doc = json.loads(r.detail)
            if doc.get("excluded", False) is True:
                continue
            for k in pooled:
                pooled[k].extend(float(v) for v in doc[k])
        except (ValueError, KeyError, TypeError, AttributeError):
            return None
    if not pooled["p99"] and not pooled["free"] and not pooled["grasp"]:
        return None
    return {k: float(np.median(v)) for k, v in pooled.items() if v}


def _pooled_normalization_ok(
    seen: list[CheckResult],
    p99_max: float = NORM_P99_MAX,
    grasp_range: tuple[float, float] = NORM_GRASP_RANGE,
) -> bool:
    """Pool the per-episode normalization stats stored as JSON in ``detail``.

    Passes iff the pooled (median over episodes and arms) grasp median lies in ``grasp_range``
    (the spec's [0.8, 1.2], or the source config's ``norm_grasp_range``), the pooled free-space
    median in [-0.1, 0.1] and the pooled 99th percentile (median of the per-episode p99s) is at
    most ``p99_max`` (the spec's 3, or the source config's ``norm_p99_max``). Episodes whose
    detail says ``"excluded": true`` (left out of the fit by the adapter, e.g. PHANTOM failure
    demos, which are off-scale by design) do not enter the pool. Missing stats (e.g. no
    constants in context) fail.
    """
    stats = pooled_normalization_stats(seen)
    if stats is None:
        return False
    if "grasp" in stats and not grasp_range[0] <= stats["grasp"] <= grasp_range[1]:
        return False
    if "free" in stats and abs(stats["free"]) > NORM_FREE_TOL:
        return False
    return not ("p99" in stats and stats["p99"] > p99_max)


def source_verdicts(
    results: Results,
    sync_median_s: float | None = None,
    norm_p99_max: float | None = None,
    norm_grasp_range: tuple[float, float] | None = None,
) -> dict[str, bool]:
    """Return, per source-level check, whether the source passed.

    Most checks pass only if every episode passed. Normalization is pooled over all episodes
    (see _pooled_normalization_ok), so one light grasp does not fail the source. The
    ``sync_median`` gate compares the median |lag| with ``sync_median_s``, the pooled p99 with
    ``norm_p99_max`` and the pooled grasp median with ``norm_grasp_range`` (the source config's
    values; the spec defaults when None).
    """
    out: dict[str, bool] = {}
    for name in SOURCE_LEVEL:
        seen = [r[name] for r in results.values() if name in r]
        if not seen:
            continue
        if name == "normalization":
            out[name] = _pooled_normalization_ok(
                seen,
                NORM_P99_MAX if norm_p99_max is None else norm_p99_max,
                NORM_GRASP_RANGE if norm_grasp_range is None else norm_grasp_range,
            )
        else:
            out[name] = all(c.passed for c in seen)
    # Derive this gate from the persisted checks too, so manifest construction cannot
    # bypass the source median gate enforced by the verify command.
    lags = [
        abs(r["sync"].value)
        for r in results.values()
        if "sync" in r and r["sync"].value is not None
    ]
    if lags:
        gate = VerifyThresholds().sync_median_s if sync_median_s is None else sync_median_s
        out["sync_median"] = float(np.median(lags)) < gate
    return out


def _check_names(results: Results) -> list[str]:
    names: list[str] = []
    for per in results.values():
        for n in per:
            if n not in names:
                names.append(n)
    return names


def write_report(
    source: str,
    results: Results,
    path: Path | str,
    sync_median_s: float | None = None,
    norm_p99_max: float | None = None,
    norm: normalize.NormConstants | None = None,
    norm_grasp_range: tuple[float, float] | None = None,
) -> Path:
    """Write ``report.md`` and ``checks.parquet`` into directory ``path``; return report.md.

    With ``norm`` the report also lists the fitted constants per group and arm (baseline,
    grasp, tau in raw units, the grasp rule and the fit-time ``p99_fprime`` tail).
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    names = _check_names(results)
    episodes = sorted(results)
    rows = [
        (ep, name, results[ep][name]) for ep in episodes for name in names if name in results[ep]
    ]
    table = pa.table(
        {
            "episode_id": pa.array([r[0] for r in rows], pa.string()),
            "check": pa.array([r[1] for r in rows], pa.string()),
            "passed": pa.array([r[2].passed for r in rows], pa.bool_()),
            "value": pa.array([r[2].value for r in rows], pa.float64()),
            "action": pa.array([r[2].action for r in rows], pa.string()),
            "detail": pa.array([r[2].detail for r in rows], pa.string()),
        }
    )
    pq.write_table(table, out / "checks.parquet")

    lines = [f"# Verification report: {source}", "", f"Episodes: {len(episodes)}", ""]
    verdicts = source_verdicts(results, sync_median_s, norm_p99_max, norm_grasp_range)
    if verdicts:
        lines += ["## Source verdict", ""]
        lines += [f"- {n}: {'pass' if ok else 'FAIL'}" for n, ok in verdicts.items()]
        lines.append("")
    if norm is not None:
        lines += norm_lines(norm, norm_p99_max, norm_grasp_range) + [""]
    lines += [
        "## Summary",
        "",
        "| check | pass | fail | actions on fail | actions on pass |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        res = [results[e][name] for e in episodes if name in results[e]]
        npass = sum(r.passed for r in res)
        on_fail = Counter(r.action for r in res if not r.passed)
        on_pass = Counter(r.action for r in res if r.passed and r.action != "keep")
        fmt = lambda c: ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "-"  # noqa: E731
        lines.append(f"| {name} | {npass} | {len(res) - npass} | {fmt(on_fail)} | {fmt(on_pass)} |")
    lines += ["", "## Failures by episode", ""]
    any_fail = False
    for ep in episodes:
        bad = [(n, results[ep][n]) for n in names if n in results[ep] and not results[ep][n].passed]
        if bad:
            any_fail = True
            lines.append(f"### {ep}")
            lines += [f"- {n} ({r.action}): {r.detail}" for n, r in bad]
            lines.append("")
    if not any_fail:
        lines += ["None.", ""]
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def norm_lines(
    norm: normalize.NormConstants,
    norm_p99_max: float | None = None,
    norm_grasp_range: tuple[float, float] | None = None,
) -> list[str]:
    """Markdown table of the fitted constants per group and arm (raw units; F' tail)."""
    gate = NORM_P99_MAX if norm_p99_max is None else norm_p99_max
    band = NORM_GRASP_RANGE if norm_grasp_range is None else norm_grasp_range
    lines = [
        f"## Normalization constants ({norm.channel}; pooled p99 of F' gate {gate:g}; "
        f"pooled grasp band [{band[0]:g}, {band[1]:g}])",
        "",
        "| group | arm | baseline | grasp | sigma | tau | grasp rule | p99 of F' (fit) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group, arms in sorted(norm.groups.items()):
        for arm, n in sorted(arms.items()):
            tail = "-" if n.p99_fprime is None else f"{n.p99_fprime:.2f}"
            lines.append(
                f"| {group} | {arm} | {n.baseline:.4g} | {n.grasp:.4g} | {n.sigma:.3g} | "
                f"{n.tau:.4g} | {n.grasp_rule} (g {n.grasp_g:.3g}) | {tail} |"
            )
    return lines


def read_checks(path: Path | str) -> Results:
    """Read ``checks.parquet`` (a file, or the directory holding it) back into results."""
    p = Path(path)
    if p.is_dir():
        p = p / "checks.parquet"
    results: Results = {}
    for r in pq.read_table(p).to_pylist():
        results.setdefault(r["episode_id"], {})[r["check"]] = CheckResult(
            passed=r["passed"], value=r["value"], action=r["action"], detail=r["detail"]
        )
    return results
