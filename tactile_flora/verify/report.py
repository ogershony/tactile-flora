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

from tactile_flora.schema import CheckResult, VerifyThresholds

Results = dict[str, dict[str, CheckResult]]
SOURCE_LEVEL = ("normalization", "label_sanity", "spot_check", "prompt_diff")


NORM_GRASP_RANGE = (0.8, 1.2)
NORM_FREE_TOL = 0.1
NORM_P99_MAX = 3.0


def _pooled_normalization_ok(seen: list[CheckResult]) -> bool:
    """Pool the per-episode normalization stats stored as JSON in ``detail``.

    Passes iff the pooled (median over episodes and arms) grasp median lies in [0.8, 1.2], the
    pooled free-space median in [-0.1, 0.1] and the pooled 99th percentile (median of the
    per-episode p99s) is at most 3. Missing stats (e.g. no constants in context) fail.
    """
    pooled: dict[str, list[float]] = {"grasp": [], "free": [], "p99": []}
    for r in seen:
        try:
            doc = json.loads(r.detail)
            for k in pooled:
                pooled[k].extend(float(v) for v in doc[k])
        except (ValueError, KeyError, TypeError):
            return False
    if not pooled["p99"] and not pooled["free"] and not pooled["grasp"]:
        return False
    if pooled["grasp"]:
        g = float(np.median(pooled["grasp"]))
        if not NORM_GRASP_RANGE[0] <= g <= NORM_GRASP_RANGE[1]:
            return False
    if pooled["free"] and abs(float(np.median(pooled["free"]))) > NORM_FREE_TOL:
        return False
    return not (pooled["p99"] and float(np.median(pooled["p99"])) > NORM_P99_MAX)


def source_verdicts(results: Results) -> dict[str, bool]:
    """Return, per source-level check, whether the source passed.

    Most checks pass only if every episode passed. Normalization is pooled over all episodes
    (see _pooled_normalization_ok), so one light grasp does not fail the source.
    """
    out: dict[str, bool] = {}
    for name in SOURCE_LEVEL:
        seen = [r[name] for r in results.values() if name in r]
        if not seen:
            continue
        if name == "normalization":
            out[name] = _pooled_normalization_ok(seen)
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
        out["sync_median"] = float(np.median(lags)) < VerifyThresholds().sync_median_s
    return out


def _check_names(results: Results) -> list[str]:
    names: list[str] = []
    for per in results.values():
        for n in per:
            if n not in names:
                names.append(n)
    return names


def write_report(source: str, results: Results, path: Path | str) -> Path:
    """Write ``report.md`` and ``checks.parquet`` into directory ``path``; return report.md."""
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
    verdicts = source_verdicts(results)
    if verdicts:
        lines += ["## Source verdict", ""]
        lines += [f"- {n}: {'pass' if ok else 'FAIL'}" for n, ok in verdicts.items()]
        lines.append("")
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
