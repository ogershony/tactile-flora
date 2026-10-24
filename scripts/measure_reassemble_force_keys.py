"""Measure which REASSEMBLE-port force column the adapter should read (decision record).

Compares ``observation.force`` (raw sensor) with ``observation.force.base`` (gravity-compensated)
over every recording of the LeRobot port (``robot-lev/reassemble``): (a) the mean force on idle
"No action." frames (free space), (b) |F| during failed vs successful non-idle segments and the
AUROC of the per-segment p50 / p99 as a failed-vs-successful separator, (c) the normalization
scale each key would give (grasp-frame median |F| minus free-space median |F|), and the idle
offsets of the two torque columns.

Usage: ``uv run python scripts/measure_reassemble_force_keys.py data/raw/reassemble_lerobot``.
The 2026-10-22 result over 149 recordings is recorded in docs/deviations.md, section 4.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

FORCE = {"raw": "observation.force", "base": "observation.force.base"}
TORQUE = {"raw": "observation.torque", "base": "observation.torque.base"}
GRIPPER = "observation.state.gripper_position"
COLS = [
    "episode_index",
    "frame_index",
    "segment.index",
    "segment.success",
    "task_index",
    GRIPPER,
    *FORCE.values(),
    *TORQUE.values(),
]
IDLE_TEXT = "No action."
GRASP_G = 0.6  # NormalizeConfig.grasp_g default: closure above which a frame is a grasp frame
GRIPPER_MAX = 0.04016  # meta/stats.json max of the finger width


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Return the rank-based AUROC of ``pos`` over ``neg`` (Mann-Whitney U, ties averaged)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    ranks[order] = np.arange(1, allv.size + 1)
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of the runs of equal values."""
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [values.size]])
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def mag(x: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norm."""
    return np.linalg.norm(x, axis=1)


def _stack(df: Any, col: str) -> np.ndarray:
    return np.stack(df[col].to_numpy()).astype(np.float64)


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-recording and per-non-idle-segment statistics."""
    splits = json.loads((root / "meta" / "splits.json").read_text())
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    idle_idx = int(tasks.loc[IDLE_TEXT, "task_index"])
    recs: list[dict[str, Any]] = []
    segs: list[dict[str, Any]] = []
    for f in sorted((root / "data").rglob("*.parquet")):
        df = pq.read_table(f, columns=COLS).to_pandas()
        for eid, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            force = {k: _stack(g, c) for k, c in FORCE.items()}
            torque = {k: _stack(g, c) for k, c in TORQUE.items()}
            task = g["task_index"].to_numpy()
            seg_index = g["segment.index"].to_numpy()
            ok = g["segment.success"].to_numpy()
            idle = task == idle_idx
            closure = 1.0 - _stack(g, GRIPPER).mean(axis=1) / GRIPPER_MAX
            grasp = (closure > GRASP_G) & ~idle
            rec = str(splits[str(int(eid))]["recording"])
            row: dict[str, Any] = {"eid": int(eid), "rec": rec, "date": rec[:10]}
            for k in FORCE:
                m = force[k][idle].mean(axis=0)
                row[f"{k}_mean"] = m
                row[f"{k}_absmean"] = float(np.linalg.norm(m))
                free = float(np.median(mag(force[k][idle])))
                row[f"{k}_scale"] = (
                    float(np.median(mag(force[k][grasp]))) - free if grasp.any() else float("nan")
                )
                row[f"{k}T_absmean"] = float(np.linalg.norm(torque[k][idle].mean(axis=0)))
            recs.append(row)
            for a, b in runs(seg_index):
                if task[a] == idle_idx:
                    continue
                s: dict[str, Any] = {"date": rec[:10], "failed": not bool(ok[a:b].all())}
                for k in FORCE:
                    m = mag(force[k][a:b])
                    s[f"{k}_p50"] = float(np.percentile(m, 50))
                    s[f"{k}_p99"] = float(np.percentile(m, 99))
                    s[f"{k}_frames"] = m
                segs.append(s)
    return recs, segs


def report(recs: list[dict[str, Any]], segs: list[dict[str, Any]]) -> None:
    """Print the markdown tables."""
    dates = sorted({r["date"] for r in recs})
    print(f"{len(recs)} recordings, {len(segs)} non-idle segments; per date:")
    print({d: sum(r["date"] == d for r in recs) for d in dates})

    print("\n(a) free-space offset: |mean force| over idle frames per recording, N")
    print("| key | stat | " + " | ".join(dates) + " | all |")
    print("|---|---|" + "---|" * (len(dates) + 1))
    for k, col in FORCE.items():
        for stat, fn in (("median", np.median), ("max", np.max)):
            cells = []
            for d in [*dates, None]:
                v = np.array([r[f"{k}_absmean"] for r in recs if d is None or r["date"] == d])
                cells.append(f"{fn(v):.2f}")
            print(f"| {col} | {stat} | " + " | ".join(cells) + " |")
        v = np.array([r[f"{k}_absmean"] for r in recs])
        mean_vec = np.median(np.array([r[f"{k}_mean"] for r in recs]), axis=0)
        print(
            f"{col}: median idle mean vector {np.round(mean_vec, 2).tolist()} N, "
            f"IQR of |mean| {np.percentile(v, [25, 75]).round(2).tolist()}"
        )
    for k, col in TORQUE.items():
        v = np.array([r[f"{k}T_absmean"] for r in recs])
        print(
            f"{col}: idle |mean| median {np.median(v):.3f} N m, max {v.max():.2f}, "
            f"recordings above 1 N m: {int((v > 1).sum())}"
        )
    print("\nforce.base recordings with idle |mean| > 1 N:")
    for r in recs:
        if r["base_absmean"] > 1:
            print(f"  episode {r['eid']} {r['rec']}: {r['base_absmean']:.2f} N")

    print("\n(a) per-recording sample, 5 per date")
    print("| eid | recording | raw mean xyz | raw abs | base mean xyz | base abs | rawT | baseT |")
    print("|---|---|---|---|---|---|---|---|")
    per_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["eid"]):
        if len(per_date[r["date"]]) < 5:
            per_date[r["date"]].append(r)
    for d in dates:
        for r in per_date[d]:
            print(
                f"| {r['eid']} | {r['rec']} | {np.round(r['raw_mean'], 2).tolist()} | "
                f"{r['raw_absmean']:.2f} | {np.round(r['base_mean'], 2).tolist()} | "
                f"{r['base_absmean']:.2f} | {r['rawT_absmean']:.3f} | {r['baseT_absmean']:.3f} |"
            )

    nf = sum(s["failed"] for s in segs)
    print(f"\n(b) |F| in non-idle segments ({nf} failed, {len(segs) - nf} successful), N")
    print("| key | class | frames | pooled p50 | pooled p99 | seg p50 median | seg p99 median |")
    print("|---|---|---|---|---|---|---|")
    for k, col in FORCE.items():
        for cls, flag in (("failed", True), ("successful", False)):
            sel = [s for s in segs if s["failed"] == flag]
            pooled = np.concatenate([s[f"{k}_frames"] for s in sel])
            print(
                f"| {col} | {cls} | {pooled.size} | {np.percentile(pooled, 50):.2f} | "
                f"{np.percentile(pooled, 99):.2f} | "
                f"{np.median([s[f'{k}_p50'] for s in sel]):.2f} | "
                f"{np.median([s[f'{k}_p99'] for s in sel]):.2f} |"
            )
    print("\n(b) AUROC of the per-segment value, failed over successful")
    print("| key | stat | all | " + " | ".join(dates) + " |")
    print("|---|---|---|" + "---|" * len(dates))
    for k, col in FORCE.items():
        for stat in ("p50", "p99"):
            cells = []
            for d in [None, *dates]:
                pos = np.array(
                    [s[f"{k}_{stat}"] for s in segs if s["failed"] and (d in (None, s["date"]))]
                )
                neg = np.array(
                    [s[f"{k}_{stat}"] for s in segs if not s["failed"] and d in (None, s["date"])]
                )
                cells.append(f"{auroc(pos, neg):.3f}")
            print(f"| {col} | {stat} | " + " | ".join(cells) + " |")

    print("\n(c) normalization scale per recording: grasp-frame median |F| - free-space median, N")
    print("| key | median | p10 | p90 | recordings with scale <= 0 |")
    print("|---|---|---|---|---|")
    for k, col in FORCE.items():
        v = np.array([r[f"{k}_scale"] for r in recs if np.isfinite(r[f"{k}_scale"])])
        print(
            f"| {col} | {np.median(v):.2f} | {np.percentile(v, 10):.2f} | "
            f"{np.percentile(v, 90):.2f} | {int((v <= 0).sum())} of {v.size} |"
        )


def main(argv: list[str]) -> None:
    """Entry point: ``<root>`` is the directory holding the port's meta/ and data/."""
    root = Path(argv[1]) if len(argv) > 1 else Path("data/raw/reassemble_lerobot")
    recs, segs = collect(root)
    report(recs, segs)


if __name__ == "__main__":
    main(sys.argv)
