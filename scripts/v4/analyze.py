"""v4 results: metrics with cluster-bootstrap intervals and the pre-registered comparisons.

Reads every ``<eval_dir>/<checkpoint>/<benchmark>/<mode>/predictions.parquet``. Failure is the
positive class (truth score <= 2); the risk score is -E[s]. Clusters for resampling: HRDexDB
object (configs/v4/hrdexdb_split.json), REASSEMBLE recording (episode id before ``_seg``),
PHANTOM episode, RoboReward video. Metrics: AUROC (+ DeLong SE), AUPRC (failure), balanced
accuracy / failure recall / success recall at the hard score (score <= 2 = failure, no tuning),
failure recall at 80% and 90% success recall, MAE of the hard score and of E[s] (1-5 scale).
Paired comparisons on the clips both cells scored: delta AUROC (10,000 cluster bootstrap
resamples, percentile CI) with a DeLong p-value, delta balanced accuracy and failure recall at
the hard score, exact McNemar. Holm across the three outcome TEST sets (HRDexDB v4, REASSEMBLE,
PHANTOM) for the primary comparison (force adapter with real force vs released FLORA without
force); DEV and the coverage subsets are outside the family. HRDexDB also gets balanced accuracy,
failure and success recall at the E[s] threshold that maximizes balanced accuracy on DEV (the
plan's only tuned quantity). Baseline cells: hand prior (added here), the force-only classifier
(scripts/v4/classifier_cells.py writes force_classifier/<bench>/none) and two parameter-free late
fusions of a force-free video model with that classifier (fusion_released, fusion_v4_control: mean
normalized risk rank; threshold-free metrics only).

Usage: python scripts/v4/analyze.py data/eval [data/hrdexdb_v4_audit.json] > results.md
(``tf eval grid`` writes cells under data/eval whatever the eval config's output_dir says;
results.json is written beside the cells)
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT = json.loads(Path("configs/v4/hrdexdb_split.json").read_text())["episodes"]
B = int(__import__("os").environ.get("V4_BOOT", "10000"))
HOLM_FAMILY = ("hrdexdb_v4_test", "reassemble_seg_test", "phantom_fail_test")


def cluster_of(bench: str, ep: str) -> str:
    if bench.startswith("hrdexdb"):
        return SPLIT[ep]["object_id"]
    if bench.startswith("reassemble"):
        return ep.split("_seg")[0]
    return ep


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
    """Paired DeLong: (delta AUROC, two-sided p)."""
    from scipy.stats import norm

    pos, neg = s1[y == 1], s1[y == 0]
    pos2, neg2 = s2[y == 1], s2[y == 0]

    def comps(p, n):
        v10 = np.array([(np.sum(x > n) + 0.5 * np.sum(x == n)) / len(n) for x in p])
        v01 = np.array([(np.sum(p > x) + 0.5 * np.sum(p == x)) / len(p) for x in n])
        return v10, v01

    a10, a01 = comps(pos, neg)
    b10, b01 = comps(pos2, neg2)
    d = a10.mean() - b10.mean()
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(pos) + (
        s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]
    ) / len(neg)
    if var <= 0:
        return float(d), float("nan")
    return float(d), float(2 * norm.sf(abs(d) / math.sqrt(var)))


def fr_at_sr(y: np.ndarray, risk: np.ndarray, sr: float) -> float:
    succ = np.sort(risk[y == 0])
    if len(succ) == 0 or y.sum() == 0:
        return float("nan")
    t = succ[min(len(succ) - 1, int(math.ceil(sr * len(succ))) - 1)]  # flag risk > t
    return float(np.mean(risk[y == 1] > t))


def auprc(y: np.ndarray, risk: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, risk)) if 0 < y.sum() < len(y) else float("nan")


def metrics(df: pd.DataFrame) -> dict:
    y = df.fail.to_numpy()
    risk = -df.es.to_numpy()
    hard = df.score.to_numpy() <= 2
    no_hard = bool(df.score.isna().all())  # late fusion: a rank score, no 1-5 answer
    out = {
        "n": len(df),
        "failures": int(y.sum()),
        "mae_hard": float("nan") if no_hard else float(np.mean(np.abs(df.score - df.truth))),
        "mae_es": float("nan") if no_hard else float(np.mean(np.abs(df.es - df.truth))),
    }
    if 0 < y.sum() < len(y):
        fr = float("nan") if no_hard else float(hard[y == 1].mean())
        sr = float("nan") if no_hard else float((~hard[y == 0]).mean())
        out.update(
            auroc=auroc(y, risk),
            auprc=auprc(y, risk),
            ba=0.5 * (fr + sr),
            fr=fr,
            sr=sr,
            fr_at_sr80=fr_at_sr(y, risk, 0.8),
            fr_at_sr90=fr_at_sr(y, risk, 0.9),
        )
    return out


def boot_ci(df: pd.DataFrame, fn, rng: np.random.Generator) -> tuple[float, float]:
    groups = df.groupby("cluster").indices
    keys = list(groups)
    vals = []
    for _ in range(B // 5):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        v = fn(df.iloc[idx])
        if not math.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load(eval_dir: Path) -> dict[tuple[str, str, str], pd.DataFrame]:
    cells = {}
    for f in sorted(eval_dir.glob("*/*/*/predictions.parquet")):
        ck, bench, mode = f.parts[-4], f.parts[-3], f.parts[-2]
        p = pd.read_parquet(f)
        truth = p.truth_target.str.extract(r"Score:\s*(\d)")[0].astype(float)
        df = pd.DataFrame(
            {
                "window_id": p.window_id,
                "episode_id": p.episode_id,
                "es": p.expected_score.astype(float),
                "score": p.score.astype(float),
                "truth": truth,
            }
        )
        df["fail"] = (df.truth <= 2).astype(int)
        df["cluster"] = [cluster_of(bench, e) for e in df.episode_id]
        cells[(ck, bench, mode)] = df.dropna(subset=["es", "score", "truth"]).reset_index(drop=True)
    return cells


def paired(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator) -> dict:
    m = a.merge(b, on=["window_id", "cluster", "fail"], suffixes=("_a", "_b"))
    y = m.fail.to_numpy()
    ra, rb = -m.es_a.to_numpy(), -m.es_b.to_numpy()
    ha, hb = m.score_a.to_numpy() <= 2, m.score_b.to_numpy() <= 2
    d, p = delong(y, ra, rb)
    groups = m.groupby("cluster").indices
    keys = list(groups)
    ds, dba, dfr = [], [], []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ds.append(auroc(yy, ra[idx]) - auroc(yy, rb[idx]))
        ba_a = 0.5 * (ha[idx][yy == 1].mean() + (~ha[idx][yy == 0]).mean())
        ba_b = 0.5 * (hb[idx][yy == 1].mean() + (~hb[idx][yy == 0]).mean())
        dba.append(ba_a - ba_b)
        dfr.append(ha[idx][yy == 1].mean() - hb[idx][yy == 1].mean())
    def ba(h: np.ndarray) -> float:
        return 0.5 * (h[y == 1].mean() + (~h[y == 0]).mean())

    correct_a, correct_b = ha == (y == 1), hb == (y == 1)
    nb, nc = int(np.sum(correct_a & ~correct_b)), int(np.sum(~correct_a & correct_b))
    from scipy.stats import binomtest

    pm = float(binomtest(nb, nb + nc, 0.5).pvalue) if nb + nc else 1.0
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {
        "n": len(m),
        "d_auroc": d,
        "d_auroc_ci": ci(ds),
        "delong_p": p,
        "d_ba": float(ba(ha) - ba(hb)),
        "d_ba_ci": ci(dba),
        "d_fr": float(ha[y == 1].mean() - hb[y == 1].mean()),
        "d_fr_ci": ci(dfr),
        "mcnemar": [nb, nc],
        "mcnemar_p": pm,
    }
    if m.score_a.isna().all() or m.score_b.isna().all():  # a late-fusion cell has no hard score
        nan = float("nan")
        out.update(d_ba=nan, d_ba_ci=[nan, nan], d_fr=nan, d_fr_ci=[nan, nan], mcnemar=[0, 0], mcnemar_p=nan)
    return out


def hand_prior_cell(cells: dict) -> None:
    """Add the hand-prior baseline (risk = the hand's TRAIN failure rate) to every HRDexDB bench."""
    train = [i for i in SPLIT.values() if i["split"] == "train"]
    rate = {h: np.mean([not i["success"] for i in train if i["hand"] == h]) for h in
            ("inspire_dftp", "inspire_f1")}
    for (ck, bench, mode), df in list(cells.items()):
        if bench.startswith("hrdexdb") and ck == "released" and mode == "omitted":
            prior = df.copy()
            risk = np.array([rate[SPLIT[e]["hand"]] for e in prior.episode_id])
            prior["es"] = -risk
            prior["score"] = np.where(risk > 0.5, 1.0, 5.0)
            cells[("hand_prior", bench, "none")] = prior


def fusion_cells(cells: dict) -> None:
    """Late-fusion baselines: the mean of the normalized risk ranks of a video model without force
    text (released FLORA or the control adapter, omitted) and the force-only classifier, on the
    clips both scored. No fitted parameter, so nothing is tuned; threshold-free metrics only (no
    hard score). Added 2026-10-23 17:23 UTC, before any TEST cell or force-adapter DEV cell was read.
    """
    from scipy.stats import rankdata

    for (ck, bench, mode), clf in list(cells.items()):
        if ck != "force_classifier":
            continue
        for vck in ("released", "v4_control"):
            vid = cells.get((vck, bench, "omitted"))
            if vid is None:
                continue
            m = vid.merge(clf[["window_id", "es"]], on="window_id", suffixes=("", "_c"))
            risk = (rankdata(-m.es.to_numpy()) + rankdata(-m.es_c.to_numpy())) / (2 * len(m))
            f = m.drop(columns=["es_c"])
            f["es"] = -risk
            f["score"] = np.nan
            cells[(f"fusion_{vck}", bench, "none")] = f.reset_index(drop=True)


def coverage_subset(cells: dict, audit: Path, max_gap_s: float = 0.5) -> None:
    """Add every HRDexDB cell restricted to trials whose video covers the timeline within 0.5 s."""
    if not audit.is_file():
        return
    trials = json.loads(audit.read_text()).get("trials", {})
    ok = {e for e, t in trials.items() if t["frame_ids_beyond_video"] / 30.0 < max_gap_s}
    for (ck, bench, mode), df in list(cells.items()):
        if bench.startswith("hrdexdb"):
            cells[(ck, bench + "_covered", mode)] = df[df.episode_id.isin(ok)].reset_index(drop=True)


def dev_threshold(dev: pd.DataFrame) -> float:
    """E[s] cut (failure if E[s] <= cut) maximizing balanced accuracy on DEV; ties: lowest cut."""
    y = dev.fail.to_numpy() == 1
    es = dev.es.to_numpy()
    best, cut = -1.0, float("nan")
    for c in np.unique(es):
        h = es <= c
        v = 0.5 * (h[y].mean() + (~h[~y]).mean())
        if v > best + 1e-12:
            best, cut = v, float(c)
    return cut


def at_cut(df: pd.DataFrame, cut: float) -> dict:
    y = df.fail.to_numpy() == 1
    h = df.es.to_numpy() <= cut
    fr, sr = float(h[y].mean()), float((~h[~y]).mean())
    return {"cut": cut, "ba": 0.5 * (fr + sr), "fr": fr, "sr": sr}


def main() -> None:
    """Write results.json beside the eval cells and print the markdown tables."""
    eval_dir = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    cells = load(eval_dir)
    hand_prior_cell(cells)
    fusion_cells(cells)
    coverage_subset(cells, Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/hrdexdb_v4_audit.json"))
    res: dict = {"cells": {}, "paired": {}, "dev_threshold": {}}
    by_bench = defaultdict(dict)
    for (ck, bench, mode), df in cells.items():
        m = metrics(df)
        if "auroc" in m:
            m["auroc_ci"] = boot_ci(df, lambda d: auroc(d.fail.to_numpy(), -d.es.to_numpy()), rng)
            if bench.startswith("hrdexdb"):
                for hand in ("inspire_dftp", "inspire_f1"):
                    sub = df[[SPLIT[e]["hand"] == hand for e in df.episode_id]]
                    if 0 < sub.fail.sum() < len(sub):
                        m[f"auroc_{hand}"] = auroc(sub.fail.to_numpy(), -sub.es.to_numpy())
        res["cells"][f"{ck}|{bench}|{mode}"] = m
        by_bench[bench][(ck, mode)] = df
    for (ck, bench, mode), df in cells.items():  # HRDexDB TEST at the DEV-chosen E[s] cut
        if not bench.startswith("hrdexdb_v4_test") or df.fail.nunique() < 2:
            continue
        dev = cells.get((ck, "hrdexdb_v4_dev", mode))
        if dev is not None and dev.fail.nunique() == 2:
            res["dev_threshold"][f"{ck}|{bench}|{mode}"] = at_cut(df, dev_threshold(dev))
    comps = [
        ("v4_force", "real", "released", "omitted", "primary"),
        ("v4_force", "real", "v4_control", "omitted", "force vs control"),
        ("v4_control", "omitted", "released", "omitted", "fine-tuning alone"),
        ("v4_force", "real", "v4_force", "shuffled", "real vs shuffled force"),
        ("v4_force", "real", "v4_force", "omitted", "real vs no force"),
        ("v4_force", "real", "released", "real", "vs released + force text"),
        ("v4_force", "real", "force_classifier", "none", "vs force classifier alone"),
        ("v4_force", "real", "fusion_released", "none", "vs late fusion, released + classifier"),
        ("v4_force", "real", "fusion_v4_control", "none", "vs late fusion, control + classifier"),
        ("released", "real", "released", "omitted", "force text, no training"),
        # second seed (supplementary; outside the pre-registered Holm family)
        ("v4_force_s1", "real", "released", "omitted", "primary, seed 1"),
        ("v4_force_s1", "real", "v4_control_s1", "omitted", "force vs control, seed 1"),
        ("v4_control_s1", "omitted", "released", "omitted", "fine-tuning alone, seed 1"),
        ("v4_force_s1", "real", "v4_force_s1", "shuffled", "real vs shuffled force, seed 1"),
        ("v4_force_s1", "real", "v4_control", "omitted", "force seed 1 vs control seed 0"),
        # exploratory (added after the DEV cells): touch alone, video replaced by mid-gray frames
        ("v4_force", "real", "v4_force", "video_blank", "real vs blank video (exploratory)"),
        ("v4_force", "video_blank", "force_classifier", "none", "blank video vs force classifier (exploratory)"),
    ]
    primary_p = {}
    for bench, d in by_bench.items():
        for ca, ma, cb, mb, name in comps:
            if (ca, ma) in d and (cb, mb) in d and d[(ca, ma)].fail.nunique() == 2:
                r = paired(d[(ca, ma)], d[(cb, mb)], rng)
                res["paired"][f"{bench}|{name}"] = r
                if name == "primary" and bench in HOLM_FAMILY:
                    primary_p[bench] = r["delong_p"]
    # Holm over the outcome datasets' primary comparison
    order = sorted(primary_p, key=lambda k: primary_p[k])
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * primary_p[k]))
        adj[k] = running
    res["holm_primary"] = adj
    Path(eval_dir / "results.json").write_text(json.dumps(res, indent=1))
    print("| cell | n | fail | AUROC [95% CI] | AUPRC | BA | FR | SR | FR@SR80 | FR@SR90 | MAE |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, m in sorted(res["cells"].items()):
        if "auroc" in m:
            print(
                f"| {k} | {m['n']} | {m['failures']} | {m['auroc']:.3f} [{m['auroc_ci'][0]:.3f}, {m['auroc_ci'][1]:.3f}] | "
                f"{m['auprc']:.3f} | {m['ba']:.3f} | {m['fr']:.3f} | {m['sr']:.3f} | {m['fr_at_sr80']:.3f} | {m['fr_at_sr90']:.3f} | {m['mae_hard']:.3f} |"
            )
        else:
            print(
                f"| {k} | {m['n']} | {m['failures']} | | | | | | | | {m['mae_hard']:.3f} (E[s] {m['mae_es']:.3f}) |"
            )
    print()
    for k, r in res["paired"].items():
        print(
            f"- {k}: dAUROC {r['d_auroc']:+.3f} [{r['d_auroc_ci'][0]:+.3f}, {r['d_auroc_ci'][1]:+.3f}] DeLong p={r['delong_p']:.2g}; "
            f"dBA {r['d_ba']:+.3f} [{r['d_ba_ci'][0]:+.3f}, {r['d_ba_ci'][1]:+.3f}]; dFR {r['d_fr']:+.3f} [{r['d_fr_ci'][0]:+.3f}, {r['d_fr_ci'][1]:+.3f}]; "
            f"McNemar {r['mcnemar'][0]} vs {r['mcnemar'][1]} p={r['mcnemar_p']:.2g} (n={r['n']})"
        )
    print("\nHRDexDB TEST at the DEV-chosen E[s] cut (failure if E[s] <= cut):")
    for k, v in sorted(res["dev_threshold"].items()):
        print(f"- {k}: cut {v['cut']:.3f} BA {v['ba']:.3f} FR {v['fr']:.3f} SR {v['sr']:.3f}")
    print("\nHolm-adjusted primary p:", {k: f"{v:.2g}" for k, v in adj.items()})


if __name__ == "__main__":
    main()
