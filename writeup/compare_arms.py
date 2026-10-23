"""Paired comparison of evaluation cells on the v3 split (dev 136 clips, test 115 clips).

Prints accuracy, AUROC, paired wins/losses with an exact McNemar test and an object-level
bootstrap for every cell against the v2 control and against the v3 tactile arm.

    .venv/bin/python writeup/compare_arms.py \\
        --cell "v5a dropout" data/acceptance/hrdexdb-expanded-v5/eval/v5a-dropout \\
        --cell "v5b chart" data/acceptance/hrdexdb-expanded-v5/eval/v5b-chart
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from tactile_flora.eval.metrics import SUCCESS_THRESHOLD as TH

V2_CONTROL = Path("data/acceptance/hrdexdb-expanded-v2/eval/expanded-v2-control")
V3_FORCE = Path("data/acceptance/hrdexdb-expanded-v3/eval/expanded-v3-force")


def load(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    d = pd.read_parquet(path).set_index("window_id")
    d["label"] = (d["truth_target"].str.extract(r"Score:\s*(\d)")[0].astype(int) >= TH).astype(int)
    d["pred"] = (d["expected_score"] >= TH).astype(int)
    d["ok"] = d["pred"] == d["label"]
    return d


def paired(a: pd.DataFrame, b: pd.DataFrame) -> str:
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    w, l = int((a.ok & ~b.ok).sum()), int((~a.ok & b.ok).sum())
    p = binomtest(w, w + l).pvalue if w + l else 1.0
    objs = a["object"].to_numpy()
    uo = np.unique(objs)
    rng = np.random.default_rng(0)
    t = a["label"].to_numpy()
    sa, sb = a["expected_score"].to_numpy(), b["expected_score"].to_numpy()
    d_auc, d_acc = [], []
    for _ in range(3000):
        pick = rng.choice(uo, len(uo))
        i = np.concatenate([np.flatnonzero(objs == o) for o in pick])
        d_acc.append(a.ok.to_numpy()[i].mean() - b.ok.to_numpy()[i].mean())
        if t[i].min() != t[i].max():
            d_auc.append(roc_auc_score(t[i], sa[i]) - roc_auc_score(t[i], sb[i]))
    lo, hi = np.percentile(d_auc, [2.5, 97.5])
    alo, ahi = np.percentile(d_acc, [2.5, 97.5])
    return (
        f"wins {w} losses {l} McNemar p={p:.3f}; acc diff {a.ok.mean() - b.ok.mean():+.3f} "
        f"CI [{alo:+.3f},{ahi:+.3f}]; AUROC diff "
        f"{roc_auc_score(t, sa) - roc_auc_score(t, sb):+.3f} CI [{lo:+.3f},{hi:+.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--modes", default="real,video_blank,chart_text,chart,shuffled")
    args = parser.parse_args()
    for split in ("val", "test"):
        bench = f"hrdexdb_aligned_{split}"
        control = load(V2_CONTROL / bench / "omitted/predictions.parquet")
        v3 = load(V3_FORCE / bench / "real/predictions.parquet")
        assert control is not None and v3 is not None
        print(f"\n=== {split} ({len(control)} clips)")
        print(f"v2 control      acc {control.ok.mean():.3f} auroc {roc_auc_score(control.label, control.expected_score):.3f}")
        print(f"v3 tactile real acc {v3.ok.mean():.3f} auroc {roc_auc_score(v3.label, v3.expected_score):.3f}   vs control: {paired(v3, control)}")
        for label, directory in args.cell:
            for mode in args.modes.split(","):
                d = load(Path(directory) / bench / mode / "predictions.parquet")
                if d is None:
                    continue
                print(
                    f"{label} {mode:11s} acc {d.ok.mean():.3f} auroc {roc_auc_score(d.label, d.expected_score):.3f} "
                    f"pred-success {d.pred.mean():.2f}\n    vs control: {paired(d, control)}\n    vs v3 real: {paired(d, v3)}"
                )


if __name__ == "__main__":
    main()
