r"""Paired comparison of evaluation cells on the v3 split (dev 136 clips, test 115 clips).

Prints accuracy, AUROC, paired wins/losses with an exact McNemar test and an object-level
bootstrap for every cell against the v2 control and against the v3 tactile arm.

    .venv/bin/python writeup/hrdexdb/compare_arms.py \\
        --cell "v5a dropout" data/acceptance/hrdexdb-expanded-v5/eval/v5a-dropout \\
        --cell "v5b chart" data/acceptance/hrdexdb-expanded-v5/eval/v5b-chart
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from tactile_flora.eval.metrics import SUCCESS_THRESHOLD as TH

V2_CONTROL = Path("data/acceptance/hrdexdb-expanded-v2/eval/expanded-v2-control")
V3_FORCE = Path("data/acceptance/hrdexdb-expanded-v3/eval/expanded-v3-force")


def load(path: Path) -> pd.DataFrame | None:
    """Load a completed cell without dropping invalid scores or duplicate predictions."""
    if not path.is_file():
        return None
    d = pd.read_parquet(path).set_index("window_id")
    if d.empty or d.index.has_duplicates or d.index.isna().any():
        raise ValueError(f"empty or duplicate window ids: {path}")
    if not np.isfinite(d["expected_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"nonfinite expected scores: {path}")
    if not d["expected_score"].between(1, 5).all():
        raise ValueError(f"expected scores outside [1, 5]: {path}")
    truth = d["truth_target"].str.extract(r"Score:\s*([1-5])\s*$")[0]
    if truth.isna().any():
        raise ValueError(f"invalid truth target: {path}")
    d["label"] = (truth.astype(int) >= TH).astype(int)
    d["pred"] = (d["expected_score"] >= TH).astype(int)
    d["ok"] = d["pred"] == d["label"]
    return d


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require complete identical evaluation populations before pairing by window id."""
    if a.empty or b.empty or a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("cannot pair empty or duplicate evaluation windows")
    if set(a.index) != set(b.index):
        raise ValueError("evaluation windows differ; refusing to compare an intersection")
    b = b.loc[a.index]
    for column in ("source", "episode_id", "object", "truth_target"):
        if column not in a or column not in b or a[column].isna().any() or b[column].isna().any():
            raise ValueError(f"missing paired identity: {column}")
        if not a[column].equals(b[column]):
            raise ValueError(f"paired identity differs: {column}")
    if a["label"].nunique() != 2:
        raise ValueError("AUROC comparison requires both outcome classes")
    return a, b


def paired(a: pd.DataFrame, b: pd.DataFrame, *, resamples: int = 3000) -> str:
    """Compute paired differences with a seeded bootstrap over whole objects."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    a, b = align(a, b)
    wins, losses = int((a.ok & ~b.ok).sum()), int((~a.ok & b.ok).sum())
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    objs = a["object"].to_numpy()
    uo = np.unique(objs)
    rng = np.random.default_rng(0)
    t = a["label"].to_numpy()
    sa, sb = a["expected_score"].to_numpy(), b["expected_score"].to_numpy()
    d_auc, d_acc = [], []
    for _ in range(resamples):
        pick = rng.choice(uo, len(uo))
        i = np.concatenate([np.flatnonzero(objs == o) for o in pick])
        d_acc.append(a.ok.to_numpy()[i].mean() - b.ok.to_numpy()[i].mean())
        if t[i].min() != t[i].max():
            d_auc.append(roc_auc_score(t[i], sa[i]) - roc_auc_score(t[i], sb[i]))
    if not d_auc:
        raise ValueError("no bootstrap draws contain both classes; increase resamples")
    lo, hi = np.percentile(d_auc, [2.5, 97.5])
    alo, ahi = np.percentile(d_acc, [2.5, 97.5])
    return (
        f"wins {wins} losses {losses} McNemar p={p:.3f}; "
        f"acc diff {a.ok.mean() - b.ok.mean():+.3f} "
        f"CI [{alo:+.3f},{ahi:+.3f}]; AUROC diff "
        f"{roc_auc_score(t, sa) - roc_auc_score(t, sb):+.3f} CI [{lo:+.3f},{hi:+.3f}]"
    )


def report(args: argparse.Namespace) -> None:
    """Print baseline and arm comparisons without selecting a model or threshold."""
    for split in ("val", "test"):
        bench = f"hrdexdb_aligned_{split}"
        control = load(V2_CONTROL / bench / "omitted/predictions.parquet")
        v3 = load(V3_FORCE / bench / "real/predictions.parquet")
        if control is None or v3 is None:
            raise FileNotFoundError(f"missing baseline predictions for {split}")
        print(f"\n## {split} ({len(control)} clips)")
        print(
            f"v2 control      acc {control.ok.mean():.3f} "
            f"auroc {roc_auc_score(control.label, control.expected_score):.3f}"
        )
        print(
            f"v3 tactile real acc {v3.ok.mean():.3f} "
            f"auroc {roc_auc_score(v3.label, v3.expected_score):.3f} "
            f"vs control: {paired(v3, control)}"
        )
        for label, directory in args.cell:
            found = False
            for mode in args.modes.split(","):
                d = load(Path(directory) / bench / mode / "predictions.parquet")
                if d is None:
                    continue
                found = True
                print(
                    f"{label} {mode:11s} acc {d.ok.mean():.3f} "
                    f"auroc {roc_auc_score(d.label, d.expected_score):.3f} "
                    f"pred-success {d.pred.mean():.2f}\n    vs control: {paired(d, control)}"
                    f"\n    vs v3 real: {paired(d, v3)}"
                )
            if args.require_complete and not found:
                raise FileNotFoundError(f"no completed predictions for {label} {split}")


def main() -> None:
    """Compare completed cells, optionally writing a reviewable Markdown artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--modes", default="real,video_blank,chart_text,chart,shuffled")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require each requested arm to have predictions on both splits.",
    )
    parser.add_argument("--output", type=Path, help="Save the comparison as Markdown.")
    args = parser.parse_args()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("# Paired HRDexDB arm comparison")
        print(
            "\nFixed success threshold 4.0; exact two-sided McNemar test; "
            "3,000 object-bootstrap draws."
        )
        print("Intervals are pointwise; p-values are unadjusted across recipe comparisons.")
        report(args)
    text = buffer.getvalue()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
