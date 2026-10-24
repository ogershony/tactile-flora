"""Print the result tables of writeup/baxter-rollouts/README.md from the finished evaluation cells.

Reads ``<base>/eval/<tag>/baxter_rollouts/<mode>/{metrics.json,predictions.parquet}`` and the
representation audit (labels, force-only classifier probability) and writes markdown to stdout.
Cells that do not exist yet are skipped, so it can be run while the job is still going.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/acceptance/baxter-rollouts-v1")
CELLS = [
    ("Baseline FLORA, no force", "released", "omitted"),
    ("Baseline FLORA + force block", "released", "real"),
    ("Fine-tuned FLORA (video only)", "expanded-v2-control", "omitted"),
    ("Fine-tuned tactile FLORA, real force", "expanded-v3-force", "real"),
    ("Fine-tuned tactile FLORA, shuffled force", "expanded-v3-force", "shuffled"),
    ("Fine-tuned tactile FLORA, force removed", "expanded-v3-force", "omitted"),
]
SHORT = {
    ("released", "omitted"): "baseline",
    ("released", "real"): "baseline+force",
    ("expanded-v2-control", "omitted"): "fine-tuned",
    ("expanded-v3-force", "real"): "tactile",
    ("expanded-v3-force", "shuffled"): "tactile, shuffled",
    ("expanded-v3-force", "omitted"): "tactile, no force",
}
THRESHOLD = 4.0


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability a random success scores above a random failure (ties count half)."""
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main() -> None:
    """Emit the headline, per-episode and rationale tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    audit = json.loads((args.base / "representation-audit.json").read_text())
    records = {r["episode_id"]: r for r in audit["records"]}
    episodes = list(records)
    truth = pd.Series({e: records[e]["outcome"] == 5 for e in episodes})
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    metrics: dict[tuple[str, str], dict] = {}
    for _, tag, mode in CELLS:
        d = args.base / "eval" / tag / "baxter_rollouts" / mode
        if not (d / "metrics.json").is_file():
            continue
        metrics[(tag, mode)] = json.loads((d / "metrics.json").read_text())
        tables[(tag, mode)] = pd.read_parquet(d / "predictions.parquet").set_index("episode_id")

    print("### Headline\n")
    print(
        "Accuracy at the 4.0 threshold on E[s]; AUROC over the 2 successes x 8 failures pairs; "
        "MAE on the 0-100 scale.\n"
    )
    print(
        "| Model | Accuracy | AUROC | MAE | Successes right (of 2) | Failures right (of 8) "
        "| Hard scores given |"
    )
    print("|---|---:|---:|---:|---:|---:|---|")
    for name, tag, mode in CELLS:
        if (tag, mode) not in tables:
            print(f"| {name} | pending | | | | | |")
            continue
        t = tables[(tag, mode)].loc[episodes]
        es = t["expected_score"].to_numpy(dtype=float)
        pred = es >= THRESHOLD
        y = truth.loc[episodes].to_numpy()
        m = metrics[(tag, mode)]["metrics"]
        counts = t["score"].astype(int).value_counts().sort_index()
        hard = ", ".join(f"{k}x{v}" for k, v in counts.items())
        print(
            f"| {name} | {m['accuracy']:.2f} | {m['auroc']:.3f} | {m['mae']:.0f} | "
            f"{int((pred & y).sum())} | {int((~pred & ~y).sum())} | {hard} |"
        )
    p = np.asarray([records[e]["classifier_p_success"] for e in episodes])
    y = truth.loc[episodes].to_numpy()
    print(
        f"| Force-only classifier (from the block) | {np.mean((p >= 0.5) == y):.2f} | "
        f"{auroc(y, p):.3f} | | {int(((p >= 0.5) & y).sum())} | "
        f"{int(((p < 0.5) & ~y).sum())} | P(success) |"
    )

    print("\n### Per-episode expected score E[s] (success threshold 4.0)\n")
    cols = [(t, m) for _, t, m in CELLS if (t, m) in tables]
    print("| Episode | Label | " + " | ".join(SHORT[c] for c in cols) + " | force P(success) |")
    print("|---|---|" + "---:|" * (len(cols) + 1))
    wins: list[str] = []
    for e in episodes:
        lab = "success" if truth[e] else "failure"
        cells = []
        right = {}
        for c in cols:
            es = float(tables[c].loc[e, "expected_score"])
            ok = (es >= THRESHOLD) == truth[e]
            right[c] = ok
            cells.append(f"{es:.2f}" + (" ok" if ok else ""))
        tac = ("expanded-v3-force", "real")
        base = ("released", "omitted")
        ctrl = ("expanded-v2-control", "omitted")
        if (
            tac in right
            and base in right
            and ctrl in right
            and right[tac]
            and not right[base]
            and not right[ctrl]
        ):
            wins.append(e)
            cells[cols.index(tac)] = "**" + cells[cols.index(tac)] + "**"
        print(
            f"| {e} | {lab} | "
            + " | ".join(cells)
            + f" | {records[e]['classifier_p_success']:.2f} |"
        )
    print(
        "\n'ok' marks a correct call. Bold: tactile FLORA right where both baseline and "
        "fine-tuned FLORA are wrong."
    )
    print(f"\nEpisodes where tactile FLORA alone is right: {', '.join(wins) if wins else 'none'}.")

    print("\n### What each model said (rationale, hard score)\n")
    for e in episodes:
        print(f"**{e}** ({'success' if truth[e] else 'failure'})\n")
        for _, tag, mode in CELLS[:4]:
            if (tag, mode) not in tables:
                continue
            row = tables[(tag, mode)].loc[e]
            parsed = json.loads(row["parsed"]) if isinstance(row["parsed"], str) else {}
            rat = (parsed.get("rationale") or "").strip().replace("\n", " ")
            print(
                f"- {SHORT[(tag, mode)]}: score {int(row['score'])} "
                f"(E[s] {float(row['expected_score']):.2f}). {rat}"
            )
        print()


if __name__ == "__main__":
    main()
