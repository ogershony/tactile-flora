"""DEV-only metrics per cell (reads only data/eval/*/hrdexdb_v4_dev/*/predictions.parquet).

Same definitions as analyze.py (failure = truth score <= 2, risk = -E[s], hard failure = score
<= 2), plus the E[s] spread and parse-failure rate the chain4 gate checks. TEST cells are never
opened. Usage: python scripts/v4/dev_peek.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import analyze as A  # noqa: E402


def main() -> None:
    """Print one line per DEV cell."""
    nan = float("nan")
    for f in sorted(Path("data/eval").glob("*/hrdexdb_v4_dev/*/predictions.parquet")):
        ck, mode = f.parts[-4], f.parts[-2]
        p = pd.read_parquet(f)
        df = pd.DataFrame({
            "episode_id": p.episode_id,
            "es": p.expected_score.astype(float),
            "score": p.score.astype(float),
            "truth": p.truth_target.str.extract(r"Score:\s*(\d)")[0].astype(float),
        }).dropna()
        df["fail"] = (df.truth <= 2).astype(int)
        m = A.metrics(df)
        g = lambda k: m.get(k, nan)  # noqa: E731
        hands = []
        for hand in ("inspire_dftp", "inspire_f1"):
            sub = df[[A.SPLIT[e]["hand"] == hand for e in df.episode_id]]
            hands.append(A.auroc(sub.fail.to_numpy(), -sub.es.to_numpy()) if 0 < sub.fail.sum() < len(sub) else nan)
        print(
            f"{ck:14s} {mode:9s} n {m['n']} f {m['failures']} AU {g('auroc'):.3f} "
            f"(dftp {hands[0]:.3f} f1 {hands[1]:.3f}) PR {g('auprc'):.3f} BA {g('ba'):.3f} "
            f"FR {g('fr'):.3f} SR {g('sr'):.3f} F80 {g('fr_at_sr80'):.3f} F90 {g('fr_at_sr90'):.3f} "
            f"MAE {m['mae_hard']:.2f} sd {df.es.std():.3f} pf {p.parse_failed.mean() if 'parse_failed' in p else nan:.3f} "
            f"scores {dict(sorted(df.score.value_counts().to_dict().items()))}"
        )


if __name__ == "__main__":
    main()
