"""Pre-registered view selection on the TRAIN camera-study subset (released FLORA, zero-shot).

Rule (fixed 2026-10-23 before any study result): rank the single native views by AUROC (failure
positive, risk = -E[s]); the collage candidates are the best single view and the top-2, top-3 and
top-4 views in rank order (phase 2 scores them). The chosen collage is the candidate with the
highest AUROC, except that a candidate with more views must beat every smaller candidate by at
least 0.01 AUROC to be preferred (more views cost tokens: 1 view ~3.2k, 4 views ~13.8k for 64
frames). Prints the ranked serials on line 1 and, when phase 2 results are given, the chosen
collage on line 2.

Usage: choose_views.py phase1.jsonl [phase2.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC of ``s`` for the positives ``y == 1`` (ties averaged)."""
    from scipy.stats import rankdata  # type: ignore[import-not-found]

    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def table(paths: list[str]) -> dict[str, dict[str, tuple[float, int]]]:
    """``{config: {episode: (expected score, label)}}`` from study output files."""
    by: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
    for p in paths:
        with open(p) as fh:
            lines = fh.read().splitlines()
        for line in lines:
            if line.strip():
                r = json.loads(line)
                ep, kind, cams = r["id"].split("|")
                by[f"{kind}|{cams}"][ep] = (r["expected_score"], r["label"])
    return by


def score(by: dict, cfg: str, eps: list[str]) -> float:
    """AUROC of one config on ``eps`` (failure positive, risk = -E[s])."""
    es = np.array([by[cfg][e][0] for e in eps])
    y = np.array([by[cfg][e][1] <= 2 for e in eps], dtype=int)
    return auroc(y, -es)


def main() -> None:
    """Print the ranked serials, then the chosen collage when phase 2 is given."""
    by = table(sys.argv[1:])
    singles = [c for c in by if c.startswith("native|") and "+" not in c]
    eps = sorted(set.intersection(*[set(by[c]) for c in singles]))
    ranked = sorted(singles, key=lambda c: -score(by, c, eps))
    serials = [c.split("|")[1] for c in ranked]
    print(",".join(serials))
    for c in ranked:
        print(f"# {c} AUROC {score(by, c, eps):.3f} n={len(eps)}", file=sys.stderr)
    multi = [f"native|{'+'.join(serials[:k])}" for k in (2, 3, 4)]
    if all(m in by for m in multi):
        cands = [ranked[0], *multi]
        eps2 = sorted(set.intersection(*[set(by[c]) for c in cands]))
        aucs = [score(by, c, eps2) for c in cands]
        best = 0
        for k in range(1, len(cands)):
            if all(aucs[k] >= aucs[j] + 0.01 for j in range(k)):
                best = k
        for c, a in zip(cands, aucs, strict=True):
            print(f"# {c} AUROC {a:.3f} n={len(eps2)}", file=sys.stderr)
        print(cands[best].split("|")[1].replace("+", ","))


if __name__ == "__main__":
    main()
