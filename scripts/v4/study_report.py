"""AUROC (failure = positive, risk = -E[s]) per config with object-cluster bootstrap CIs, and paired
differences against a reference config.
"""

import json
import sys
from collections import defaultdict

import numpy as np

recs = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
ref = sys.argv[2] if len(sys.argv) > 2 else None
by = defaultdict(dict)
for r in recs:
    ep, kind, cams = r["id"].split("|")
    by[f"{kind}|{cams}"][ep] = (
        r["expected_score"],
        r["label"],
        r["group"],
        r["prompt_tokens"],
        r["score"],
    )


def auroc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = sums[inv] / cnt[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rng = np.random.default_rng(0)
common = set.intersection(*[set(v) for v in by.values()])
eps = sorted(common)
groups = sorted({by[next(iter(by))][e][2] for e in eps})
gidx = {g: [i for i, e in enumerate(eps) if by[next(iter(by))][e][2] == g] for g in groups}
boots = [np.concatenate([gidx[g] for g in rng.choice(groups, len(groups))]) for _ in range(2000)]
print(f"{len(eps)} clips in every config, {len(groups)} objects")
res = {}
for cfg, d in sorted(by.items()):
    es = np.array([d[e][0] for e in eps])
    y = np.array([d[e][1] <= 2 for e in eps], dtype=int)
    risk = -es
    hard = np.array([d[e][4] <= 2 for e in eps])
    a = auroc(y, risk)
    ba = 0.5 * (hard[y == 1].mean() + (~hard[y == 0]).mean())
    res[cfg] = risk
    toks = np.mean([d[e][3] for e in eps])
    line = (
        f"{cfg:45s} AUROC {a:.3f}  hardBA {ba:.3f}  FR {hard[y == 1].mean():.3f}  tokens {toks:.0f}"
    )
    if ref and ref in by and cfg != ref:
        rr = -np.array([by[ref][e][0] for e in eps])
        dd = [auroc(y[i], risk[i]) - auroc(y[i], rr[i]) for i in boots if 0 < y[i].sum() < len(i)]
        line += f"  vs ref {np.mean(dd):+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]"
    print(line)
