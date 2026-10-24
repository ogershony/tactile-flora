"""Compact, screen-sized summary of results.json (scripts/v4/analyze.py) for terminal reading.

One line per cell: AUROC [CI] | AUPRC | BA FR SR at the hard score | FR@SR80 FR@SR90 | MAE | n/fail,
then the paired comparisons (dAUROC [CI] p, dFR, McNemar) and the DEV-cut operating points.
Usage: python scripts/v4/summarize_results.py data/eval/results.json [benchmark-substring]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Print the summary."""
    res = json.loads(Path(sys.argv[1]).read_text())
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    cells = {k: v for k, v in res["cells"].items() if want in k}
    for k in sorted(cells, key=lambda k: (k.split("|")[1], k)):
        m = cells[k]
        if "auroc" in m:
            ci = m["auroc_ci"]
            print(
                f"{k:52s} AU {m['auroc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] PR {m['auprc']:.3f} "
                f"BA {m['ba']:.3f} FR {m['fr']:.3f} SR {m['sr']:.3f} "
                f"F80 {m['fr_at_sr80']:.3f} F90 {m['fr_at_sr90']:.3f} MAE {m['mae_hard']:.2f} "
                f"n {m['n']}/{m['failures']}"
            )
        else:
            print(f"{k:52s} MAE {m['mae_hard']:.3f} (E[s] {m['mae_es']:.3f}) n {m['n']}")
    for k, r in res["paired"].items():
        if want in k:
            c = r["d_auroc_ci"]
            print(
                f"{k:52s} dAU {r['d_auroc']:+.3f} [{c[0]:+.3f},{c[1]:+.3f}] p {r['delong_p']:.1e} "
                f"dBA {r['d_ba']:+.3f} dFR {r['d_fr']:+.3f} McN {r['mcnemar'][0]}/{r['mcnemar'][1]} "
                f"p {r['mcnemar_p']:.1e}"
            )
    for k, v in sorted(res.get("dev_threshold", {}).items()):
        if want in k:
            print(f"{k:52s} DEVCUT {v['cut']:.3f} BA {v['ba']:.3f} FR {v['fr']:.3f} SR {v['sr']:.3f}")
    print("HOLM", {k: f"{v:.1e}" for k, v in res.get("holm_primary", {}).items()})


if __name__ == "__main__":
    main()
