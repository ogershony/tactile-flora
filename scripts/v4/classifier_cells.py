"""Force-only classifier baseline as eval cells: data/eval/force_classifier/<bench>/none/.

Each window of the benchmark manifest goes through the harness's real-mode path (provider, then
window_features with the source's norm constants), and the scorer that the force block quotes
(force.v4.yaml classifier_json, matched by source and arm as render.classifier_line does) gives
P(success), out-of-fold for its own training episodes. expected_score = 1 + 4 P, so the risk
ordering is 1 - P; score = 5 if P >= 0.5 else 1 (the classifier's own threshold, no tuning).
Benchmarks whose source has no scorer (PHANTOM) are skipped.
Usage: python scripts/v4/classifier_cells.py hrdexdb_v4_dev hrdexdb_v4_test reassemble_seg_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tactile_flora.config import EvalConfig, ForceConfig, load_config
from tactile_flora.eval import harness
from tactile_flora.force import render


def p_success(fcfg: ForceConfig, f, source: str, episode_id: str) -> float | None:
    """render.classifier_line's scorer choice, returning the unrounded probability."""
    for path in (fcfg.render.classifier_json or "").split(","):
        sc = render._scorer(path.strip())
        if source == sc.source and sc.arm in f.arms:
            return float(sc.score(f, episode_id))
    return None


def main() -> None:
    """Write one predictions.parquet per benchmark named on the command line."""
    data_root = Path("data")
    ecfg = load_config(Path("configs/eval/benchmarks.v4.yaml"), EvalConfig)
    fcfg = load_config(Path("configs/force.v4.yaml"), ForceConfig)
    for bench in sys.argv[1:]:
        spec = ecfg.benchmarks[bench]
        mpath = harness._manifest_path(spec.manifest, data_root)
        prov = harness.make_provider(
            mpath, set(spec.prompts), fcfg, Path("configs/sources"), None, data_root
        )
        batches = harness._load_batches(prov)
        consts = harness._read_consts(data_root, {b.episode.source for b in batches.values()})
        recs = []
        for b in batches.values():
            ep = b.episode
            for wid, w in zip(b.window_ids, b.windows, strict=True):
                f = harness.window_features(ep, w, consts[ep.source], fcfg, b.shift_s)
                p = p_success(fcfg, f, ep.source, ep.episode_id)
                if p is None:
                    continue
                recs.append(
                    {
                        "window_id": wid,
                        "source": ep.source,
                        "episode_id": ep.episode_id,
                        "expected_score": 1.0 + 4.0 * p,
                        "score": 5.0 if p >= 0.5 else 1.0,
                        "p_success": p,
                        "truth_target": w.target,
                    }
                )
        if not recs:
            print(bench, "no scorer for this source; skipped", flush=True)
            continue
        out = harness.cell_dir(data_root, "force_classifier", bench, "none")
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(out / "predictions.parquet", index=False)
        print(bench, len(recs), "->", out, flush=True)


if __name__ == "__main__":
    main()
