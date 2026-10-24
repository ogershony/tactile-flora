"""Check the batched vLLM path against one generate call per clip on the same engine.

Runs harness.run_eval twice on one benchmark cell (default phantom_fail_test, omitted) into a
scratch data root: once through a wrapper that exposes only ``generate`` (the per-clip path),
once through VLLMBackend.generate_many (TF_VLLM_BATCH, default 8). Prints wall time per clip
for each and the largest |delta E[s]| and score disagreements; exits 1 if any score differs,
mean |delta E[s]| > 0.02 or max > 0.10. Result 2026-10-23 (PHANTOM test, omitted, batch 8): per clip
4.68 s, batched 1.36 s; mean 0.0067, max 0.056, no score change.
Usage: python scripts/v4/batch_parity.py [benchmark] [mode]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

from tactile_flora.config import EvalConfig, ForceConfig, load_config
from tactile_flora.eval import harness
from tactile_flora.eval.inference import VLLMBackend
from tactile_flora.video.readers import SourceRootReader


class PerClip:
    """Only ``generate``: run_eval falls back to one call per clip."""

    name = "vllm"

    def __init__(self, inner: VLLMBackend) -> None:
        self.inner = inner

    def generate(self, *a, **k):  # noqa: ANN002, ANN003, ANN201, D102
        return self.inner.generate(*a, **k)


def main() -> None:
    """Run both paths and compare."""
    bench = sys.argv[1] if len(sys.argv) > 1 else "phantom_fail_test"
    mode = sys.argv[2] if len(sys.argv) > 2 else "omitted"
    ecfg = load_config(Path("configs/eval/benchmarks.v4.yaml"), EvalConfig)
    fcfg = load_config(Path("configs/force.v4.yaml"), ForceConfig)
    be = VLLMBackend("released", SourceRootReader(Path("configs/sources")))
    be.load()
    res = {}
    for tag, backend in (("per_clip", PerClip(be)), ("batched", be)):
        root = Path("/content/batch_parity") / tag
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "verified").exists():  # the provider reads norm constants under the root
            (root / "verified").symlink_to(Path("data/verified").resolve())
        t0 = time.time()
        cell = harness.run_eval("released", bench, mode, ecfg, fcfg, root, backend, seed=0)  # type: ignore[arg-type]
        dt = time.time() - t0
        p = pd.read_parquet(cell / "predictions.parquet")
        res[tag] = p.set_index("window_id")
        print(f"{tag}: {len(p)} clips in {dt:.0f} s = {dt / len(p):.2f} s/clip", flush=True)
    a, b = res["per_clip"], res["batched"].loc[res["per_clip"].index]
    d = (a.expected_score - b.expected_score).abs()
    flips = int((a.score != b.score).sum())
    print(f"max |dE[s]| {d.max():.4f}  mean {d.mean():.4f}  score disagreements {flips}/{len(a)}")
    print("batch", os.environ.get("TF_VLLM_BATCH", "8"))
    sys.exit(0 if flips == 0 and d.mean() <= 0.02 and d.max() <= 0.10 else 1)


if __name__ == "__main__":
    main()
