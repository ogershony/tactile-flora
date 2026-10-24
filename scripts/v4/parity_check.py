"""Gate: our harness (released FLORA, omitted force, flora collage, score-only) must reproduce
NVIDIA's batch_utils on the same cached frames: mean |delta E[s]| <= 0.02 and max <= 0.10 on 20
TRAIN study clips (bounds set from the measured engine noise, see the end of main).

Builds data/manifest/v4_parity.parquet (20 HRDexDB train rows from the camera-study subset) and
/content/bench_parity.yaml, runs ``tf eval run`` for released/omitted through vLLM, and compares
each clip's expected score with the batch_utils run of the same single camera (study phase 1,
``native|<camera>``). Usage: parity_check.py <phase1.jsonl> <camera> <model_dir>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tactile_flora.verify import manifest


def main() -> None:
    """Run the parity gate and exit non-zero when it fails."""
    phase1, cam, model = sys.argv[1], sys.argv[2], sys.argv[3]
    src = yaml.safe_load(open("configs/sources/hrdexdb.yaml"))
    saved = src["hrdexdb_cameras"]
    # The manifests were built with the chosen collage, whose first camera is the scene camera
    # (it sets duration_s and the video clock). Parity runs on that camera: with any other one
    # the windows of trials whose cameras differ by a frame (8 of 444 rendered) do not reproduce.
    if saved and saved[0] != cam:
        print(f"parity camera {saved[0]} (first of the chosen collage) instead of {cam}")
        cam = saved[0]
    ref = {}
    for line in Path(phase1).read_text().splitlines():
        r = json.loads(line)
        ep, kind, cams = r["id"].split("|")
        if kind == "native" and cams == cam:
            ref[ep] = r["expected_score"]
    rows = [r for r in manifest.read_manifest("data/manifest/hrdexdb_v4_train.parquet") if r.episode_id in ref]
    rows = sorted(rows, key=lambda r: r.episode_id)[:20]
    pq.write_table(pa.Table.from_pylist([r.model_dump() for r in rows], schema=manifest._SCHEMA),
                   "data/manifest/v4_parity.parquet")
    bench = {"output_dir": "data/eval_parity", "benchmarks": {"hrdexdb_parity": {
        "manifest": "data/manifest/v4_parity.parquet", "prompts": ["outcome"], "metrics": ["mae"],
        "score_only": True, "collage": "flora"}}}
    Path("/content/bench_parity.yaml").write_text(yaml.safe_dump(bench))
    src["hrdexdb_cameras"] = [cam]
    yaml.safe_dump(src, open("configs/sources/hrdexdb.yaml", "w"), sort_keys=False)
    try:
        out = subprocess.run(
            ["/content/vllm-venv/bin/tf", "eval", "run", "--config", "/content/bench_parity.yaml",
             "--checkpoint", "released", "--benchmark", "hrdexdb_parity", "--force-mode", "omitted",
             "--backend", "vllm", "--model-dir", model, "--force-config", "configs/force.v4.yaml"],
            capture_output=True, text=True, check=False)
        print(out.stdout[-2000:], out.stderr[-3000:])
    finally:
        src["hrdexdb_cameras"] = saved
        yaml.safe_dump(src, open("configs/sources/hrdexdb.yaml", "w"), sort_keys=False)
    pred = next(Path("data").glob("**/hrdexdb_parity/omitted/predictions.parquet"))
    p = pd.read_parquet(pred)
    d = [(e, float(es), ref[e]) for e, es in zip(p.episode_id, p.expected_score, strict=True)]
    diffs = [abs(a - b) for _, a, b in d]
    for e, a, b in d:
        print(f"{e:45s} harness {a:.3f} batch_utils {b:.3f} delta {a - b:+.3f}")
    mean_d = sum(diffs) / len(diffs)
    print(f"PARITY n={len(d)} max|d|={max(diffs):.4f} mean|d|={mean_d:.4f}")
    # Pass: mean |d| <= 0.02 and max |d| <= 0.10. The engine's own batch nondeterminism
    # (scripts/v4/batch_parity.py, 2026-10-23: one engine, identical inputs, batches of 8 vs one
    # clip per call, PHANTOM test 40 clips) is mean 0.0067 / max 0.056 with no hard-score change,
    # so the former max <= 0.02 would fail on noise; a preprocessing or prompt mismatch moves
    # most clips the same way, which the mean bound catches.
    sys.exit(0 if mean_d <= 0.02 and max(diffs) <= 0.10 else 1)


if __name__ == "__main__":
    main()
