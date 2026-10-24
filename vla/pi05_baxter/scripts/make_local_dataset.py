"""Local copy of OG7407/baxter-cube-pick-left-v0 whose action stats describe relative chunks.

Pi0.5 is trained here on delta joint targets (action[t+k] - state[t] for the 7 joints, gripper kept
absolute), the openpi recipe for joint-position arms. LeRobot normalizes actions with
meta/stats.json["action"] before the model and unnormalizes after it, so those stats must describe
the relative chunks the model actually sees, not the absolute joint targets. This script copies the
dataset (17 MB) and rewrites only that entry, computed over the training episodes at the chosen
chunk length. Everything else is byte-identical to the Hub revision.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=10)
ap.add_argument("--val", default="5,14", help="held-out episode indices, excluded from the stats")
args = ap.parse_args()

src = Path(snapshot_download("OG7407/baxter-cube-pick-left-v0", repo_type="dataset"))
out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out, symlinks=False)

df = pd.read_parquet(out / "data/chunk-000/file-000.parquet")
val = {int(v) for v in args.val.split(",") if v}
deltas = []
for ep, x in df.groupby("episode_index"):
    if ep in val:
        continue
    s = np.stack(x["observation.state"].to_numpy())
    a = np.stack(x["action"].to_numpy())
    n = len(x)
    for t in range(n):
        idx = np.minimum(np.arange(t, t + args.chunk), n - 1)
        d = a[idx].copy()
        d[:, :7] -= s[t, :7]
        deltas.append(d)
d = np.concatenate(deltas).astype(np.float64)

stats = json.loads((out / "meta/stats.json").read_text())
old = stats["action"]
new = {
    "min": d.min(0).tolist(),
    "max": d.max(0).tolist(),
    "mean": d.mean(0).tolist(),
    "std": d.std(0).tolist(),
    "count": [int(d.shape[0])],
}
for q in (1, 10, 50, 90, 99):
    new[f"q{q:02d}"] = np.quantile(d, q / 100, axis=0).tolist()
for k in old:
    if k not in new:
        print("dropping unknown stat key", k)
stats["action"] = new
(out / "meta/stats.json").write_text(json.dumps(stats, indent=4))
np.set_printoptions(precision=4, suppress=True)
print("relative chunk samples", d.shape)
print("q01", np.array(new["q01"]))
print("q99", np.array(new["q99"]))
print("old abs q01", np.array(old["q01"]))
