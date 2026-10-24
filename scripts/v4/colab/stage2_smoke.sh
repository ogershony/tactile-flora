#!/bin/bash
# v4 stage 2: set the chosen collage, then a memory/speed smoke run: one optimizer step on the 8
# longest HRDexDB training clips (force arm, real force) with the full v4 recipe.
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/content/venv/bin/python; TF=/content/venv/bin/tf
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a /content/logs/v4.status; }
CHOSEN=${1:-$(cat /content/study/chosen.txt)}
$PY - "$CHOSEN" <<'PY'
import sys, yaml
cams = sys.argv[1].split(",")
p = "configs/sources/hrdexdb.yaml"; d = yaml.safe_load(open(p)); d["hrdexdb_cameras"] = cams
yaml.safe_dump(d, open(p, "w"), sort_keys=False); print("hrdexdb_cameras", cams)
PY
sed -i "s#^base_model: nvidia/Flora-Rewarder#base_model: /content/models/Flora-Rewarder#" configs/train/v4_*.yaml
$PY - <<'PY'
import pyarrow as pa, pyarrow.parquet as pq, yaml
from tactile_flora.verify import manifest
rows = [r for r in manifest.read_manifest("data/manifest/v4_train.parquet") if r.source == "hrdexdb"]
# longest trials first: duration is not in the manifest, so use the frame cache's frame counts
import json
from pathlib import Path
def n(r):
    m = json.loads((Path("data/frames/hrdexdb") / r.episode_id / "video.json").read_text())
    return max(v["n_frames"] for v in m.values())
rows = sorted(rows, key=n, reverse=True)[:8]
pq.write_table(pa.Table.from_pylist([r.model_dump() for r in rows], schema=manifest._SCHEMA), "data/manifest/v4_smoke.parquet")
d = yaml.safe_load(open("configs/train/v4_force.yaml")); d["manifest"] = "data/manifest/v4_smoke.parquet"; d["effective_batch"] = 8
yaml.safe_dump(d, open("configs/train/v4_smoke.yaml", "w"), sort_keys=False)
print("smoke rows", len(rows))
PY
st smoke_start
$TF train --config configs/train/v4_smoke.yaml --force-config configs/force.v4.yaml > /content/logs/train_v4_smoke.log 2>&1
rc=$?
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> /content/logs/train_v4_smoke.log
st "smoke_exit_$rc"
tail -n 15 /content/logs/train_v4_smoke.log
