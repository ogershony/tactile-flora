#!/bin/bash
# Pick the best checkpoint (<= MAXSTEP) by held-out chunk L1, stage it under deploy/, push to a PRIVATE HF model repo.
# usage: HF_WRITE_TOKEN=... publish.sh <run_dir> <maxstep> <tag> [exact_step]     (token only read from env, never echoed)
set -euo pipefail
RUN=$1; MAXSTEP=$2; TAG=$3; EXACT=${4:-}; REPO=OG7407/baxter-act-cube-left   # 4th arg: publish exactly this step instead of the best <= MAXSTEP
cd "${BAXTER_ACT_HOME:-$HOME/baxter_teleop}" && source .venv/bin/activate
BEST=$(python -c "
import json; d=json.load(open('$RUN/heldout_l1.json')); c={int(s):v['l1_chunk_mean'] for s,v in d.items() if int(s)<=$MAXSTEP}
b=min(c,key=c.get); print(b)")
[ -n "$EXACT" ] && BEST=$EXACT
SRC="$RUN/checkpoints/$(printf %06d $BEST)/pretrained_model"
DST="deploy/${TAG}"; rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC" "$DST/pretrained_model"; cp "$RUN/heldout_l1.json" "$RUN/loss_curves.png" "$DST/"
cat > "$DST/README.md" <<MD
---
license: other
library_name: lerobot
pipeline_tag: robotics
tags: [robotics, lerobot, act, baxter]
datasets: [OG7407/baxter-cube-pick-left]
---
# ACT policy: Baxter left-arm cube pick (checkpoint step $BEST, tag $TAG)
Trained with lerobot 0.6.1 on OG7407/baxter-cube-pick-left (50 eps, 15 fps; episodes 45-49 held out).
chunk_size=30, n_action_steps=15, resnet18, VAE on, global batch 65 (5x RTX 2080 Ti, fp16), lr 1e-5.
Inputs at 15 Hz: observation.state [8], observation.images.head and .wrist (360x640 RGB). Output: action [8]
(left_s0..left_w2 rad, left_gripper 0-1). Normalization stats live in the processor files next to the weights.
Load: ACTPolicy.from_pretrained('$REPO') (or the pretrained_model/ folder) + make_pre_post_processors(..., pretrained_path=...).
Held-out action L1 per checkpoint: see heldout_l1.json; curves: loss_curves.png.
MD
echo "best checkpoint <= $MAXSTEP: step $BEST  -> $(realpath $DST)"
python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
api.create_repo("$REPO", repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id="$REPO", folder_path="$DST", path_in_repo=".", commit_message="$TAG: step $BEST")
api.upload_folder(repo_id="$REPO", folder_path="$DST", path_in_repo="checkpoints/$TAG", commit_message="$TAG snapshot")
info = api.model_info("$REPO"); print("pushed to https://huggingface.co/$REPO  private =", info.private)
PY
