#!/bin/bash
# v4 stage 4: score the LOCKED test sets once per final model (released, control, force).
# Usage: stage4_test.sh [checkpoints...]   (default: released data/runs/v4_control data/runs/v4_force)
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY=/content/venv/bin/python; TFT=/content/venv/bin/tf; TFV=/content/vllm-venv/bin/tf
FC="--force-config configs/force.v4.yaml"; MODEL=$TACTILE_FLORA_MODEL_DIR
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a /content/logs/v4.status; }
TESTS=${TESTS:-"hrdexdb_v4_test reassemble_seg_test phantom_fail_test roboreward_500"}
CKS=${@:-released data/runs/v4_control data/runs/v4_force}
for ck in $CKS; do
  case $ck in released) MODES=omitted,real;; *v4_force*) MODES=real,omitted,shuffled;; *) MODES=omitted;; esac
  if [ "$ck" != released ]; then
    [ -f $ck/merged/config.json ] || $TFT eval merge-adapter --run $ck --base $MODEL > /content/logs/merge_$(basename $ck).log 2>&1
  fi
  $PY - "$ck" "$MODES" $TESTS <<'PY'
import sys, yaml
ck, modes, benches = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
# RoboRewardBench has no force: every checkpoint is scored there without a block
yaml.safe_dump({"seed": 0, "checkpoints": [ck], "modes": {b: (["omitted"] if b == "roboreward_500" else modes) for b in benches}}, open("/content/grid_v4_test.yaml", "w"), sort_keys=False)
PY
  $TFV eval grid --config /content/grid_v4_test.yaml --eval-config configs/eval/benchmarks.v4.yaml --backend vllm --model-dir $MODEL --checkpoint $ck $FC >> /content/logs/test_v4_$(basename $ck).log 2>&1
  st "test_$(basename $ck)_exit_$?"; pkill -9 -f EngineCore
done
