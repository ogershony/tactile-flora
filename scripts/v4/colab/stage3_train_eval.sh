#!/bin/bash
# v4 stage 3: train the matched force and control adapters, merge each, score DEV, then the test
# sets once (released, control, force). Resumable: finished adapters and cells are skipped.
# Usage: stage3_train_eval.sh [arms...]   (default: v4_force v4_control)
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/content/venv/bin/python; TFT=/content/venv/bin/tf; TFV=/content/vllm-venv/bin/tf
FC="--force-config configs/force.v4.yaml"; MODEL=$TACTILE_FLORA_MODEL_DIR
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a /content/logs/v4.status; }
ARMS=${@:-v4_force v4_control}
gpu_free() { while pgrep -f 'EngineCore|tf eval|tf train' > /dev/null; do sleep 30; done; }
prep2() {  # DEV/TEST rows once pass C (chosen collage) is rendered; TRAIN-only fits are reproduced exactly
  [ -f /content/study/prep2.done ] && return 0
  until grep -q PASS_C_DONE /content/logs/render_c.log 2>/dev/null; do sleep 60; done
  $PY scripts/v4/prepare_hrdexdb_v4.py --raw data/raw/hrdexdb --frames data/frames/hrdexdb --out data \
    --cameras "$(cat /content/study/chosen.txt)" > /content/logs/prep2.log 2>&1 && touch /content/study/prep2.done && st prep2_done
}
grid() {  # grid <ckpt> <modes> <benchmarks...>
  local ck=$1 modes=$2; shift 2
  $PY - "$ck" "$modes" "$@" <<'PY'
import sys, yaml
ck, modes, benches = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
yaml.safe_dump({"seed": 0, "checkpoints": [ck], "modes": {b: modes for b in benches}}, open("/content/grid_v4.yaml", "w"), sort_keys=False)
PY
  $TFV eval grid --config /content/grid_v4.yaml --eval-config configs/eval/benchmarks.v4.yaml --backend vllm --model-dir $MODEL --checkpoint $ck $FC >> /content/logs/eval_v4_$(basename $ck).log 2>&1
  local rc=$?; pkill -9 -f EngineCore; return $rc
}
for arm in $ARMS; do
  RUN=data/runs/$arm
  if [ ! -f $RUN/adapter/adapter_model.safetensors ]; then
    gpu_free; st "${arm}_train_start"
    $TFT train --config configs/train/$arm.yaml $FC --dry-run > /content/logs/train_${arm}_dry.log 2>&1; tail -n 6 /content/logs/train_${arm}_dry.log
    $TFT train --config configs/train/$arm.yaml $FC > /content/logs/train_$arm.log 2>&1
    [ -f $RUN/adapter/adapter_model.safetensors ] || { st "${arm}_TRAIN_FAILED"; tail -n 30 /content/logs/train_$arm.log; continue; }
    st "${arm}_train_done"
  fi
  [ -f $RUN/merged/config.json ] || $TFT eval merge-adapter --run $RUN --base $MODEL > /content/logs/merge_$arm.log 2>&1 || { st "${arm}_MERGE_FAILED"; continue; }
  case $arm in v4_force*) MODES=real,omitted,shuffled;; *) MODES=omitted;; esac
  prep2; gpu_free; grid $RUN $MODES hrdexdb_v4_dev; st "${arm}_dev_done"
done
# released FLORA on DEV through the same harness (omitted = the released prompt; real = + force text)
prep2; gpu_free; grid released omitted,real hrdexdb_v4_dev; st released_dev_done
echo "DEV done; test sets are scored by stage4_test.sh once the models are final"
