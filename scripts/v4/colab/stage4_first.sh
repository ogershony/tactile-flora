#!/bin/bash
# The real-force cells of one checkpoint on the given test sets (default: the three Holm sets), ahead
# of the full stage 4, so a runtime that ends early still leaves its primary comparison.
# Usage: stage4_first.sh <checkpoint> [benchmark ...]
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY=/content/venv/bin/python; TFV=/content/vllm-venv/bin/tf; MODEL=$TACTILE_FLORA_MODEL_DIR; ck=$1; shift
$PY - "$ck" "$@" <<'PY'
import sys, yaml
benches = sys.argv[2:] or ["hrdexdb_v4_test", "reassemble_seg_test", "phantom_fail_test"]
yaml.safe_dump({"seed": 0, "checkpoints": [sys.argv[1]], "modes": {b: ["real"] for b in benches}}, open("/content/grid_v4_first.yaml", "w"), sort_keys=False)
PY
$TFV eval grid --config /content/grid_v4_first.yaml --eval-config configs/eval/benchmarks.v4.yaml --backend vllm --model-dir $MODEL --checkpoint $ck --force-config configs/force.v4.yaml >> /content/logs/test_v4_$(basename $ck).log 2>&1
echo "test_first_$(basename $ck)_exit_$? $(date -u +%FT%TZ)" >> /content/logs/v4.status
pkill -9 -f EngineCore
