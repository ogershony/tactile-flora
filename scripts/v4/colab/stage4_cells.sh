#!/bin/bash
# Score one checkpoint on the given benchmarks in the given force modes (one tf eval grid call).
# Usage: stage4_cells.sh <checkpoint> <modes,comma> <benchmark ...>
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY=/content/venv/bin/python; TFV=/content/vllm-venv/bin/tf; MODEL=$TACTILE_FLORA_MODEL_DIR; ck=$1; modes=$2; shift 2
$PY - "$ck" "$modes" "$@" <<'PY'
import sys, yaml
ck, modes, benches = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
yaml.safe_dump({"seed": 0, "checkpoints": [ck], "modes": {b: modes for b in benches}}, open("/content/grid_v4_cells.yaml", "w"), sort_keys=False)
PY
$TFV eval grid --config /content/grid_v4_cells.yaml --eval-config configs/eval/benchmarks.v4.yaml --backend vllm --model-dir $MODEL --checkpoint $ck --force-config configs/force.v4.yaml >> /content/logs/test_v4_$(basename $ck).log 2>&1
echo "cells_$(basename $ck)_${modes//,/+}_exit_$? $(date -u +%FT%TZ)" >> /content/logs/v4.status
pkill -9 -f EngineCore
