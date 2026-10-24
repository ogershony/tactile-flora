#!/bin/bash
# Exploratory (added 2026-10-23 17:35 UTC, after the force adapter's DEV cells showed video carries
# most of the HRDexDB signal; outside the pre-registered comparisons): the seed-0 force adapter with
# every frame replaced by mid-gray and the real force block, on the HRDexDB and REASSEMBLE test
# sets. Asks how far the adapter gets from touch alone, against the force-only classifier whose
# P(success) line the block carries.
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY=/content/venv/bin/python; TFV=/content/vllm-venv/bin/tf; MODEL=$TACTILE_FLORA_MODEL_DIR
$PY - <<'PY'
import yaml
modes = {"hrdexdb_v4_test": ["video_blank"], "reassemble_seg_test": ["video_blank"]}
yaml.safe_dump({"seed": 0, "checkpoints": ["data/runs/v4_force"], "modes": modes}, open("/content/grid_v4_blank.yaml", "w"), sort_keys=False)
PY
$TFV eval grid --config /content/grid_v4_blank.yaml --eval-config configs/eval/benchmarks.v4.yaml --backend vllm --model-dir $MODEL --checkpoint data/runs/v4_force --force-config configs/force.v4.yaml >> /content/logs/test_v4_blank.log 2>&1
echo "test_v4_force_blank_exit_$? $(date -u +%FT%TZ)" >> /content/logs/v4.status
pkill -9 -f EngineCore
