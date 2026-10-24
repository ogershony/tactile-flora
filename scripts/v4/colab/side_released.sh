#!/bin/bash
# Released FLORA on REASSEMBLE/PHANTOM tests while pass A renders; stopped before pass A ends.
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
printf 'seed: 0\ncheckpoints: [released]\nmodes:\n  reassemble_seg_test: [omitted, real]\n  phantom_fail_test: [omitted]\n' > /content/grid_side.yaml
nice -n 5 /content/vllm-venv/bin/tf eval grid --config /content/grid_side.yaml --eval-config configs/eval/benchmarks.v4.yaml --backend vllm --model-dir $TACTILE_FLORA_MODEL_DIR --checkpoint released --force-config configs/force.v4.yaml >> /content/logs/side_released.log 2>&1 &
G=$!
while kill -0 $G 2>/dev/null; do
  n=$(grep -o '^\[[0-9]*/300' /content/logs/render_a.log | tail -1 | tr -d '[' | cut -d/ -f1)
  if [ "${n:-0}" -ge 275 ] || grep -q PASS_A_DONE /content/logs/render_a.log; then
    echo "watchdog: pass A at $n, stopping $(date -u +%FT%TZ)" >> /content/logs/side_released.log
    kill -9 $G; pkill -9 -f 'tf eval grid'; break
  fi
  sleep 20
done
pkill -9 -f EngineCore
echo "side_done $(date -u +%FT%TZ)" >> /content/logs/side_released.log
