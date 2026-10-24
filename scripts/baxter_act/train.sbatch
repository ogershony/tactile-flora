#!/bin/bash
#SBATCH -J act-baxter
#SBATCH -A tritondroids
#SBATCH -p long
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH -w zixian
#SBATCH --gres=rtx2080ti:5
#SBATCH -c 25
#SBATCH --mem=50G
#SBATCH -o /home/%u/.cache/lerobot-slurm/%j-act-baxter.out
# ACT on OG7407/baxter-cube-pick-left, multi-GPU via HF Accelerate (lerobot-train uses it internally).
# Preemption on `long` requeues the job; on restart it resumes from checkpoints/last of RUN.
# Knobs: RUN (output dir name), STEPS, BS (per-GPU batch), NGPU (must match --gres).
# Group limit on account tritondroids is cpu=32, mem=96G, rtx2080ti=8 shared with the HOPE jobs, hence 5 GPUs x 13 = 65 global batch.
source "${TACTILE_FLORA_ROOT:-$HOME/tactile-flora}/scripts/baxter_act/slurm_common.sh"
RUN="${RUN:-act_baxter_cube_left}"
STEPS="${STEPS:-50000}"
BS="${BS:-13}"
NGPU="${NGPU:-5}"
OUT="outputs/$RUN"
LAST="$OUT/checkpoints/last/pretrained_model/train_config.json"
COMMON=( --policy.push_to_hub=false --num_workers=4 --wandb.enable=false )
if [ -f "$LAST" ]; then
  echo "[act] resuming from $LAST"
  ARGS=( --config_path="$LAST" --resume=true )
else
  ARGS=(
    --dataset.repo_id=OG7407/baxter-cube-pick-left
    --dataset.eval_split=0.1 --eval_steps=1000
    --policy.type=act --policy.device=cuda
    --policy.chunk_size=30 --policy.n_action_steps=15
    --batch_size="$BS" --steps="$STEPS" --save_freq=5000 --log_freq=200
    --output_dir="$OUT" --job_name="$RUN"
  )
fi
accelerate launch --multi_gpu --num_processes="$NGPU" --mixed_precision=fp16 \
  --main_process_port=$((29500 + SLURM_JOB_ID % 1000)) \
  "$(which lerobot-train)" "${ARGS[@]}" "${COMMON[@]}"
rc=$?
echo "[act] end $(date) exit=$rc"; exit $rc
