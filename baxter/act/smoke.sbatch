#!/bin/bash
#SBATCH -J act-smoke
#SBATCH -A tritondroids
#SBATCH -p debug
#SBATCH -w zixian
#SBATCH --gres=rtx2080ti:1
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH -o /home/%u/.cache/lerobot-slurm/%j-act-smoke.out
source "${TACTILE_FLORA_ROOT:-$HOME/tactile-flora}/baxter/act/slurm_common.sh"
rm -rf outputs/act_baxter_smoke
lerobot-train \
  --dataset.repo_id=OG7407/baxter-cube-pick-left \
  --policy.type=act --policy.device=cuda \
  --policy.chunk_size=30 --policy.n_action_steps=15 \
  --policy.push_to_hub=false \
  --batch_size=8 --steps=300 --save_freq=300 --log_freq=50 \
  --num_workers=4 \
  --output_dir=outputs/act_baxter_smoke --job_name=act_baxter_smoke --wandb.enable=false
echo "[act] end $(date) exit=$?"
