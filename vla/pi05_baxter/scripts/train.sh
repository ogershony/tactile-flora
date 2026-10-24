#!/usr/bin/env bash
# Pi0.5 LoRA fine-tune on the Baxter left-arm cube pick (OG7407/baxter-cube-pick-left-v0).
# Frozen PaliGemma VLM (SigLIP + Gemma 2B); LoRA r32 on every linear layer of the 300M action
# expert plus the action/time projections; relative joint targets over a 10-step (1.0 s) chunk.
# Usage: scripts/train.sh <run_name> <steps> [extra lerobot-train args...]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1; STEPS=$2; shift 2
TRAIN_EPS='[0,1,2,3,4,6,7,8,9,10,11,12,13,15,16,17]'   # 5 and 14 held out for validation
TARGETS='.*gemma_expert\.model\.layers\.[0-9]+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)'
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python scripts/train_entry.py \
  --dataset.repo_id=OG7407/baxter-cube-pick-left-v0 \
  --dataset.root="$PWD/data/baxter-cube-pick-left-v0-rel10" \
  --dataset.episodes="$TRAIN_EPS" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.path="$PWD/assets/pi05_base_local" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.optimizer_lr=2e-4 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.scheduler_decay_lr=2e-5 \
  --peft.method_type=LORA \
  --peft.r=32 \
  --peft.lora_alpha=32 \
  --peft.target_modules="$TARGETS" \
  --batch_size=32 \
  --steps="$STEPS" \
  --num_workers=8 \
  --log_freq=25 \
  --save_freq=500 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir="runs/$RUN" \
  --job_name="$RUN" \
  "$@"
