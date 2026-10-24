# Shared setup for the ACT training sbatch scripts. Sourced, not executed.
set -uo pipefail
PROJ="${BAXTER_ACT_HOME:-$HOME/baxter_teleop}"   # venv, outputs/, deploy/ live here, outside git
source "$PROJ/.venv/bin/activate"
# Token comes from the HF CLI cache (never written to scripts/logs).
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token")}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
cd "$PROJ"
echo "[act] host $(hostname) job=${SLURM_JOB_ID:-none} start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
