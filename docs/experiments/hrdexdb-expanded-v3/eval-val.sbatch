#!/bin/bash
#SBATCH --account=scc
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodelist=matthew
#SBATCH --gres=mi210:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=flora-v3-val
#SBATCH --output=/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v3/jobs/v3-val-%j.log
set -euo pipefail
cd /home/ogershony/tactile-flora
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
data/cluster/rocm-venv/bin/python -u scripts/cluster/evaluate_hrdexdb_clean.py --base data/acceptance/hrdexdb-expanded-v3 --eval-config data/acceptance/hrdexdb-expanded-v3/config/eval-val.yaml --benchmark hrdexdb_aligned_val --control-run expanded-v3-control --force-run expanded-v3-force --force-config data/acceptance/hrdexdb-expanded-v3/config/force.yaml --force-modes real,shuffled,video_blank
