#!/bin/bash
#SBATCH --account=scc
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=flora-v3-report
#SBATCH --output=/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v3/jobs/v3-report-%j.log
set -euo pipefail
cd /home/ogershony/tactile-flora
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
data/cluster/rocm-venv/bin/python -u scripts/cluster/summarize_hrdexdb_expanded.py --base data/acceptance/hrdexdb-expanded-v3 --force-run expanded-v3-force --control-run expanded-v3-control
