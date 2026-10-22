#!/bin/bash
#SBATCH --account=scc
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=flora-expanded-report
#SBATCH --output=/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v2/jobs/summary-%j.log
set -euo pipefail
cd /home/ogershony/tactile-flora
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
data/cluster/rocm-venv/bin/python -u scripts/cluster/summarize_hrdexdb_expanded.py
