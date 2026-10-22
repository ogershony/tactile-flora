#!/bin/bash
#SBATCH --account=scc
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --nodelist=matthew
#SBATCH --gres=mi210:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=flora-expanded-sft
#SBATCH --output=/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v2/jobs/train-%j.log
set -euo pipefail
cd /home/ogershony/tactile-flora
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1

data/cluster/rocm-venv/bin/python - <<'ENVIRONMENT'
import json,platform,importlib.metadata
from pathlib import Path
import torch
result={"host":platform.node(),"torch":torch.__version__,"hip":torch.version.hip,
        "packages":{name:importlib.metadata.version(name) for name in ["transformers","peft","accelerate","numpy","av"]},
        "gpus":[{"index":i,"name":torch.cuda.get_device_name(i),"arch":torch.cuda.get_device_properties(i).gcnArchName,"total_gib":torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]}
Path('data/acceptance/hrdexdb-expanded-v2/training-environment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
ENVIRONMENT
data/cluster/rocm-venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/cluster/run_hrdexdb_sft.py --base data/acceptance/hrdexdb-expanded-v2 --run-name expanded-v2-force
data/cluster/rocm-venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/cluster/run_hrdexdb_sft.py --base data/acceptance/hrdexdb-expanded-v2 --config data/acceptance/hrdexdb-expanded-v2/config/train-control.yaml --run-name expanded-v2-control
