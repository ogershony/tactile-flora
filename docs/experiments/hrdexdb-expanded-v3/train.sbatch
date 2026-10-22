#!/bin/bash
#SBATCH --account=scc
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --nodelist=matthew
#SBATCH --gres=mi210:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=flora-v3-sft
#SBATCH --output=/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v3/jobs/v3-sft-%j.log
set -euo pipefail
cd /home/ogershony/tactile-flora
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
data/cluster/rocm-venv/bin/python - <<'ENVIRONMENT'
import json,platform,importlib.metadata
from pathlib import Path
import torch
names=['transformers','peft','accelerate','numpy','av']
props=torch.cuda.get_device_properties
result={'host':platform.node(),'torch':torch.__version__,'hip':torch.version.hip,
 'packages':{n:importlib.metadata.version(n) for n in names},
 'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),'arch':props(i).gcnArchName,
  'total_gib':props(i).total_memory/2**30} for i in range(torch.cuda.device_count())]}
Path('data/acceptance/hrdexdb-expanded-v3/training-environment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
ENVIRONMENT
data/cluster/rocm-venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/cluster/run_hrdexdb_sft.py --base data/acceptance/hrdexdb-expanded-v3 --force-config data/acceptance/hrdexdb-expanded-v3/config/force.yaml --run-name expanded-v3-force
data/cluster/rocm-venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/cluster/run_hrdexdb_sft.py --base data/acceptance/hrdexdb-expanded-v3 --force-config data/acceptance/hrdexdb-expanded-v3/config/force.yaml --config data/acceptance/hrdexdb-expanded-v3/config/train-control.yaml --run-name expanded-v3-control
