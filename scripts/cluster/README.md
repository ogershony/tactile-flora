# Cluster training smoke

## Codex session access

`codex-slurm.toml` is a configuration fragment for a new session that needs to run native Slurm
commands. Merge it into the configuration used by the launcher, outside a restricted session,
and restart the client. It enables command networking and local/private-network access while
inheriting workspace filesystem permissions. It permits networking generally, rather than
enforcing a Slurm-only destination allowlist. Managed policy or explicit launcher settings can
still override it. Do not combine permission profiles with legacy `sandbox_mode` or
`sandbox_workspace_write` configuration. Creating this fragment does not change an active
session's enforced permissions.

Reference: [official permission profile documentation](https://learn.chatgpt.com/docs/permissions).

An allocation created in another terminal does not move an existing Codex session to that node.
After networking is enabled, use the existing allocation's job ID for job steps, for example
`srun --jobid=JOB_ID hostname` and `srun --jobid=JOB_ID nvidia-smi`. Alternatively, start Codex
from a shell running inside the allocated compute node. Keep the terminal holding the allocation
open until GPU work finishes.

## Training check

`smoke_train.py` checks that the Hugging Face Qwen3.5 model, PEFT LoRA wrapping, the project's
answer-only SFT loop, and adapter serialization run together. It trains against one repeated
synthetic token sequence using a randomly initialized, tiny four-layer model. It never downloads
or loads a released model and its falling loss is only a plumbing check, not a quality result or
evidence that the released FLORA model is accepted on a cluster.

## Requirements

Use Python 3.11 and install this checkout so `tactile_flora` is importable. Install PyTorch with
the CUDA or ROCm wheel appropriate for the allocated node, then install compatible versions of
torchvision, Transformers, PEFT, and safetensors. The repository's `[train]` extra declares the training
packages; the site-specific PyTorch wheel may need to be installed separately before that extra.
Check the installed versions on the allocated node rather than assuming a module or wheel is
available. CUDA runs require one visible GPU. CPU is supported for a slow functional check.

From the repository root, run locally with:

```sh
uv run --extra train python scripts/cluster/smoke_train.py
```

For Slurm, first request an interactive allocation using the account, partition, and GPU resource
name configured at your site. For example, on a site that accepts this generic request:

```sh
srun --gres=gpu:1 --pty bash
cd /path/to/tactile-flora
uv run --extra train python scripts/cluster/smoke_train.py
```

The script defaults to 20 steps, `cuda`, and `data/cluster/smoke`. `--steps` must be at least 2
so the script can compare initial and final loss. Override the settings with `--steps`, `--device`,
and `--output`; for example:

```sh
uv run --extra train python scripts/cluster/smoke_train.py \
  --steps 30 --device cuda:0 --output /path/to/smoke-run
```

`--help` does not import PyTorch, Transformers, or PEFT. Each run writes `loss.csv`, `log.jsonl`,
`adapter/`, and `result.json` to the chosen output directory. A prior `result.json` is removed
before training so a failed rerun cannot appear successful based on stale metadata. The result
records the model scope,
seed, step count, float32 precision, device, package versions, GPU and runtime details where
available, memory use, and a structural check plus SHA-256 digest for the adapter weights.

## Existing recorded evidence

The workspace's `data/cluster/smoke/result.json` records a 20-step run on host
`zixian.nodes.scc.sdsc.edu` with an NVIDIA GeForce RTX 2080 Ti, PyTorch `2.14.1+cu126`, CUDA
`12.6`, and Transformers `5.8.1`. The recorded loss decreased from about `4.895` to `3.561`, and
peak allocated GPU memory was about 20.2 MB. The adapter README records PEFT `0.21.2`. This is
historical evidence present in the workspace; rerunning this script is required to produce the new
metadata fields and adapter integrity record. Neither result establishes acceptance of a released
model or validates a production training configuration.

A fresh Slurm run now exists at `data/acceptance/cluster-gpu-smoke/result.json`, with the same
20-step losses and a validated 40-tensor adapter. On this cluster, the successful request was:

```sh
srun --account=tritondroids --partition=debug --gres=rtx2080ti:1 \
  --mem=8G --cpus-per-task=2 --time=00:05:00 \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/smoke_train.py --output data/acceptance/cluster-gpu-smoke
```

The original interactive allocation's 1 GB host RAM caused an OOM kill even though this tiny
model needs only about 20 MB GPU memory. Request host RAM explicitly.

## Released inference gate

After the PHANTOM subset has been downloaded and verified (verification may report failed
source gates), prepare actual clips and the released processor without downloading weights:

```sh
uv run --extra train python scripts/cluster/smoke_inference.py --prepare-only
```

`--episode` selects a downloaded PHANTOM episode; the default is the first discovered id.
`--revision` selects a checkpoint revision, resolved to a commit SHA before downloads. Model
and Xet caches live under the output directory. This requires torchvision matching the installed
torch wheel, now included in `[train]`. Use the same wheel index for both packages.

Preparation tests dense and outcome prompts with omitted and real force, decodes real video,
and records actual processor token counts and video grids in `result.json`. It writes
`force.real-tokenizer.yaml`, usable with `tf verify --force-config PATH` and other pipeline
commands. A preparation result explicitly has `gate_passed: false`.

On the MI210s or fallback GPU, run the same command without `--prepare-only`. Full mode checks
free GPU memory against checkpoint bytes plus 20% headroom before downloading weights; this
is an initial capacity estimate, not a guarantee that inference fits. It then generates dense
and outcome responses for both conditions, saves raw outputs and parser verdicts, and only
passes if all four parse successfully. Failed or blocked runs return nonzero and replace any
old success record. This is inference input/output acceptance, not data acceptance or a score.

Preparation passed on revision `7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`, using torch
`2.14.1+cu126`, torchvision `0.29.1+cu126`, transformers `5.8.1`. Dense force uses 2,594 tokens;
outcome force uses 2,615, below the configured 3,000 budget. Real preparation uncovered and
fixed the unsupported `do_sample_frames` field inside video metadata; the option is passed
through processor kwargs instead.

As of 2026-10-21 both MI210 nodes are down. The RTX 2080 Ti preflight correctly reports
`blocked` (about 11.2 GB free, 55.6 GB weights), recorded separately under
`data/acceptance/released-gate-status`. Released generation has not passed.
