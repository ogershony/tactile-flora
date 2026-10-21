# tactile-flora

Adds a fingertip-force channel to NVIDIA's FLORA-Rewarder (Qwen3.5-27B) by rendering force as text in the prompt, fine-tunes it with LoRA on public force datasets, and evaluates it on labeled tactile sets and on a Baxter robot.

Status: milestone 1 (skeleton and schema). Every stage entry point is a stub that raises `NotImplementedError("milestone N")`.

## Quickstart

```bash
uv python pin 3.11
uv sync --extra dev          # small default deps; add --extra train on GPU machines, --extra baxter on the robot workstation
uv run tf --help
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

`scripts/tf` is a thin wrapper that runs `uv run tf "$@"`. Every command takes `--config` and prints the config path, git hash and output path before doing work.

## Commands

- `tf verify --config configs/sources/phantom.yaml --source phantom`: run every check on a source and write its report under `data/verified/`.
- `tf manifest --config configs/sources/phantom.yaml`: turn passing episodes into manifest rows under `data/manifest/`.
- `tf manifest freeze-holdout --config configs/sources/phantom.yaml`: write `holdout_ids.json` once; frozen splits never change.
- `tf pseudo --config configs/sources/sharpadex.yaml --source sharpadex`: vision-only FLORA pseudo-labels (refuses eval-only sources).
- `tf train --config configs/train/force.yaml [--dry-run]`: LoRA SFT on the accepted manifest; `configs/train/control.yaml` is the matched no-force control.
- `tf eval run --config configs/eval/benchmarks.yaml --checkpoint released --benchmark phantom_val --force-mode omitted`: one eval cell.
- `tf eval grid --config configs/eval/grid.yaml`: expand the whole ablation grid, skipping finished cells.
- `tf eval report --config configs/eval/benchmarks.yaml`: markdown tables per benchmark.
- `tf baxter calibrate|record|label|to-lerobot --config configs/baxter.yaml`: Baxter capture stack.

## Training

`tf train --config configs/train/force.yaml [--dry-run] [--allow-stale-manifest]` runs LoRA SFT (rank 32, alpha 64 on the language model's attention and MLP projections; vision encoder and merger frozen; Gated DeltaNet parameters in fp32; FSDP with activation checkpointing when more than one GPU is visible, run under `torchrun`). `configs/train/control.yaml` differs only in `force_mode` (omitted).

- It refuses to start unless `data/manifest/holdout_ids.json` exists and every manifest row's `report_version` equals the current full git hash. `--allow-stale-manifest` skips only the second check and is for development; never use it for a reported run.
- `--dry-run` tokenizes the whole manifest with the Qwen3.5 processor, prints the token histogram (min/p50/p90/max/total) and the estimated wall-clock (9 s/window default), writes `dry_run.json`, and exits 0.
- A run writes `data/runs/<config stem>/` with `config.yaml`, `manifest_hash.txt`, `git_hash.txt`, `loss.csv`, `log.jsonl` (loss, lr, tok/s, GPU memory every 10 steps) and `adapter/`.
- The heavy dependencies are the `[train]` extra and are imported lazily. Choose the torch wheel index with the environment variable `TF_TORCH_INDEX` (an index URL, e.g. `https://download.pytorch.org/whl/rocm6.2` for the MI210s, `https://download.pytorch.org/whl/cu124` for the cloud GPU), for example `UV_EXTRA_INDEX_URL=$TF_TORCH_INDEX uv sync --extra train`. Record the versions that worked on each machine here once milestone 7 passes.
