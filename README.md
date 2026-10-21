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
