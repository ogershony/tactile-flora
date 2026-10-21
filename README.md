# tactile-flora

Adds a fingertip-force channel to NVIDIA's FLORA-Rewarder (Qwen3.5-27B) by rendering force as text in the prompt, fine-tunes it with LoRA on public force datasets (PHANTOM/HapticWAM, SharpaDex, RH20T, REASSEMBLE, AIRoA MoMa), and evaluates it on labeled tactile sets (HRDexDB, SoftVTBench, a RoboReward subsample) and on a Baxter robot with home-built FSR fingertips. A matched no-force control run and a set of force ablations (omitted, zero, real, shuffled, video_blank) test whether the model actually reads the channel.

Specification: `docs/spec.md`. Deviations from it: `docs/deviations.md`. Run status and owner to-do list: `docs/status.md`. Dataset and prompt research: `docs/research/`.

## Quickstart

```bash
uv python pin 3.11
uv sync --extra dev          # default deps are small; --extra train on GPU machines, --extra baxter on the robot workstation
uv run pytest -q             # CPU only, no datasets or GPU needed (fixtures are committed)
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run tf --help
```

Extras: `[train]` torch, transformers, peft, accelerate, tokenizers; `[baxter]` pyserial, rospkg; `[dev]` pytest, ruff, pyright, wandb, type stubs. Python is pinned to 3.11 and `uv.lock` pins every dependency. `scripts/tf` is a thin wrapper around `uv run tf "$@"`.

Every command takes `--config`, and prints the config path, the git hash and the output path before doing work. No command writes outside `data/` (Baxter capture writes to its configured `output_dir`).

### Choosing the torch wheel (ROCm or CUDA)

Set `TF_TORCH_INDEX` to a PyTorch wheel index URL and pass it to uv:

```bash
export TF_TORCH_INDEX=https://download.pytorch.org/whl/rocm6.2   # MI210s (gfx90a)
# export TF_TORCH_INDEX=https://download.pytorch.org/whl/cu124   # cloud RTX Pro 6000
UV_EXTRA_INDEX_URL=$TF_TORCH_INDEX uv sync --extra train
```

## Commands

One line per top-level command. All of them accept `--help`.

- `tf verify --config configs/sources/phantom.yaml --source phantom`: run every check on a source and write `data/verified/phantom/{report.md,checks.parquet,norm.json,hashes.parquet}`. `--spot-pass N` records the reviewer's spot-check pass count.
- `tf manifest --config configs/sources/phantom.yaml --name train`: turn passing episodes into manifest rows in `data/manifest/<name>.parquet`.
- `tf manifest freeze-holdout --config configs/sources/phantom.yaml`: write `data/manifest/holdout_ids.json` once; frozen splits never change afterwards.
- `tf pseudo --config configs/sources/sharpadex.yaml --source sharpadex --backend stub --limit 5`: vision-only FLORA pseudo-labels into `data/verified/<source>/pseudo.parquet` (refuses eval-only sources and held-out ids).
- `tf train --config configs/train/force.yaml --dry-run`: LoRA SFT on the accepted manifest; `--dry-run` only tokenizes and estimates. `configs/train/control.yaml` is the matched no-force control.
- `tf eval run --config configs/eval/benchmarks.yaml --checkpoint released --benchmark phantom_val --force-mode omitted --backend stub`: one checkpoint x benchmark x force-mode cell into `data/eval/<tag>/<benchmark>/<mode>/`.
- `tf eval grid --config configs/eval/grid.yaml`: expand the whole ablation grid, skipping finished cells.
- `tf eval report --config configs/eval/benchmarks.yaml`: markdown tables per benchmark (refuses to compare predictions with different manifest hashes).
- `tf eval make-roboreward --config configs/eval/benchmarks.yaml --metadata RoboReward/test/metadata.jsonl --n 500 --seed 0`: stratified roboreward_500 manifest plus windows sidecar (outcome only, no force).
- `tf baxter calibrate --config configs/baxter.yaml`: fit per-pad weight-to-newton curves from known weights.
- `tf baxter record --config configs/baxter.yaml --notes "mug, 2 N"`: record one episode directory; refuses on flat or saturated channels.
- `tf baxter label --config configs/baxter.yaml --labeler alice`: label recorded episodes from video only (force is never shown); `--merge alice --merge bob` merges two labelers.
- `tf baxter to-lerobot --config configs/baxter.yaml`: convert episode directories to a LeRobot v3 dataset, ready for `tf verify --source baxter`.

`--backend stub|hf|vllm` (default `hf`) applies to `pseudo`, `eval run` and `eval grid`; `hf` and `vllm` need the `[train]` extra and the released checkpoint, and are untested (see status below). `stub` is a deterministic fake model used by the tests and for dry runs of the pipeline.

## Data flow

Stages talk only through files under `data/`. Training and eval never read raw datasets directly; they read the accepted manifest (eval rebuilds windows through the registered source adapters from manifest ids and anchors). All schemas are pydantic models in `tactile_flora/schema.py` (and `tactile_flora/config.py` for configs).

```
raw dataset (data/raw/<source>, per-dataset layout, read only by tactile_flora/sources/*)
  | tf verify
  v
data/verified/<source>/   report.md, checks.parquet (CheckResult rows), norm.json (per-source
  |                       normalization constants, hashed), hashes.parquet (perceptual frame hashes),
  |                       spot/ (spot-check mosaics), pseudo.parquet (tf pseudo; optional)
  |                       data/verified/benchmark_hashes.parquet (overlap index against eval sets)
  | tf manifest, tf manifest freeze-holdout
  v
data/manifest/            <name>.parquet (one ManifestRow per accepted Window), <name>.shifts.json
  |                       (raw sync lags), holdout_ids.json (HoldoutFile, frozen once)
  | tf train
  v
data/runs/<config stem>/  config.yaml, manifest_hash.txt, git_hash.txt, loss.csv, log.jsonl,
  |                       dry_run.json (dry run only), adapter/
  | tf eval run / grid
  v
data/eval/<tag>/<benchmark>/<mode>/   predictions.parquet, metrics.json, config.json, log.jsonl
  | tf eval report
  v
markdown tables (conditions as rows, shuffle-delta column)
```

Key schemas in `tactile_flora/schema.py`: `Episode` (video refs, `ForceStream` of pad or wrist arms, `GripperStream`, labels, source meta), `Window`, `Label` and `Span` (dense or outcome, with origin human > source > programmatic > pseudo), `ManifestRow`, `CheckResult`, `HoldoutFile`, `ResampledForce`. Timestamps are float seconds on the episode clock (zero at the first video frame). Every artifact carries the git hash that produced it; the manifest rows carry `report_version` (full git hash) and `norm_version` (hash of the norm constants).

## Training

`tf train --config configs/train/force.yaml [--dry-run] [--allow-stale-manifest]` runs LoRA SFT (rank 32, alpha 64 on the language model's attention and MLP projections; vision encoder and merger frozen; Gated DeltaNet parameters in fp32; FSDP with activation checkpointing when more than one GPU is visible, run under `torchrun`). `configs/train/control.yaml` differs only in `force_mode` (omitted); a unit test enforces that.

- It refuses to start unless `data/manifest/holdout_ids.json` exists and every manifest row's `report_version` equals the current full git hash. `--allow-stale-manifest` skips only the second check and is for development; never use it for a reported run.
- `--dry-run` tokenizes the whole manifest with the Qwen3.5 processor, prints the token histogram (min/p50/p90/max/total) and the estimated wall-clock (9 s/window default), writes `dry_run.json`, and exits 0.
- A run writes `data/runs/<config stem>/` with `config.yaml`, `manifest_hash.txt`, `git_hash.txt`, `loss.csv`, `log.jsonl` (loss, lr, tok/s, GPU memory every 10 steps) and `adapter/`.
- The heavy dependencies are the `[train]` extra and are imported lazily. Choose the torch wheel index with `TF_TORCH_INDEX` as described in the quickstart.

## Milestone status

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Skeleton and schema | Built and tested on fixtures |
| 2 | Force channel (resample, normalize, render, ablations) | Built and tested on fixtures |
| 3 | Prompts, parsers, window geometry | Built and tested on fixtures |
| 4 | Labels (mapping, events, merge) | Built and tested on fixtures |
| 5 | Source adapters (5 training, 3 eval-only) | Built and tested on fixtures; no adapter has run on a real episode |
| 6 | Verification and manifest | Built and tested on fixtures |
| 7 | Released-model inference | Code complete; never run against the real checkpoint or a GPU. This is the ROCm gate |
| 8 | Pseudo-labeling | Code complete; only the stub backend has run |
| 9 | Training | Code complete; dry run and a smoke run exist on fixtures only, no real GPU run |
| 10 | Eval harness | Code complete; only the stub backend has run. roboreward_500 and phantom_val numbers do not exist yet |
| 11 | Baxter stack | Code complete; untested on hardware. The FSR part (ASIN B0CZ6L5NMM) is unconfirmed, so `sensor_confirmed: false` in `configs/baxter.yaml` |

"Tested on fixtures" means synthetic episodes under `tests/fixtures/` laid out like the real datasets. See `docs/status.md` for the test count and the full list of what is unverified.

## Environment facts

- Python 3.11 (`requires-python = ">=3.11,<3.12"`), uv, `uv.lock` committed.
- Released FLORA inference code requires vllm 0.24.0, transformers 5.8.1 and qwen-vl-utils 0.0.14 (from its `batch_utils/requirements.txt`). `pyproject.toml` does not yet pin transformers or vllm for the `[train]` extra; reconcile at milestone 7.
- Machines: 5x AMD MI210 (primary training and eval; ROCm risk: Gated DeltaNet kernels come from Triton-based flash-linear-attention, gfx90a is not the tested path), cloud RTX Pro 6000 96 GB (fallback if ROCm blocks milestone 7), 8x RTX 2080 (decode, verify, CI; never load the 27B), Baxter workstation (ROS 1, `[baxter]` extra).
- Exact versions that worked, to be filled in after milestone 7 passes:
  - MI210 / ROCm: torch `TBD`, ROCm `TBD`, transformers `TBD`, flash-linear-attention `TBD`, triton `TBD`
  - Cloud CUDA: torch `TBD`, CUDA `TBD`, transformers `TBD`, flash-linear-attention `TBD`

## Gates

In order, from the spec:

1. Milestone 7: the released model produces a parsable dense output on one window on the MI210s (else the cloud GPU).
2. Every training source passes verification and `holdout_ids.json` is frozen before SFT starts.
3. Force-video sync is under 100 ms on a Baxter episode before any Baxter recording counts.
4. The released checkpoint's no-force rows on phantom_val and roboreward_500 exist before any trained-model row is read.

Sanity expectations when reading results: roboreward_500 within about 3 MAE points of 40.0 for released, control and force-omitted; the force model's failure recall must drop under `shuffled`; its `omitted` row must match the control within noise.

## Layout

```
configs/        sources/*.yaml, force.yaml, train/, eval/, baxter.yaml (pydantic-validated, extra keys forbidden)
tactile_flora/  schema, config, cli; sources/ force/ labels/ verify/ prompts/ train/ eval/ baxter/ video/
tests/          unit/ (per module), fixtures/ (synthetic episodes per source), test_pipeline_smoke.py
docs/           spec.md, deviations.md, status.md, review1.md, research/
```
