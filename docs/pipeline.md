# Pipeline and command reference

The `tf` command line, the data flow between stages, the training loop, and the environment
facts that were verified on real hardware. Start with the [README](../README.md) for the
overview; this page is the reference. Every command accepts `--help` and `--config`, prints the
config path, the git hash and the output path before doing work, and writes only under `data/`.

## Choosing the torch wheel (ROCm or CUDA)

The verified MI210 environment uses a separate venv so CUDA and ROCm wheels stay independent:

```bash
uv venv --python 3.11 data/cluster/rocm-venv
uv pip install --python data/cluster/rocm-venv/bin/python \
  torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --python data/cluster/rocm-venv/bin/python -e '.[train]'
```

## Real-data and training acceptance

Download a bounded PHANTOM subset (one contact training success, one training failure,
and one validation success), then run verification separately from production artifacts:

```bash
uv run python scripts/download_phantom.py --max-bytes 250000000
uv run tf verify --config configs/sources/phantom.yaml --source phantom --data-root data/acceptance/phantom
```

The downloader resolves a commit revision, checks the byte limit before downloading, and writes
checksums and provenance to `data/raw/phantom/manifests/download.json`. Supply `--revision SHA`
to reproduce a subset. Three episodes check adapter compatibility; they do not establish benchmark
coverage. Verification reports pending spot review and any failed data gates without creating an
accepted training manifest.

Run the tiny random-model LoRA smoke on an allocated GPU with
`uv run --extra train python scripts/cluster/smoke_train.py`. It supports `--device cpu` for a
functional check. [Cluster smoke instructions](scripts/cluster/README.md) describe launch options,
recorded versions, and artifact validation. A fresh Slurm GPU run passed 20 steps and adapter
validation under `data/acceptance/cluster-gpu-smoke`.

Prepare real PHANTOM clips with the released processor and tokenizer before loading weights:

```sh
uv run --extra train python scripts/cluster/smoke_inference.py --prepare-only
```

This writes four input/token-count records and `force.real-tokenizer.yaml` under
`data/acceptance/released-inference`. Full released-model generation also passed on two MI210s:
dense and outcome prompts, with force omitted and included, all parse successfully.
On an allocated GPU, omit `--prepare-only` to repeat that gate. The successful ROCm command and
versions are in [the cluster instructions](scripts/cluster/README.md).

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

`--backend stub|hf|vllm` (default `hf`) applies to `pseudo`, `eval run` and `eval grid`; `hf` and `vllm` need the `[train]` extra and the released checkpoint. HF passed the one-episode inference gate on two MI210s; vllm remains untested. `stub` is a deterministic fake model used by the tests and for dry runs of the pipeline.

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
- The heavy dependencies are the `[train]` extra and are imported lazily. Install the torch and torchvision wheels for the allocated hardware as described in the quickstart.

## Environment facts

- Python 3.11 (`requires-python = ">=3.11,<3.12"`), uv, `uv.lock` committed.
- Released FLORA inference code requires vllm 0.24.0, transformers 5.8.1 and qwen-vl-utils 0.0.14 (from its `batch_utils/requirements.txt`). The verified HF path needs no vllm or qwen-vl-utils. `[train]` pins transformers 5.8.1, peft 0.21.2 and accelerate 1.15.0; the lock resolves tokenizers 0.22.2.
- Machines: 5x AMD MI210 (primary training and eval; ROCm risk: Gated DeltaNet kernels come from Triton-based flash-linear-attention, gfx90a is not the tested path), cloud RTX Pro 6000 96 GB (fallback if ROCm blocks milestone 7), 8x RTX 2080 (decode, verify, CI; never load the 27B), Baxter workstation (ROS 1, `[baxter]` extra).
- Exact versions that worked:
  - Two MI210s / ROCm: torch `2.12.1+rocm7.1`, HIP `7.1.52802`, torchvision `0.27.1+rocm7.1`, transformers `5.8.1`, Triton ROCm `3.7.1`; flash-linear-attention and causal-conv1d absent (native PyTorch fallback). Peak GPU allocations: 35.80 and 40.68 GiB.
  - Cloud CUDA: torch `TBD`, CUDA `TBD`, transformers `TBD`, flash-linear-attention `TBD`

## Gates

In order, from the spec:

1. Milestone 7: the released model produces a parsable dense output on one window on the MI210s (else the cloud GPU).
2. Every training source passes verification and `holdout_ids.json` is frozen before SFT starts.
3. Force-video sync is under 100 ms on a Baxter episode before any Baxter recording counts.
4. The released checkpoint's no-force rows on phantom_val and roboreward_500 exist before any trained-model row is read.

Sanity expectations when reading results: roboreward_500 within about 3 MAE points of 40.0 for released, control and force-omitted; the force model's failure recall must drop under `shuffled`; its `omitted` row must match the control within noise.
