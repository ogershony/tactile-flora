# Handoff: evaluate baseline, fine-tuned and fine-tuned tactile FLORA on the Baxter rollouts

Instructions for a Claude Code session on a machine with GPU compute. Everything below was
prepared and dry-run (stub backend, CPU) on the login node on 2026-10-24; only the GPU
inference step is left. Read this whole page before running anything.

## The task, as the user stated it

> Evaluate baseline FLORA, fine-tuned FLORA, and fine-tuned tactile FLORA on
> https://huggingface.co/datasets/OG7407/baxter-cube-pick-left-rollouts. Especially where
> tactile FLORA outperformed baseline and fine-tuned FLORA. Write it all up in a nice write-up.
> Success is on episodes 0 and 1.

The user first asked for "tactile-v4"; that adapter does not exist (lost with a Colab runtime,
see `writeup/v4-results.md` on branch `plan-v4`), and the user then said to use the v3 adapters.
So the three models are:

| Name | Checkpoint | Where it lives |
|---|---|---|
| Baseline FLORA | `nvidia/Flora-Rewarder`, revision `7de68d22c4a9decec5ddb7f86ddcaee8f82d3367` | Hugging Face Hub (public) |
| Fine-tuned FLORA (video-only control) | `expanded-v2-control/adapter` | SDSC login node, `/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v2/runs/expanded-v2-control/adapter` |
| Fine-tuned tactile FLORA (v3 force) | `expanded-v3-force/adapter` | SDSC login node, `/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v3/runs/expanded-v3-force/adapter` |

Each adapter is a 609 MB PEFT LoRA directory (`adapter_config.json`, `adapter_model.safetensors`).
The adapters are not in git and not on the Hub (an upload was attempted on 2026-10-24 and
blocked by permissions); the user copies them to this machine, or uploads them to private
repos under OG7407 and you download them with `huggingface-cli download <repo> --local-dir ...`.

## What already exists on the branch

Branch `hrdexdb-v3-force-representation` (pushed), commits 53903bf and later:

- `tactile_flora/sources/baxter_rollouts.py`: source adapter that reads the LeRobot v3 dataset
  directly (concatenated AV1 mp4 per camera addressed by in-file offset, native 200 Hz FSR
  counts from `raw/ep*/tactile_200hz.csv`, `labels.json` override). The left gripper's two pads
  are presented as arm `R` because the HRDexDB-trained adapters only ever saw `pad R` lines.
- `configs/eval/baxter_rollouts/force.yaml` and `force-classifier.json`: the v3 force config
  and the HRDexDB force-only classifier whose probability is printed in the block.
- `scripts/cluster/prepare_baxter_rollouts.py`: copies the dataset, writes the operator labels
  (episodes 0 and 1 success, all others failure), fits normalization on the ten rollouts,
  builds the outcome manifest and `representation-audit.json` (the exact force block per
  episode, summary line, classifier probability).
- `scripts/cluster/evaluate_baxter_rollouts.py`: six cells, each checkpoint in a fresh
  interpreter: released omitted + real, control omitted, force real + shuffled + omitted.
  Writes `<base>/evaluation-result.json` with metrics and per-episode predictions.
- `writeup/baxter-rollouts.md`: the write-up with data, method and figure sections done and a
  `RESULTS_PLACEHOLDER` line where the model results go. `writeup/baxter-rollouts/force_traces.png`
  is the per-rollout force figure.
- Tests: `tests/unit/test_sources_baxter_rollouts.py`; the full suite passed.

## Steps

All commands from the repo root. Python 3.11 and uv; the `[train]` extra needs torch for the
machine's accelerator (see README "Choosing the torch wheel"). Set `HF_TOKEN` (the user's token,
account OG7407) so the private adapter repos and the dataset resolve.

1. **Environment**

   ```sh
   git clone https://github.com/ogershony/tactile-flora.git && cd tactile-flora
   git checkout hrdexdb-v3-force-representation
   uv python pin 3.11 && uv sync --extra dev --extra train   # or a CUDA/ROCm venv per README
   uv run pytest -q tests/unit/test_sources_baxter_rollouts.py tests/unit/test_config.py
   ```

2. **Fetch model, adapters, dataset** (about 55 GB for the base model, 1.2 GB adapters, 11 MB data)

   ```sh
   export HF_TOKEN=...
   uv run huggingface-cli download nvidia/Flora-Rewarder --revision 7de68d22c4a9decec5ddb7f86ddcaee8f82d3367 --local-dir data/models/flora-rewarder
   uv run huggingface-cli download OG7407/baxter-cube-pick-left-rollouts --repo-type dataset --local-dir data/raw/baxter_rollouts_download
   # adapters: from the SDSC login node (paths in the table above), e.g.
   mkdir -p data/adapters/expanded-v2-control data/adapters/expanded-v3-force
   scp -r <sdsc-login>:/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v2/runs/expanded-v2-control/adapter data/adapters/expanded-v2-control/
   scp -r <sdsc-login>:/home/ogershony/tactile-flora/data/acceptance/hrdexdb-expanded-v3/runs/expanded-v3-force/adapter data/adapters/expanded-v3-force/
   ```

   The adapter directory names matter: the harness names each cell after the directory above
   `adapter` (`expanded-v2-control`, `expanded-v3-force`), and the write-up uses those names.

3. **Prepare the benchmark** (CPU, under a minute)

   ```sh
   uv run python scripts/cluster/prepare_baxter_rollouts.py \
     --dataset data/raw/baxter_rollouts_download \
     --base data/acceptance/baxter-rollouts-v1 \
     --success-episodes 0,1 \
     --label-note "operator statement 2026-10-24: episodes 0 and 1 succeeded; all other rollouts failed" \
     --arm R --cameras head \
     --dataset-id OG7407/baxter-cube-pick-left-rollouts \
     --tokenizer-json data/models/flora-rewarder/tokenizer.json
   ```

   Expected output: 10 episodes; normalization baseline 184, grasp 892.5, tau 219.425,
   scale 708.5; labels ep0000 and ep0001 = 5, the rest = 1; classifier P(success) 0.40 for
   ep0000, 0.04 for ep0001, 0.05 to 0.11 for the failures. If the numbers differ, stop and
   compare `representation-audit.json` against this page before spending GPU time.

4. **Wiring check without a GPU** (optional, 10 s): append `--backend stub` to the command in
   step 5, confirm six `metrics.json` files appear, then delete `data/acceptance/baxter-rollouts-v1/eval`
   and `evaluation-result.json` before the real run.

5. **Run the six cells** (two MI210s or one 80 GB class GPU; the 27B model in bf16 with fp32
   DeltaNet parameters needs about 76 GB total; `device_map="auto"` shards across visible GPUs).
   Each checkpoint loads once (about 9 min on MI210) and scores 10 clips; expect under an hour.

   ```sh
   export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
   uv run python -u scripts/cluster/evaluate_baxter_rollouts.py \
     --base data/acceptance/baxter-rollouts-v1 \
     --model-id data/models/flora-rewarder \
     --control-adapter data/adapters/expanded-v2-control/adapter \
     --force-adapter data/adapters/expanded-v3-force/adapter \
     --released-modes omitted,real --force-modes real,shuffled,omitted --backend hf
   ```

   Output: `data/acceptance/baxter-rollouts-v1/eval/<tag>/baxter_rollouts/<mode>/{predictions.parquet,metrics.json,log.jsonl}`
   and `data/acceptance/baxter-rollouts-v1/evaluation-result.json`. Check `parse_failed` and
   `no_logprobs` are 0 in every `metrics.json`. If a cell crashes, rerun with `--worker N`
   (0 released, 1 control, 2 force) after fixing the cause; completed cells are overwritten.

6. **Fill in the write-up.** Replace `RESULTS_PLACEHOLDER` in `writeup/baxter-rollouts.md` with:
   - a headline table (one row per cell: accuracy at the 4.0 threshold, AUROC, MAE on the
     0-100 scale, how many of the 2 successes and 8 failures each model got right);
   - a per-episode table of expected score E[s] for every cell (10 rows x 6 columns), bold the
     cells where the tactile model is right and both baseline and fine-tuned FLORA are wrong,
     and name those episodes explicitly: that is what the user asked for ("especially where
     tactile FLORA outperformed baseline and fine-tuned FLORA");
   - the ablation reading: real vs shuffled vs omitted force on the tactile adapter says
     whether the block was used at all on this data;
   - honest caveats: 10 clips and 2 positives, so no significance claims; zero-shot transfer
     from a dexterous hand to a parallel gripper; 96 px head-camera thumbnails; episode 1's
     contact begins 0.03 s before its last frame; normalization fitted on the test clips
     themselves (unavoidable without a Baxter training set, and label-free).
   Keep the write-up style of `writeup/comparison.md`: plain sentences, tables, no hype.
   Also update the memory/notes line in `docs/status.md` if the repo convention asks for it.

7. **Commit and push** to `hrdexdb-v3-force-representation`: the write-up, the
   `evaluation-result.json` copied to `writeup/baxter-rollouts/evaluation-result.json` (the
   `data/acceptance` tree is gitignored), and nothing else from `data/`.

## Facts that save time

- Only episode 0 shows a clean grasp in the force; episode 1's contact is a single sampled
  frame at the very end; the eight failures closed on nothing (closure 1.0, force 0).
- The dataset's right-gripper channels `R_a`, `R_b` are always 0 (no sensors); the adapter
  ignores them. Do not "fix" this.
- `labels.json` beats the recorder's `meta.json` label; the recorder calls episode 1 a failure
  (joint-jump guard stop at 2.7 s), the user calls it a success. Keep the user's label and say
  so in the write-up.
- The benchmark is `eval_only`; `tf train` must never see it.
- Never submit to the `normal` Slurm partition on the SDSC cluster node `matthew` while another
  user's `long` job runs there: it preempts them. That is why the run moved to this machine.
- Do not change `tactile_flora/config.py` while an evaluation is running (the harness compares
  saved configs to the live schema).
