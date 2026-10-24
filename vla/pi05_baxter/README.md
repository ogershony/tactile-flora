# pi0.5 on Baxter: left-arm cube pick

Fine-tunes Physical Intelligence's pi0.5 (`lerobot/pi05_base`) on
[OG7407/baxter-cube-pick-left-v0](https://huggingface.co/datasets/OG7407/baxter-cube-pick-left-v0)
(18 kinesthetic demos, 1,622 frames at 10 Hz, head + left-wrist cameras, 7 joints + gripper) on one
24 GB GPU, and serves it to the Baxter runner over HTTP. Trained on 2026-10-24 on an RTX 5090
laptop GPU (95 W cap) with LeRobot 0.6.1.

Held-out results, step 5,000 (demos 5 and 14, 141 frames, open loop, one fixed noise seed):

| Metric | pi0.5 | Hold still |
|---|---:|---:|
| Mean joint error over the next 1.0 s | 0.030 rad | 0.042 rad |
| First-step joint error | 0.007 rad | |
| Joint error 1.0 s ahead | 0.050 rad | |
| Gripper error (0 to 1 scale) | 0.061 | 0.094 |
| Chunks starting > 0.25 rad from the arm | 0 of 141 | |
| Same check on 406 frames of the live ACT rollouts | 0 of 406 (mean largest first-step offset 0.031 rad) | |

Per-checkpoint numbers: `runs/pi05_baxter_r32_rel10/eval.jsonl` (every 2nd frame); held-out error goes
from 0.036 rad at step 500 to 0.029 rad at step 3,000 and stays flat to 5,000. Training took 3 h 23 min.
Figures: `../../writeup/hackathon/figures/`. Checkpoints (160 MB each) are not committed.

## What is different from a stock LeRobot fine-tune

1. **Relative chunks.** `--policy.use_relative_actions=true` with the gripper kept absolute: the model
   predicts each 10-step (1.0 s) chunk as offsets from the current joints. LeRobot normalizes actions
   with `meta/stats.json["action"]`, which describes absolute targets, so `scripts/make_local_dataset.py`
   writes a local copy of the dataset whose action statistics are recomputed on the relative chunks of the
   training episodes (demos 5 and 14 are held out). Motivation: 8 of the 9 failed ACT rollouts
   (OG7407/baxter-cube-pick-left-rollouts) were aborted because a chunk started more than 0.25 rad from
   the arm.
2. **Frozen VLM, LoRA on the action expert.** PEFT LoRA r32 (alpha 32) on q/k/v/o/gate/up/down of every
   layer of the 300M action expert plus `action_in_proj`, `action_out_proj`, `time_mlp_in`,
   `time_mlp_out`; PaliGemma (SigLIP + Gemma 2B) frozen. 14.1M trainable parameters; a checkpoint is
   160 MB (adapter + optimizer state).
3. **Frozen-prefix training step** (`scripts/frozen_prefix.py`, applied by `scripts/train_entry.py`).
   LeRobot's training forward runs the image/text prefix and the action suffix through one joint pass, so
   autograd backpropagates through the frozen 2B language model and batch 32 runs out of memory on 24 GB.
   The prefix never attends to the suffix, so its keys and values are computed once under `no_grad` (as
   `sample_actions` does at inference) and only `denoise_step` runs with gradients. `scripts/check_frozen_prefix.py`
   compares it with the stock pass on one batch: loss within 0.09%, cosine similarity of every LoRA
   gradient at least 0.994, peak memory 9.5 GiB instead of 15.1 GiB at batch 2.
4. **Two cameras, no padded third.** pi05_base declares a right-wrist camera; with it absent LeRobot feeds
   a masked blank image. Masked tokens are neither attended to nor counted in positions, so dropping that
   input (`assets/pi05_base_local/config.json`) is exactly equivalent and saves about a quarter of the
   prefix compute.
5. **Tokenizer.** The PaliGemma tokenizer on the Hub is gated; the identical SentencePiece model is public
   at `gs://big_vision/paligemma_tokenizer.model` (what openpi uses). `assets/paligemma_tokenizer/` wraps it
   for `AutoTokenizer`; ids match SentencePiece exactly.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "lerobot[pi,dataset]==0.6.1" peft sentencepiece protobuf
# base weights + local copy whose preprocessor points at the local tokenizer
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/pi05_base'))"
mkdir -p assets/paligemma_tokenizer
curl -o assets/paligemma_tokenizer/tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
# assets/pi05_base_local: config.json (right-wrist camera removed), policy_*processor.json (tokenizer path),
# model.safetensors -> symlink to the snapshot's model.safetensors
.venv/bin/python scripts/make_local_dataset.py --out data/baxter-cube-pick-left-v0-rel10
.venv/bin/python scripts/check_frozen_prefix.py          # optional equivalence check
scripts/train.sh pi05_baxter_r32_rel10 5000               # 2.4 s/step on a 95 W RTX 5090 laptop
.venv/bin/python scripts/eval_heldout.py runs/pi05_baxter_r32_rel10/checkpoints/*/pretrained_model \
    --out runs/pi05_baxter_r32_rel10/eval.jsonl --rollouts 3
```

## Run it on the robot

```bash
.venv/bin/python scripts/serve_pi05.py runs/pi05_baxter_r32_rel10/checkpoints/<best>/pretrained_model --port 8765
```

From the runner, `scripts/pi05_client.py`'s `query(url, head_rgb, wrist_rgb, joints8)` returns a (10, 8)
array of absolute joint targets, one per 0.1 s, starting at the current frame. Send the cameras as
recorded (640x360 RGB, head camera and left-hand camera) and the joints in the dataset's order
`left_s0 left_s1 left_e0 left_e1 left_w0 left_w1 left_w2 left_gripper`. Execute a prefix of each chunk
(5 steps is a reasonable start) and query again; keep the runner's own speed, workspace and jump limits on.

## Files

| File | Purpose |
|---|---|
| `scripts/train.sh` | the training command (all hyperparameters) |
| `scripts/train_entry.py`, `scripts/frozen_prefix.py` | lerobot-train with the frozen-prefix step |
| `scripts/check_frozen_prefix.py` | loss and gradient equivalence check against the stock forward |
| `scripts/make_local_dataset.py` | local dataset copy with relative-chunk action statistics |
| `scripts/eval_heldout.py` | open-loop held-out evaluation and live-rollout jump check |
| `scripts/serve_pi05.py`, `scripts/pi05_client.py` | HTTP policy server and client for the Baxter runner |
| `scripts/plot_results.py` | validation curve and held-out trajectory figures |
