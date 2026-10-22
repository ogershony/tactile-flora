# v4 plan: a stronger force-over-video result within one day

Goal: turn the v3 result (tactile +2.8 accuracy points, +0.043 AUROC over the fine-tuned
video-only model, only the AUROC gain significant) into one that is significant on accuracy too,
or know by tomorrow evening that this dataset cannot deliver it.

## Why v3 stalled, in one paragraph

Three things cap v3. The force signal is moderate: every force-only model lands at AUROC 0.76 to
0.82, and video alone is already at 0.83 to 0.90. The training set is tiny: 350 clips, 88
gradient updates, loss at its floor by update 20. And the held-out set is small: 251 clips over
32 objects cannot certify a 3-point gain (about 1,000 would be needed at the observed win/loss
ratio). The first cap is physics and stays. The second and third are both data volume, and the
data exists.

## Finding that changes the plan

Measured today on the existing outputs, no training:

| Test split (115 clips) | Accuracy | AUROC |
|---|---:|---:|
| fine-tuned FLORA, video only | 69.6% | 0.733 |
| fine-tuned tactile FLORA (v3) | 72.2% | 0.800 |
| force-only gradient-boosted classifier (no video, 350 training clips) | 76.5% | 0.777 |
| logistic stack of v3 score + control score + force classifier, fit on dev | 77.4% | 0.800 |

On the harder test objects the small force classifier beats both fine-tuned VLMs on accuracy,
and v3 does not fully exploit it even though its probability is printed in the force block.
The VLM is the bottleneck, not the force. More training clips attack exactly that, and a
post-hoc stack recovers part of it for free.

## Track A: double the data with the second Inspire hand (the main bet)

HRDexDB's `inspire_f1` hand was unusable until today: its tactile file is a pickled object
array. It is now decoded (restricted unpickler, per-pad normal and tangential force, 8 pads,
50 Hz; see `tactile_flora/sources/hrdexdb.py`), the two video-clock defects every f1 trial has
are handled, and it loads end to end. It adds 587 labeled clips (164 failures) on 92 objects,
77 of them shared with the hand already used, same scene camera.

Force-only separability on a 48-clip sample matches dftp (AUROC 0.79 to 0.82). Half of the f1
failures are no-contact misses, which force catches trivially and video may also catch; the f1
shear channel is the part video cannot see.

Steps, with the engineering risk flagged:

1. **Download** (1 to 2 h wall-clock, mostly waiting). Extend
   `scripts/cluster/download_hrdexdb_expanded.py`: per-embodiment file list (`right_tactile`,
   `right_joint_states`, their `_time` files, timestamps, `vid/22641005.mp4`), include the
   `inspire_f1` catalog rows, about 15 GB. Set `HF_TOKEN` to avoid the unauthenticated rate
   limit.
2. **Splits across both hands.** Objects shared between hands must land in the same split, or
   the test set leaks object identity through the other hand. Keep the frozen v2 object
   assignment for the 77 shared objects, hash the 15 f1-only objects into train/dev/test with
   the same rule. Expect roughly 680 train, 270 dev, 230 test clips.
3. **Normalization per hand.** The tared representation divides by the median grasp force, which
   the current `norm.json` holds once per source. f1 and dftp differ in scale and resting offset,
   so the constants must be fitted per session (hand). This is the one place the pipeline may
   need real code, not configuration: check whether `normalize.fit` can key on
   `Episode.session`; if not, fit two constant sets and select by session in
   `window_features`. Half a day of work at most; do it first.
4. **Refit the force classifier** on the new training set (existing `fit_cv`), out-of-fold
   probabilities for training clips as in v3.
5. **Train.** Same recipe as v3 (LoRA r32, lr 2e-5, 4 epochs, seed 17). Measured throughput is
   11.6 s per clip-epoch, so 680 clips x 4 epochs is about 8.8 h per arm. That fits the 12 h
   `long` partition only as two separate jobs: force arm first, control second. Do not submit
   anything to the `normal` partition while they run: on matthew `normal` preempts `long`
   (this killed the v3 control today). Chain evals with `afterok` on the `long` partition too.
6. **Evaluate.** Dev and test, control omitted and force real first (about 3 h for both
   splits); shuffled and blank-video ablations afterwards. Reuse the released-model cells for
   dftp clips; only f1 clips need new released cells.

Timeline if the download starts this evening:

| When | What |
|---|---|
| tonight, +0 to +3 h | download script, splits, per-hand norm, classifier refit, manifests, dry run |
| +3 h | force SFT starts (about 9 h) |
| +12 h | control SFT starts (about 9 h) |
| +21 h | dev + test eval, control and force real (about 3 h) |
| +24 h | numbers in hand; ablations queued behind |

Expected outcome, stated before running: with about 500 held-out clips the AUROC gain should be
clearly significant and the accuracy gain should reach p around 0.1 to 0.15 if it stays at
3 points. A larger accuracy gain is plausible because v3's gain is concentrated on uncertain
clips and the f1 failures are more force-separable. If the gain shrinks with more data, the
3-point v3 margin was partly noise, and that is also worth knowing.

## Track B: squeeze the current models while the GPU is busy (CPU only, today)

These are legitimate because dev is held out from training, and they need no retraining.

- **Stack v3 with the control and the force classifier.** Logistic regression on three numbers
  fit on dev lifts test accuracy from 72.2% to 77.4%. Report it as a separate line, not as the
  VLM's number.
- **Dev-tuned threshold.** 4.0 is the task's threshold, but the fine-tuned model scores low;
  3.15 chosen on dev gives +1.7 points on test. Minor, free.
- **Error review of the 33 clips both models get wrong.** They cluster on a few objects
  (icecream_scoop, plant_pot, organizer_beige, brass_pot). Watch them with the force trace: if
  the label is debatable (dropped after 9 s, lifted but not held) the ceiling is label noise and
  no model will fix it.

## Track C: recipe changes to fold into v4 (cheap, decide before launch)

- **Keep 4 epochs and the v3 recipe** so v4 differs from v3 in data only; otherwise a gain
  cannot be attributed.
- **Expose f1 tangential force** as a third channel or in the summary line. This is the one
  signal video cannot see. It needs a schema change (`PadArmForce` holds two channels), so it is
  a v5 item unless step 3 above finishes early.
- **Seeds, not epochs.** If GPU time remains after v4, a second seed of each arm tells how much
  of the margin is training noise. More epochs on the same clips will not help: the loss is
  already at the memorization floor.

## What to decide now

1. Spend the day on Track A (recommended), or run two more seeds of v3 instead (cheaper, but
   it can only tighten the error bar, not move the mean).
2. Whether the control must be retrained on the expanded data (yes, for a fair benchmark and
   because it also measures what more video alone buys).
3. Whether to set `HF_TOKEN` for the download; without it the 15 GB pull may take several hours.

## Blockers I cannot clear alone

- `scancel` and job submission on the `normal` partition during training are user actions here.
- The dead v3 chain (jobs 6340 to 6343) still needs cancelling.
- `jobs/eval-train.sbatch` passes a `--skip-released` flag the eval script does not define; fix
  before reusing the train-set eval.
