# Baseline, fine-tuned and fine-tuned tactile FLORA on ten Baxter cube-pick rollouts

Date: 2026-10-24. Branch `hrdexdb-v3-force-representation`. Data:
[`OG7407/baxter-cube-pick-left-rollouts`](https://huggingface.co/datasets/OG7407/baxter-cube-pick-left-rollouts)
(LeRobot v3, 10 episodes). Everything on disk is under
`data/acceptance/baxter-rollouts-v1/` (copy of the dataset, labels, manifest, normalization,
representation audit, the six evaluation cells and `evaluation-result.json`; copies of the
result and audit JSON are in `baxter-rollouts/`). The per-rollout force figure is
`baxter-rollouts/force_traces.png`.

## Which models

The request named "tactile-v4". No adapter by that name exists on the Hub (the OG7407 account
holds only the ACT policy), on Drive, or on disk, and the `plan-v4` branch's own write-up
(`writeup/v4-results.md` there) records that the v4 force and control adapters were lost with
the Colab runtime on 2026-10-24; only their HRDexDB results survived. The evaluation therefore
uses the fine-tuned adapters that do exist on disk, which are the same three models compared in
`writeup/comparison.md`:

| Name in this write-up | Checkpoint | Inputs | Training |
|---|---|---|---|
| Baseline FLORA | released `nvidia/Flora-Rewarder` | 64 video frames (96 px cells) | none |
| Fine-tuned FLORA | `hrdexdb-expanded-v2/runs/expanded-v2-control` | same video | LoRA on 350 HRDexDB grasp clips, force omitted |
| Fine-tuned tactile FLORA | `hrdexdb-expanded-v3/runs/expanded-v3-force` | same video + text block of normalized fingertip force, summary line, force-only classifier probability | same recipe and clips, force shown |

Both adapters were trained on a dexterous Inspire hand in the HRDexDB rig, never on Baxter, a
parallel gripper, FSR pads, or a head camera. This is a zero-shot transfer test for everything
except the released model's general pretraining.

## The data

Ten live rollouts of the ACT policy `OG7407/baxter-act-cube-left` on the task "pick up the red
cube" (left arm, 15 fps head and wrist cameras, 200 Hz fingertip FSR counts on the left
gripper's two pads; the right gripper has no sensors and reads zero throughout). Labels: the
operator's statement that episodes 0 and 1 succeeded and the other eight failed, written to
`raw/labels.json` and used in place of the recorder's labels. The recorder agrees on nine of
ten; it had auto-labelled episode 1 a failure because the policy was stopped by a joint-jump
guard at 2.7 s.

| Episode | Operator label | Recorder | Length | What the sensors saw |
|---|---|---|---:|---|
| ep0000 | success | success (operator, "picked") | 5.6 s | gripper stops at 0.37 (cube between the pads), force rises to the grasp level at 2.7 s and holds for 2.9 s while the fingertip lifts 20 cm |
| ep0001 | success | fail (auto, joint-jump stop) | 2.7 s | gripper closing on the cube; force appears 0.03 s before the last frame, in one sampled frame only |
| ep0002-0004, 0008, 0009 | failure | fail (auto) | 6.8-20 s | gripper closes fully on nothing (closure 1.0), zero force, arm lifts empty |
| ep0005-0007 | failure | fail (auto) | 2.7-2.8 s | stopped while closing, zero force |

![Force and closure traces](baxter-rollouts/force_traces.png)

So the force channel carries one clean success signature (ep0000), one marginal one (ep0001,
a single contact frame at the very end), and eight clips with no contact at all. The video
side is harder than it sounds at 96 px: the head camera looks down the arm, the cube is a few
pixels wide, and in the long failures the arm lifts and moves exactly as in a success, just
with nothing between the fingers.

## How the force was shown to the tactile model

The pipeline is the v3 one, unchanged: the same `force.yaml`, tokenizer budget, summary line
and force-only classifier (`hrdexdb-expanded-v3/config/force-classifier.json`, fitted on
HRDexDB training clips). Two things are specific to this data and are recorded in
`representation-audit.json`:

- **Normalization constants were fitted on these ten rollouts** (there is no training split):
  baseline 184 counts (the two pads' resting 92 + 92), median grasp 892.5 counts, contact
  threshold 219 counts (the 5% floor of the grasp excess, since the free-space signal has zero
  variance). One normalized unit is the grasp force in ep0000.
- **The left gripper is presented as arm `R`** (`baxter_rollouts_arm: R`). The HRDexDB
  adapters only ever saw `pad R` lines with `L none`; showing the same letter keeps the text
  in-distribution. Gripper closure is `(0.96 - position) / 0.92`, so a close stopped by the cube
  reads 0.64 and an empty close 1.0.

The force-only classifier, which is printed into the block, already gives a preview of what
the force can and cannot do here (success probability per episode):

| ep0000 | ep0001 | ep0002 | ep0003 | ep0004 | ep0005 | ep0006 | ep0007 | ep0008 | ep0009 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.40** | 0.04 | 0.09 | 0.11 | 0.05 | 0.09 | 0.09 | 0.09 | 0.05 | 0.05 |

It ranks ep0000 first and every failure below it, but puts ep0001 last: with a 0.13 s hold it
looks like a touch, not a grasp. AUROC 0.50 (one success ranked first, one last), 8/10 correct
at the 0.5 cut because it calls everything a failure.

## Results

Six cells, job 6490 on two MI210s (8 min 12 s in total; the model loads in about 90 s from
the local cache and each clip takes 6 to 7 s). `parse_failed` and `no_logprobs` are 0 in every
cell. Raw outputs: `baxter-rollouts/evaluation-result.json`; the tables below are printed by
`baxter-rollouts/summarize.py` into `baxter-rollouts/tables.md`, which also lists every
rationale.

### Headline

Accuracy at the 4.0 threshold on E[s]; AUROC over the 2 successes x 8 failures pairs; MAE on the 0-100 scale.

| Model | Accuracy | AUROC | MAE | Successes right (of 2) | Failures right (of 8) | Hard scores given |
|---|---:|---:|---:|---:|---:|---|
| Baseline FLORA, no force | 0.80 | 0.375 | 140 | 0 | 8 | 2x10 |
| Baseline FLORA + force block | 0.80 | 1.000 | 100 | 0 | 8 | 1x4, 2x6 |
| Fine-tuned FLORA (video only) | 0.80 | 0.312 | 80 | 0 | 8 | 1x10 |
| Fine-tuned tactile FLORA, real force | 0.80 | 1.000 | 80 | 0 | 8 | 1x10 |
| Fine-tuned tactile FLORA, shuffled force | 0.80 | 0.250 | 90 | 0 | 8 | 1x9, 2x1 |
| Fine-tuned tactile FLORA, force removed | 0.80 | 0.438 | 80 | 0 | 8 | 1x10 |
| Force-only classifier (from the block) | 0.80 | 0.500 | | 0 | 8 | P(success) |

**The hard answers say nothing.** Every model gives 80% accuracy, and it is the majority-class
number: no cell scores either success at or above the 4.0 threshold. Baseline FLORA answers
"2" on all ten clips with and without force; both fine-tuned adapters answer "1" on all ten
(they were trained on HRDexDB's "no rationale / Score" template and are confidently
"failure" on Baxter footage). The comparison therefore lives in the expected score E[s], which
is what AUROC and MAE read.

**Ranking is where the models differ, and force decides it.** The two models that see the
force block rank both successes above all eight failures (AUROC 1.0). The two that see only
video rank below chance: fine-tuned FLORA puts ep0000 below five failures and ep0001 below
six; baseline FLORA puts ep0000 below six failures and ep0001 below four.

### Per-episode expected score

| Episode | Label | baseline | baseline+force | fine-tuned | tactile | tactile, shuffled | tactile, no force | force P(success) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ep0000 | success | 1.91 | 2.41 | 1.34 | 2.41 | 1.03 | 1.43 | 0.40 |
| ep0001 | success | 2.00 | 1.92 | 1.05 | 1.60 | 1.03 | 1.12 | 0.04 |
| ep0002 | failure | 2.01 ok | 1.78 ok | 1.40 ok | 1.05 ok | 1.05 ok | 1.52 ok | 0.09 |
| ep0003 | failure | 2.03 ok | 1.81 ok | 1.99 ok | 1.09 ok | 1.08 ok | 2.97 ok | 0.11 |
| ep0004 | failure | 1.91 ok | 1.06 ok | 1.60 ok | 1.02 ok | 1.02 ok | 1.32 ok | 0.05 |
| ep0005 | failure | 1.82 ok | 1.05 ok | 1.05 ok | 1.03 ok | 1.03 ok | 1.05 ok | 0.09 |
| ep0006 | failure | 1.99 ok | 1.00 ok | 1.05 ok | 1.02 ok | 1.62 ok | 1.11 ok | 0.09 |
| ep0007 | failure | 1.85 ok | 1.48 ok | 1.03 ok | 1.03 ok | 1.03 ok | 1.06 ok | 0.09 |
| ep0008 | failure | 2.04 ok | 1.70 ok | 1.75 ok | 1.04 ok | 1.06 ok | 2.31 ok | 0.05 |
| ep0009 | failure | 2.02 ok | 1.74 ok | 1.55 ok | 1.07 ok | 1.07 ok | 2.02 ok | 0.05 |

'ok' marks a correct call. Bold: tactile FLORA right where both baseline and fine-tuned FLORA are wrong.

Episodes where tactile FLORA alone is right: none.

### Where tactile FLORA beat both baseline and fine-tuned FLORA

There is no episode where tactile FLORA is right at the threshold and the other two are
wrong, because nobody crosses the threshold. In the ranking, which is the only signal these
ten clips give, the tactile model wins on every clip that matters:

- **ep0000 (success, the clean grasp).** Tactile E[s] 2.41, its top-scored clip. Fine-tuned
  FLORA gives it 1.34, sixth of ten (below ep0002, ep0003, ep0004, ep0008 and ep0009, all
  failures); baseline gives it 1.91, eighth of ten. The baseline's rationale for this clip is "the cube
  remained on the table in the final frames"; the cube was lifted 20 cm. At 96 px the video
  models do not see the lift, the force block does.
- **ep0001 (success, contact in the last frame only).** Tactile 1.60, second of ten, above
  every failure. Fine-tuned FLORA scores it 1.05, eighth of ten, below six failures; baseline
  2.00, fifth, below four failures. One contact line at the end of the block, with closure 0.64 (stopped by the
  cube) rather than 1.0 (closed on nothing), was enough to move it.
- **ep0003 and ep0008 (failures, long empty lifts).** These are the clips the video-only
  adapter likes most: 1.99 and 1.75, its two highest scores, because the arm closes, lifts
  and moves exactly as in a success. The tactile model gives them 1.09 and 1.04. With the
  force removed from the same adapter they jump to 2.97 and 2.31, the two highest scores in
  that cell.

Margin: the tactile model's lowest success (1.60) sits 0.51 above its highest failure (1.09).
Baseline FLORA with the force block also separates the classes, but by 0.11 (1.92 vs 1.81),
and its MAE is 100 against 80: it reads the block (its ep0001 rationale says "exerted a small
force") and lowers the failures, but it will not raise a success it believes it saw fail.

### Does the tactile model read the clip's own force?

Yes. Replacing each clip's block with another clip's (shuffled donors, which are mostly the flat
traces of the failures) drops AUROC from 1.0 to 0.25 and ep0000 from 2.41 to 1.03. Removing
the block entirely gives 0.44, with the empty-lift failures rising to the top. So the ranking
of the fine-tuned tactile model on this data is carried by the force text, not by anything it
learned about video, and the direction matches v3 on HRDexDB: force helps where the video is
ambiguous, and here the video is ambiguous on every clip.

### Caveats

- Ten clips, two positives. A perfect AUROC here is 16 of 16 ordered pairs; one swapped pair
  costs 0.06. No significance claims are made or possible.
- Zero-shot transfer: the adapters were trained on a 12-DoF Inspire hand with per-pad
  sensors in a 23-camera rig, evaluated on a Baxter parallel gripper with two FSR pads and a
  head camera at 15 fps, with the left gripper presented as arm `R`.
- 96 px thumbnails. The baseline's rationales show it cannot see the lift at this input size;
  the v4 branch found released FLORA much stronger at native resolution on HRDexDB, and the
  same would likely hold here.
- Normalization was fitted on the same ten clips (label-free, but still the test set).
  Episode 1's contact begins 0.03 s before its last frame, so its success rests on one line of
  the block.
- No threshold tuning was done; with two positives any tuned threshold would be fitted to
  the same clips it is scored on.

## Reproduce

Step by step for a fresh machine: `docs/handoff/baxter-rollouts.md`. In short:

```sh
# dataset copy, labels, normalization, manifest, audit (CPU)
uv run python scripts/cluster/prepare_baxter_rollouts.py --dataset <download> \
  --base data/acceptance/baxter-rollouts-v1 --success-episodes 0,1 --arm R --cameras head \
  --tokenizer-json <flora snapshot>/tokenizer.json
# six cells (released omitted/real, control omitted, force real/shuffled/omitted), ~76 GB GPU
uv run python scripts/cluster/evaluate_baxter_rollouts.py --base data/acceptance/baxter-rollouts-v1 \
  --model-id <flora snapshot> --control-adapter <expanded-v2-control>/adapter \
  --force-adapter <expanded-v3-force>/adapter --backend hf
```

The adapter is `tactile_flora/sources/baxter_rollouts.py` (registered as `baxter_rollouts`,
eval only); `--backend stub` on the evaluation script runs the whole pipeline on a CPU for
wiring checks. The force config and classifier are `configs/eval/baxter_rollouts/`.
