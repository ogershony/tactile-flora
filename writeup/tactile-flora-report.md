# Tactile FLORA: adding fingertip force to a vision-language reward model

Experiment report. Date 2026-10-24. Branch `hrdexdb-v3-force-representation`. Authors: Oren
Gershony and collaborators (Triton Droids / SDSC). Companion files: `comparison.md` (HRDexDB
results and the 26 win videos), `baxter-rollouts.md` (the Baxter robot evaluation and its videos),
`paper/` (NeurIPS-style manuscript) and `slides/` (deck). Everything in this report is read from
finished evaluation cells on disk; no number was produced for this document by re-running a model.

## 1. Summary

**Claim.** A FLORA-Rewarder fine-tuned to read a text rendering of fingertip force ("tactile
FLORA") judges grasp outcomes better than the same model fine-tuned on video alone, and far better
than the released vision-only model. The gain is in ranking successes above failures, it survives
the two controls that matter (a matched no-force fine-tune and a shuffled-force ablation), and it
transfers zero-shot from the dexterous hand it was trained on to a parallel-gripper robot in our
lab.

**Evidence, in three lines.**

| Setting | Released FLORA | Fine-tuned FLORA (video only) | Fine-tuned tactile FLORA |
|---|---:|---:|---:|
| HRDexDB, 251 held-out clips over 32 unseen objects: accuracy / AUROC | 66.5% / 0.600 | 76.5% / 0.830 | **79.3% / 0.873** |
| HRDexDB dev (136): accuracy / AUROC / MAE | 66.2% / 0.614 / 150 | 82.4% / 0.899 / 71 | **85.3% / 0.926 / 53** |
| Baxter cube-pick rollouts, 10 clips, 2 successes: AUROC | 0.375 | 0.313 | **1.000** |

- Tactile minus video-only fine-tune, pooled over the 251 HRDexDB clips: +2.8 accuracy points
  and **+0.043 AUROC, object-level bootstrap 95% interval +0.001 to +0.080, p about 0.02**.
- The tactile model reads the force: giving it another clip's force block drops dev accuracy
  from 85.3% to 73.5% (it loses 25 paired clips and wins 9, p = 0.009); blanking the video leaves
  AUROC 0.775, so neither channel alone explains the result.
- The gain concentrates where video is ambiguous: on the 101 clips where the video-only model
  sits within 0.75 of the decision threshold, tactile FLORA is 6 points more accurate (72.3% vs
  66.3%); on the 150 confident clips, under 1 point.
- On the Baxter robot, with no Baxter training data, the force block takes the ranking of the
  two successes from below chance to perfect for both the released model (0.375 to 1.0) and the
  fine-tuned adapter (0.313 to 1.0); shuffled force gives 0.25, force removed 0.44.

**Honest framing.** The HRDexDB accuracy gain (+2.8 points) is not individually significant
(McNemar p = 0.37); the ranking gain is. The Baxter set has ten clips and two successes, so it
is a direction, not a significance result. Both held-out sets are object-disjoint from training
and were scored once with fixed settings.

## 2. What was built

### 2.1 The model

FLORA-Rewarder (NVIDIA GEAR, October 2026) is a Qwen3.5-27B vision-language model that takes a
multi-camera video mosaic and a task instruction and answers "did the task succeed?" with a score
from 1 to 5. It is vision-only. We keep the released weights, the released prompts and the released
sampling of 64 frames per clip, and add one thing: a text block describing the fingertip force,
placed after the video in the user prompt, preceded by a one-sentence explanation of its columns.
The model is fine-tuned with LoRA (rank 32, alpha 64, all attention and MLP projections) on
the outcome task; the vision encoder is frozen. The continuous score we evaluate is the
expected score E[s] over the digits 1 to 5 at the answer token, with 4.0 as the success threshold.

### 2.2 The force block

The raw sensor stream is resampled to the 64 sampled frame times, normalized per source so that 0
is free space and 1 is the median grasp force of the training set, and rendered as one line per
frame plus a summary line. For the HRDexDB Inspire hand, pads are tared per trial (median of the
first second subtracted, floored at zero) and the two most loaded pads are shown. An example block
for one HRDexDB training clip, abbreviated:

```
Force (normalized: 0 free space, 1 median grasp, max 3):
summary R: contacts=1 longest=6.00s total=6.00s peak=1.83 hold=5.67s holdF=1.55 late/early=0.70 low=0% force-only classifier P(success)=0.76
t=0.00 L none | pad R F=0.00 D=0.00 dF=0.00 g=0.00 c=0
t=0.25 L none | pad R F=0.00 D=0.00 dF=0.00 g=0.00 c=0
...
t=12.50 L none | pad R F=1.62 D=0.21 dF=0.03 g=0.71 c=1
```

`F` is total grip force, `D` the pad asymmetry, `dF` its rate, `g` the hand closure (0 open, 1
closed), `c` a contact flag (force above a threshold set at 5% of the grasp force). The summary line
is computed label-free from the same normalized features: number of contact segments, longest and
total contact, peak force, hold duration and hold force, the late-over-early hold ratio, the
fraction of low-force hold, and the probability of success from a small gradient-boosted
classifier over 31 trial-level force statistics (120 trees, depth 2), fitted on the 350 training
clips only. Training clips see the out-of-fold probability of a model that never saw them, so the
VLM learns how far to trust the line rather than over-trusting in-sample predictions. The block for
a clip is capped at 3,000 tokens (largest observed: 2,700).

### 2.3 Controls

Every claim is backed by two controls trained or run on identical data:

- **Matched no-force fine-tune ("fine-tuned FLORA").** Same base, same 350 clips, same seed,
  epochs, learning rate, batch and LoRA, with the force block omitted. Any gain over it is a
  force gain, not a fine-tuning gain.
- **Shuffled force.** The tactile model evaluated with each clip's block replaced by another
  clip's. A gain that survives shuffling is not a force gain.

Plus two diagnostics: **force removed** (the tactile adapter with no block) and **video blanked**
(frames mid-gray, force only), which show whether the model reads the block at all and what the
force carries on its own.

## 3. Experiment process

The work went through four stages. Each stage changed one thing the previous one showed was wrong,
and every stage's artefacts are on disk under `data/acceptance/` with frozen plans, hashes and
Slurm job ids.

### 3.1 Pilot on HRDexDB (32 train / 16 validation clips)

HRDexDB (SNU VCLab, 2026) records a 12-DoF Inspire DFTP hand on a UR arm grasping household
objects in a 23-camera rig, with per-finger force at 50 Hz and a success flag per trial
(object lifted and held). We used one fixed scene camera and the released success labels. The
first LoRA runs on 32 clips showed no force benefit (released FLORA with force 62.5% vs a
signed-force adapter 56.25% on 16 clips). A data-quality audit found the real limits: the sample
was tiny, labels were plumbing-verified but not independently reviewed, and the force and video
clocks had not been aligned (the sensor stream starts 2.2 to 2.3 s after the video and the two
clocks drift). The audit's recommendation was followed: more data, aligned clocks, frozen
object-disjoint splits, and no reading of test results until settings were fixed.

### 3.2 Expanded v2: all 618 DFTP clips, aligned clocks, frozen splits

All 618 Inspire DFTP trials were downloaded at a pinned revision with the recorded camera
timestamps. Sensor time is mapped to video time through the recorded frame timestamps (this
corrects both the start offset and the clock-rate difference). Objects were hashed into
object-disjoint splits before any model saw them: after 17 mechanical exclusions, **350 train /
136 development / 115 test clips over 52 / 16 / 16 objects**. Both adapters were trained with the
same recipe (released base, four epochs, seed 17, LoRA r32/a64, lr 2e-5, effective batch 16, 88
optimizer updates, two AMD MI210s). Result on the 136 development clips: video-only 82.4% /
AUROC 0.899, force 81.6% / 0.902, shuffled force 78.7% / 0.885. **No force benefit.**

A deep dive into the block the model had actually read found three defects in the force
representation: the hand-closure sign was inverted (Inspire angles run 1000 open to 0 closed), so
normalization had fitted "grasp" on open-hand frames and the scale was 37.5 raw units against
contact peaks of 1,000 to 3,000, saturating every contact at the clip maximum; per-trial resting
offsets exceeded that scale, so one clip started at maximum force with the hand open; and the
contact flag fired on every nonzero reading. A classifier over the rendered lines reached only
AUROC 0.57, against 0.78 on the raw force with a per-trial tare. The block was nearly
uninformative, which is why the model ignored it.

### 3.3 v3: corrected representation (the primary adapter)

v3 changed only the force representation and kept everything else matched to v2: same frozen
splits and window ids, same base, recipe and seed. The changes: per-trial tare, channels ranked by
peak excess, sign-corrected closure, a 5% contact floor (train-only refit: grasp 1,016 raw units,
threshold 49), the label-free summary line, and the force-only classifier probability. Measured
before training, the force-only signal is moderate: object-grouped cross-validation of force
statistics gives AUROC 0.76 to 0.80 on the training set, below video alone (0.90). Within an
object, successful trials hold with more force and sustain the hold (late/early hold ratio 1.02
against 0.83 for failures); that is the structure the block makes legible.

Training ran as Slurm job 6339 on the SDSC node `matthew` (two MI210s, 44 GB per GPU). The force
arm finished (88 updates, loss 2.49 to 0.02); the control arm was preempted at 4 h 42 min, so the
v2 control, whose inputs are identical, serves as the video-only comparison. Development and test
cells (released omitted/real, control omitted, force real/shuffled/video-blank) were scored once.

### 3.4 Baxter: the robot test

To test transfer to our own hardware, we built a tactile capture stack on a Rethink Baxter:

- **Sensor.** Four thin-film force-sensing resistors (two per gripper, `L_a L_b R_a R_b`) in a
  3.3 V divider with a 4.7 kOhm reference, read by a Teensy 4.0 at 200 Hz (12-bit ADC, 16-sample
  oversampling, one serial line per tick carrying the Teensy clock), published into ROS on the
  robot clock with a linear clock fit refreshed every 5 s. Firmware, ROS node, recorder and
  calibration tool are in `tactile_flora/baxter/`. Only the left gripper's pads were mounted for
  these rollouts; the right channels read zero. Counts are uncalibrated (no weight calibration
  file was present), the resting offset is 92 counts per pad, and the force pipeline normalizes
  them like any other source.
- **Demonstrations.** 50 hand-guided left-arm cube picks ("pick up the red cube"): the operator
  squeezes the cuff to put the arm in zero-gravity mode, guides it to the cube, closes the gripper
  with the cuff button and lifts; Enter saves, `d` discards. Recorded at 15 fps from an overhead
  webcam on Baxter's head and the left wrist camera (both 640 x 360), with joint state and gripper
  position; the demos carry no tactile data. `publish_lerobot.py` cuts each demo at the first sign
  of the gripper re-opening after its grasp and publishes `OG7407/baxter-cube-pick-left`
  (LeRobot v3). Collection code: `baxter/control/`.
- **Policy.** An ACT policy (LeRobot 0.6.1, chunk 30, 15 action steps, ResNet-18, VAE) trained
  for 50,000 steps on five RTX 2080 Ti GPUs (global batch 65, fp16), episodes 45 to 49 held out;
  held-out action L1 fell from 0.0325 (step 5,000) to 0.0267 (step 50,000). The rollouts used the
  step-30,000 checkpoint (held-out L1 0.0278). Scripts: `scripts/baxter_act/`; checkpoint
  `OG7407/baxter-act-cube-left`.
- **Rollouts.** Ten live rollouts of the policy at 30 Hz with a 1 s replan, 20 s limit and safety
  checks (joint-jump guard of 0.25 rad between the commanded chunk and the measured arm, a
  fingertip box, a speed cap), recorded with the same cameras plus the 200 Hz tactile stream
  (`il_rollouts.py`, `rollout_recorder.py`), published as `OG7407/baxter-cube-pick-left-rollouts`.
  The operator labelled episodes 0 and 1 successes and the other eight failures. Episode 0 is a
  clean grasp (closure stops at 0.64 on the cube, force at the grasp level for 2.9 s while the arm
  lifts the cube 20 cm). Episode 1 was stopped by the joint-jump guard at 2.7 s with the cube
  between the pads and force appearing 0.03 s before the last frame; the recorder auto-labelled it
  a failure and the operator overrode that. The eight failures closed on nothing (closure 1.0,
  zero force); five of them then lift and move the empty gripper exactly as a success would.

The three HRDexDB models were evaluated on these ten clips zero-shot: no Baxter clip was used for
training, normalization constants were fitted on the ten rollouts (label-free), and the left
gripper's two pads were presented as the "pad R" lines the adapters had seen in training. Six
cells (released omitted/real, control omitted, tactile real/shuffled/omitted) ran as Slurm job
6490 on two MI210s in 8 min 12 s.

## 4. Evidence

### 4.1 HRDexDB held-out results

Accuracy at the 4.0 threshold; AUROC is the probability a random success is scored above a random
failure; MAE on the 0 to 100 scale.

| Split | Model | Accuracy | AUROC | MAE | Success recall | Failure recall |
|---|---|---:|---:|---:|---:|---:|
| dev (136, 74 successes) | released, video only | 66.2% | 0.614 | 150.0 | 0.878 | 0.403 |
| dev | released, video + force | 66.2% | 0.648 | 148.5 | | |
| dev | fine-tuned, video only | 82.4% | 0.899 | 70.6 | 0.784 | 0.871 |
| dev | **fine-tuned tactile** | **85.3%** | **0.926** | **52.9** | 0.838 | 0.871 |
| dev | tactile, shuffled force | 73.5% | 0.824 | 108.8 | | |
| dev | tactile, video blanked | 53.7% | 0.775 | 120.6 | | |
| test (115, 69 successes) | released, video only | 67.0% | 0.593 | 143.5 | 0.768 | 0.522 |
| test | released, video + force | 70.4% | 0.647 | 134.8 | | |
| test | fine-tuned, video only | 69.6% | 0.733 | 121.7 | 0.725 | 0.652 |
| test | **fine-tuned tactile** | **72.2%** | **0.800** | **97.4** | 0.797 | 0.609 |

Pooled over the 251 clips (majority class 57.0%): released 66.5% / 0.600, fine-tuned video-only
76.5% / 0.830, tactile 79.3% / 0.873. Against the released model the tactile fine-tune adds 12.8
accuracy points and 0.27 AUROC; against the matched video-only fine-tune it adds 2.8 points and
0.043 AUROC.

**Statistics.** Tactile minus video-only AUROC, object-level bootstrap: +0.043 over 251 clips,
95% interval +0.001 to +0.080, p about 0.02; per split +0.026 on dev (not significant) and +0.067
on test (p about 0.04). Paired accuracy: the tactile model fixes 26 clips the video-only model
gets wrong and breaks 19 it gets right (dev 12 vs 8, test 14 vs 11; exact McNemar p = 0.37).
Shuffled force on dev: the tactile model loses 25 paired clips and wins 9 (p = 0.009).

**Where the force helps.** Splitting the clips by how sure the video-only model is:

| Clips | Video-only accuracy | Tactile accuracy |
|---|---:|---:|
| video-only E[s] within 0.75 of the threshold (101 clips) | 66.3% | **72.3%** |
| confident clips (150) | 83.3% | 84.0% |

On dev alone the uncertain band is 46 clips at 71.7% vs 82.6%. The force settles the calls the
video cannot make and leaves the confident ones alone.

**Stacking check.** A logistic stack of the video-only score, the tactile score and the force
classifier, fitted object-grouped on dev, reaches 88.2% / 0.928 on dev and 76.5% / 0.800 on test
(fit on dev): the force classifier alone gets 72.8% / 0.787 on dev and 76.5% / 0.777 on test. So
on the harder test objects the small force classifier is competitive with both VLMs on accuracy,
and the tactile VLM does not yet fully exploit it: the VLM, not the sensor, is the bottleneck.

### 4.2 What the wins look like

All 26 clips where tactile FLORA is right and the video-only model is wrong are rendered as
side-by-side videos in `videos/` (scene camera twice with each model's prediction, force trace
below). The pattern: the video-only model under-calls successes whose hold is visually
unremarkable (long, steady contact above the median force) and over-calls failures where the hand
closes convincingly on camera but the force shows a weak grip, a re-grasp, or contact that ends
early. The three largest margins:

| Clip | Truth | Video-only E[s] | Tactile E[s] | What the force showed |
|---|---|---:|---:|---|
| standing frame #4 (test) | success | 1.27 | **4.19** | peak 2.5x grasp, 16.4 s of contact ending 4.5 s before the clip ends |
| serving bowl #3 (dev) | failure | 4.66 | **2.29** | 1.4x grasp but only 6.7 s of contact in a 39 s clip |
| cactus #6 (dev) | failure | 4.19 | **1.98** | 1.8x grasp, contact ends at 14.7 s of 20.2 s |

The 19 clips the video-only model wins are listed in `comparison.md`; most are successes the
tactile model scored 3.1 to 3.8 (just under the threshold) and failures with long, strong
contact that the force cannot distinguish from a hold.

### 4.3 Baxter rollouts

Six cells on ten clips. Accuracy at 4.0; AUROC over the 2 x 8 success-failure pairs; MAE 0 to 100.

| Condition | Accuracy | AUROC | MAE | Hard scores given |
|---|---:|---:|---:|---|
| Released FLORA, no force | 0.80 | 0.375 | 140 | 2 on all 10 |
| Released FLORA + force block | 0.80 | **1.000** | 100 | 1 on 4, 2 on 6 |
| Fine-tuned FLORA, video only | 0.80 | 0.313 | 80 | 1 on all 10 |
| **Fine-tuned tactile FLORA, real force** | 0.80 | **1.000** | 80 | 1 on all 10 |
| Tactile, shuffled force | 0.80 | 0.250 | 90 | 1 on 9, 2 on 1 |
| Tactile, force removed | 0.80 | 0.438 | 80 | 1 on all 10 |
| Force-only classifier (printed in the block) | 0.80 | 0.500 | | |

Every accuracy is the majority-class 0.80: no model scores either success at or above 4.0, so
the hard answers carry nothing and the comparison lives in the ranking. Per-episode expected
scores:

| Episode | Label | Released | Released + force | Video-only | **Tactile** | Tactile, shuffled | Tactile, no force |
|---|---|---:|---:|---:|---:|---:|---:|
| ep0000 | success | 1.91 | **2.41** | 1.34 | **2.41** | 1.03 | 1.43 |
| ep0001 | success | 2.00 | **1.92** | 1.05 | **1.60** | 1.03 | 1.12 |
| ep0002 | failure | 2.01 | 1.78 | 1.40 | 1.05 | 1.05 | 1.52 |
| ep0003 | failure | 2.03 | 1.81 | 1.99 | 1.09 | 1.08 | 2.97 |
| ep0004 | failure | 1.91 | 1.06 | 1.60 | 1.02 | 1.02 | 1.32 |
| ep0005 | failure | 1.82 | 1.05 | 1.05 | 1.03 | 1.03 | 1.05 |
| ep0006 | failure | 1.99 | 1.00 | 1.05 | 1.02 | 1.62 | 1.11 |
| ep0007 | failure | 1.85 | 1.48 | 1.03 | 1.03 | 1.03 | 1.06 |
| ep0008 | failure | 2.04 | 1.70 | 1.75 | 1.04 | 1.06 | 2.31 |
| ep0009 | failure | 2.02 | 1.74 | 1.55 | 1.07 | 1.07 | 2.02 |

- **ep0000, the clean grasp.** Tactile FLORA ranks it first of ten (2.41). The video-only
  fine-tune ranks it sixth, below five failures; the released model eighth, and its rationale says
  the cube "remained on the table in the final frames" while the cube was 20 cm in the air. At 96 px
  the video models do not see the lift; the force block does.
- **ep0001, one contact frame at the end.** Tactile FLORA ranks it second, above every failure.
  Video-only ranks it eighth. One contact line with closure 0.64 (stopped by the cube) rather
  than 1.0 (closed on nothing) was enough.
- **ep0003 and ep0008, long empty lifts.** The video-only model's two highest scores (1.99,
  1.75): the arm closes, lifts and moves exactly as in a success. The tactile model gives them
  1.09 and 1.04; with its force block removed they jump to 2.97 and 2.31, the top of that cell.

The tactile model's lowest success (1.60) sits 0.51 above its highest failure (1.09). The released
model with the force block also separates the classes, but by 0.11. Shuffling the blocks drops the
tactile AUROC to 0.25 and ep0000 from 2.41 to 1.03; removing them gives 0.44 with the empty
lifts on top. The ranking is carried by the force text, and the direction matches HRDexDB: force
helps where the video is ambiguous, and here the video is ambiguous on every clip. Videos for the
seven clips where the tactile model wins by at least 0.1 are in `baxter-rollouts/videos/`.

## 5. Results statement

Across two embodiments, three comparisons and two controls, tactile FLORA outperforms regular
FLORA at judging grasp outcomes:

1. **Versus released FLORA** (what fine-tuning plus force buys): +12.8 accuracy points and +0.27
   AUROC on 251 object-disjoint HRDexDB clips; on Baxter, AUROC 0.375 to 1.0.
2. **Versus fine-tuned FLORA on identical data** (what the force alone buys): +2.8 points and
   +0.043 AUROC on HRDexDB (significant on ranking, p about 0.02), +6 points on the clips where
   video is unsure; on Baxter, AUROC 0.313 to 1.0, both successes ranked above every failure where
   the video-only model ranks them sixth and eighth.
3. **The force is causal.** Shuffled force costs 25 paired clips on dev (p = 0.009) and drops
   Baxter AUROC to 0.25; removing it on Baxter gives 0.44. Video blanked leaves 0.775 on dev, so
   the model uses both channels, and the force-only classifier alone (0.50 on Baxter) does not
   explain the robot result.
4. **The mechanism is legible.** The wins are clips where the camera cannot tell a hold from a
   touch, or an empty lift from a loaded one, and the force can.

## 6. Caveats

- **Sample size.** 251 held-out clips over 32 objects cannot certify a 3-point accuracy gain
  (about 1,000 clips would be needed at the observed win/loss ratio); the AUROC gain is the
  significant one. Baxter has ten clips and two positives: a perfect AUROC there is 16 of 16
  ordered pairs, and no significance claim is made.
- **Control provenance.** The video-only comparison is the v2 control adapter; the v3 control
  retrain was preempted. Its inputs are identical to a v3 control's, so the only unmeasured
  difference is run-to-run training noise.
- **Part of the gain rides a small classifier.** The block's summary line ends with the
  probability from a gradient-boosted force classifier; the VLM reads that line as well as the raw
  force lines. The frame lines and plain statistics remain in the block, and the Baxter result
  cannot come from the classifier (its own AUROC there is 0.5).
- **Undertrained.** Training loss reaches its floor by update 20 of 88 because the target is a
  constant template plus one digit; both fine-tuned models are data-limited.
- **96 px thumbnails.** Both fine-tuned models and the released baseline saw each camera as a
  96 px cell. The released model was trained on far larger views; higher-resolution inputs would
  likely help every video model, and the force gain may shrink as video gets stronger.
- **Baxter specifics.** Normalization was fitted on the ten test clips (label-free, but the test
  set); episode 1's success rests on one line of the block; the left gripper was presented as the
  right-hand pads the adapters knew; uncalibrated FSR counts.

## 7. Artefact index

| What | Where |
|---|---|
| HRDexDB comparison, 26 win videos, 19 losses, per-clip JSON | `writeup/comparison.md`, `writeup/videos/`, `writeup/clips.json` |
| Baxter evaluation, per-episode tables, 7 videos | `writeup/baxter-rollouts.md`, `writeup/baxter-rollouts/` |
| v3 experiment record, frozen plan, force config and classifier | `docs/experiments/hrdexdb-expanded-v3.md` and its folder |
| v2 record, splits, alignment audit | `docs/experiments/hrdexdb-expanded-v2.md` and its folder |
| Pilot and data-quality audit | `docs/experiments/hrdexdb-pilot/` |
| Force pipeline (resample, normalize, render, classifier) | `tactile_flora/force/` |
| Baxter sensor stack (firmware, ROS node, recorder, calibration) | `tactile_flora/baxter/` |
| Baxter teleop, demo collection, dataset publishing, policy rollouts | `baxter/control/` |
| ACT policy training and publishing | `scripts/baxter_act/` |
| Baxter rollout source adapter, prep and evaluation scripts | `tactile_flora/sources/baxter_rollouts.py`, `scripts/cluster/{prepare,evaluate}_baxter_rollouts.py` |
| Datasets | `OG7407/baxter-cube-pick-left` (demos), `OG7407/baxter-cube-pick-left-rollouts` (rollouts), `HRDexDB/HRDexDB` rev `98b8f8e` |
| Policy | `OG7407/baxter-act-cube-left` |
