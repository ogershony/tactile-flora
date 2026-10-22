# Expanded HRDexDB v3: corrected force representation

The v2 run (`experiments/hrdexdb-expanded-v2.md`) finished with no force benefit on the
136-clip development set: control 82.4% / AUROC 0.899, force 81.6% / 0.902, and force with
shuffled donors 78.7% / 0.885. Both runs reached a training loss near 0.02 by update 25 of
88 because the outcome target is the constant `Rationale: no rationale` plus one score digit.
A deep dive into the force block the v2 model actually read found three defects. The code
defaults keep all of them so earlier runs stay reproducible; v3 opts out of them.

1. **Inverted hand closure.** The adapter assumed larger joint values mean a more closed
   hand. Inspire DFTP angles run 1000 (open) to 0 (closed): across all 601 retained clips the
   joint mean correlates with total force at a median of -0.87 and is negative in 98.7%. The
   normalization fit selects grasp frames by closure above 0.6, so it fitted "grasp" on
   open-hand frames: grasp 251.0 against a baseline of 213.5, scale 37.5 raw units.
2. **Saturated, offset-dominated force.** With a 37.5-unit scale and clip at 3.0, any
   contact above 326 raw units rendered as `F=3.00`, while contact peaks run 1,000-3,000.
   Per-trial resting offsets (channel medians about 13/144/-1/56/110/29) exceed the scale, so
   one clip starts at `F=3.00` with the hand open. Channels were ranked by mean, which those
   offsets dominate: the chosen pair differs between trials of the same object.
3. **Near-constant contact flag.** The free-space sigma of a tared integer sensor is 0.3
   raw units, so the three-sigma threshold flagged every nonzero reading.

Measured on the exact rendered lines, a gradient-boosted classifier over simple block
statistics reached development AUROC 0.57 for v2's representation (features such as the
contact fraction and saturation fraction barely differ between outcomes). The same probe on the raw
force with per-trial tare gives 0.78, and the trial-relative statistics below give 0.80.

## What the force carries

Object-grouped five-fold cross-validation on the 350 training clips (majority class 68.6%):

| Force-only features | CV accuracy | CV AUROC |
|---|---:|---:|
| absolute levels, contact timing | 74.0% | 0.71 |
| trial-relative hold statistics | 76-77% | 0.78-0.80 |
| implemented summary line (64-frame grid) | 78.3% | 0.78 |

Only 2 of 110 training failures lack contact, and 100 have a contact plateau of at least
5 s, so contact timing alone cannot separate outcomes. The separating structure is within
object: in 31 of 36 training objects with both outcomes, successful trials hold with more
force, and successes sustain the hold (late/early hold-force ratio 1.02 against 0.83). The
signal is therefore real but moderate, weaker than video alone (control AUROC 0.90), and
no label-free encoding examined makes it strong on its own.

## v3 changes

- `hrdexdb_force_representation: tared`: per-channel resting offsets (median of the first
  second of the sensor stream) subtracted and floored at zero; channels ranked by peak
  excess; inspire_dftp closure sign-corrected. Train-only refit: baseline 0, grasp 1,016,
  tau 49.4 (5% floor via `contact_min_frac`), no saturation.
- `render.summary: true`: one label-free summary line per arm between the header and the
  frame lines: contact segments, longest and total contact, peak, hold duration, hold force,
  late/early hold ratio, low-force hold fraction. It is computed from the same normalized
  features that the lines show, including under the shuffled and zero ablations.
- `render.classifier_json`: the summary line ends with the success probability of a
  force-only gradient-boosted classifier (`tactile_flora/force/classifier.py`) fitted on the
  350 training clips only, exported to JSON and evaluated in numpy. Training episodes show
  the score of the object-grouped fold model that never saw them (out-of-fold accuracy
  78.3%, AUROC 0.764), so the fine-tuned model learns how far to trust it instead of
  over-trusting in-sample predictions; development and test clips use the full model.
  Richer features did not help the classifier (every model tried lands at CV AUROC
  0.76-0.78), but stacking its score with the v2 control's own video score lifts
  development accuracy from 83.1% to 88.2% (AUROC 0.886 to 0.910, object-grouped CV on the
  136 clips). This is the mechanism the run tests: whether the VLM learns that weighting.
  Any gain from this line comes partly from a small classifier, not only from the VLM
  reading raw force; the plain summary statistics and frame lines remain in the block.
- Everything else is matched to v2: same frozen splits and window ids, same released base,
  four epochs, seed 17, LoRA rank 32 / alpha 64, learning rate 2e-5, effective batch 16,
  88 updates per run, final checkpoints only. The control is retrained so run-to-run noise
  is visible (its inputs are identical to v2's control).
- Evaluation adds `video_blank` on the force adapter (force only, frames mid-gray) to show
  whether the trained model reads the block at all.

Artifacts: `data/acceptance/hrdexdb-expanded-v3/` (raw is a symlink to v2's download),
[representation audit](hrdexdb-expanded-v3/representation-audit.json),
[training plan](hrdexdb-expanded-v3/training-plan.json) and batch scripts alongside.

Reproduce from the repository root after the v2 download and audit exist:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/cluster/prepare_hrdexdb_v3.py --summary --classifier
.venv/bin/python scripts/cluster/launch_hrdexdb_v3.py --after <job to wait for>
```

## Launch

Submitted 2026-10-22 as Slurm **6339** (force then control, two MI210s, after the v2 report
job 6332), development evaluation **6340**, test evaluation **6341** and report **6342**, each
dependent on the previous job's success. The [frozen plan](hrdexdb-expanded-v3/training-plan.json)
pins manifest-generation commit `a72f6a3`, core-code and input hashes including the
[classifier](hrdexdb-expanded-v3/force-classifier.json) and [force config](hrdexdb-expanded-v3/force.yaml);
the launcher's provenance check was simulated for job 6339 before submission. Dry runs
tokenize every training sample at about 17.2k (force) and 14.5k (control) tokens; the largest
outcome force block is 2,700 tokens. Six dev/test conditions: released omitted/real, control
omitted, force real/shuffled/video_blank.
