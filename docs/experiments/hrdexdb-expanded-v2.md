# Expanded HRDexDB with recorded video clocks

The cleaned 25/13 pilot did not demonstrate a training or force benefit. This follow-up
addresses its sample size and the incorrect assumption that sensor recording starts at
video frame zero. It does not assume that correcting these inputs guarantees better scores.

The source revision is `98b8f8ea2e361a3c7df38b5d8f3f8395913773f2` of
`HRDexDB/HRDexDB`. Download all 618 Inspire DFTP clips with one fixed scene camera
(`22641005`), force, hand position, hand time, and camera timestamps/frame IDs. No other
camera streams or embodiments are needed. Downloads use atomic writes and SHA256 receipts.

Before viewing new videos or running inference, freeze object assignments. All historical
training objects remain train; all historical validation objects remain development
validation. Hash-order previously unused objects with seed string `hrdexdb-v2:20261021`;
reserve 16 new objects for test and eight for additional development validation. This yields
363 train / 139 development / 116 test clips, across 52 / 16 / 16 objects, before exclusions.
No model results determine membership. Test inference waits until training settings are
fixed; the old pilot and its reviewed exclusions remain preserved separately.

The [upstream clock loader and viewer](https://github.com/snuvclab/HRDexDB/blob/main/hrdexdb/common.py)
use `raw/timestamps/timestamp.npy`, `frame_id.npy`, and one-based video frame IDs. Map sensor
epoch timestamps to encoded video seconds using these correspondences, with interpolation
between recorded frames. Crop sensor samples outside the video interval. This corrects both
the start offset and capture/encoded clock-rate differences; a single constant shift would
not suffice. In the original pilot, valid timelines start 2.220–2.335 seconds after the hand
stream and have materially different capture and encoded durations.

Set `hrdexdb_use_video_timestamps: true` explicitly for new source configs. Legacy configs
retain their former behavior. The new mode refuses missing, non-monotonic, non-integer or
incomplete frame timelines. It requires IDs starting at one and ending at the actual video
frame count; it does not guess a mapping when frames and timestamp IDs disagree. Force and
gripper arrays are cropped together and share the corrected time base. This verifies the
released clock correspondence; it is not an independent hardware-latency calibration.

Before fitting, apply fixed quality rules to every split: preserve the ten original visual
exclusions; reject invalid arrays, conflicting raw/catalog labels, invalid clocks, full-video
decode failures, camera timestamp gaps over 0.25 seconds, and exact duplicate videos (all
copies). Never move an excluded held-out clip into training. New source labels are not
claimed to be independently human reviewed. Train-only force normalization is refit after
exclusions. Preserve all frozen holdout assignments, including excluded episodes.

Matched training settings are fixed before test evaluation: released FLORA revision
`7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`, four epochs, seed 17, LoRA rank 32 / alpha 64,
learning rate 2e-5, effective batch 16, warmup 0.1, gradient clipping 1.0. Train force and
omitted-force control from the released base and evaluate final checkpoints. The trainer
now closes partial global batches before the next epoch, fixing the seven-versus-eight
update discrepancy discovered in the pilot; regression tests cover one and two ranks.
No validation-selected checkpoint or test-based tuning is planned. Classification keeps
the existing expected-score threshold of four.

Run from the repository root:

```sh
.venv/bin/python scripts/cluster/download_hrdexdb_expanded.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/prepare_hrdexdb_aligned.py
```

Artifacts live under `data/acceptance/hrdexdb-expanded-v2/`: `download-plan.json`, download
receipts, `alignment-audit.json`, manifests/holdout, train-only normalization and configs.
The separate `hrdexdb-aligned-pilot-v2` root checks the corrected timing on the original
validation IDs; clock validation retains 24 training and 13 validation clips there.
The expanded audit is complete; training results are pending.

The download completed: 618 episodes, 4,326 per-episode files and 20.22 GB including the
catalog. [Frozen source and object assignments](hrdexdb-expanded-v2/split.json) are portable.
The aligned pilot's released real-force result is **69.2% (9/13), AUROC 0.650**, with zero
parse failures, versus 61.5% (8/13) in the prior real-force pipeline. This includes a refit of
normalization on 24 aligned training clips after `blue_alarm/0` failed frame-count validation;
it does not isolate the clock change alone. No-force remains stronger at 84.6% (11/13).
[Diagnostic predictions, audit and configuration](hrdexdb-pilot/aligned-result.json) preserve
the comparison. Expanded training is needed to assess whether the corrected inputs help SFT.

The [completed mechanical audit](hrdexdb-expanded-v2/alignment-audit.json) retains **350
train / 136 development / 115 test** clips across the original 52 / 16 / 16 object groups.
Training has 240 successes / 110 failures; development 74 / 62; test 69 / 46. Seventeen
clips are excluded: the ten original visual exclusions plus seven additional invalid clock
records (one original exclusion also has an invalid clock). All 601 retained videos decode
fully, totaling 519,472 frames. No additional labels were flipped or independently certified.
Both training preflights pass with exactly 350 samples and no frozen holdout leakage.
Each four-epoch run plans 88 optimizer updates, including its short final batch each epoch.
Both source configs and train-only normalization are frozen before training.
