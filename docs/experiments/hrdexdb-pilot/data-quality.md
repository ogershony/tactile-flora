# HRDexDB data quality diagnosis

The current evidence does **not establish incorrect outcome labels as the main limit**.
The stronger limits are the small sample (32 training clips, 16 validation clips), incomplete
supervision of grasp stability, visibility from one camera, and unverified force/video timing.
Relabeling everything with the same outcome VLM would risk reproducing its current errors.

The mechanical audit checked all 48 downloaded pilot episodes:

- All 48 manifest labels agree with both `grasp_result.json` and the release catalog.
  This verifies label plumbing, not independent label correctness.
- Training and validation objects are disjoint; no videos have identical SHA-256 digests.
  Related object families such as cactus variants still cross the split, and byte checks
  do not exclude near duplicates.
- All force and position arrays are finite and have matching timestamp lengths; timestamps
  increase strictly. Median sample intervals range from 20.08 to 20.22 ms.
- Video duration exceeds sensor timestamp span by 0.241–3.144 seconds. This is a coverage
  difference, **not an estimated synchronization offset**. Forty-five clips differ by over
  0.5 seconds, including every validation clip. The adapter currently assumes both streams
  start together and holds edge sensor readings outside their available span. Establish the
  actual frame-to-sensor clock mapping before attributing weak force results to label noise.

The portable episode checks and input hashes are in [quality-audit.json](quality-audit.json).
The audit samples eight overview frames per clip; it does not claim full frame-by-frame
integrity or human acceptance.

## What visual inspection suggests

Codex inspected six-frame overviews of all 48 episodes and denser 24-frame overviews of
six selected cases. These are diagnostic observations, not independent human labels:

| Episode suffix | Current label | Observation / reason to review |
|---|---|---|
| `cactus3/0` | Failure | Object tips/rolls after attempted grasp around 23–26 s; later a human resets it. The motion classifier calls this success. |
| `plant_pot/1` | Failure | Clear lift around 16–18 s, followed by return to the table around 19–20 s. Review whether control was lost; a single lift frame cannot decide stability. |
| `metal_scoop_small/1` | Failure | Hand hides the object during a long attempt; the classifier calls this success. Compare with the successful `/0` clip and use another camera if needed. |
| `spam_can/0` and `/1` | Success / failure | Both show lifting and return. The distinction requires temporal stability or loss-of-control evidence, rather than a generic “object lifted” description. |
| `pastel_light_green_bowl/0` and `/1` | Success / failure | Both include object motion/lifting. Check the actual stable hold and release. |
| `grey_pot/4`, `pringles/3` | Failure | Overview shows little visible action. Watch the full clips to distinguish genuine no-action negatives from capture failures. |
| `rectangular_frying_pan/3` | Failure | Large object motion and changing orientation; review loss of control versus deliberate placement. |

The released FLORA predictions often use near-identical “grasped and lifted, then placed
back” rationales for opposite-label trials. That suggests a criterion or temporal reasoning
problem; it is not evidence that a label should be flipped to agree with the model.
The paper describes annotations in terms of final grasp stability
([section 3.4](https://arxiv.org/html/2604.14944v1#S3.SS4)). The older local research note's
10-second hold criterion should not be imposed on every released clip without confirming
which experiment and release it applies to.

## Where human and VLM effort is useful

For an independently adjudicated gold set, a human can review all 16 validation clips: they total
7.4 minutes of raw playback. Add the highlighted training cases above, then the remaining
training clips if practical. All 48 clips total 21.6 minutes before pauses and annotation.
Keep a separate, versioned adjudication file rather than overwriting release labels.
Do not tune and report on the same adjudicated validation set as if it were an untouched test.

For each clip record stable lift, failed/unstable grasp, or cannot determine; lift start;
loss/placement time; occlusion; human intervention; and a short evidence-based reason.
Do not force occluded or borderline clips into a confident binary label. A second view can
resolve missing visual evidence more effectively than asking another model to guess.

Use a VLM to **propose event times and evidence**, then send disagreements and uncertain
cases to a human. Validate those proposals against the human-reviewed set before scaling to
more training data. Keep evaluation labels human-adjudicated. The current FLORA model is a
poor automatic judge of its own failure cases; its repeated success rationales are precisely
what this audit is investigating. Changing outcome labels alone also cannot provide the
dense contact/slip timing supervision that the force objective needs.

### Completed AI first-pass review

Codex subsequently reviewed timestamped 1 Hz frame sequences across all 16 validation clips,
plus cropped 4 Hz sequences around events in four clips. All 13,327 validation video frames
decoded successfully while producing the sheets; the visual judgments were based on the
sampled sequences, not continuous video playback. Source labels were already known, so this
is not blind adjudication or independent human ground truth.

The review supports eight stable-lift labels and five failure labels, leaving three outcomes
uncertain. Full per-episode evidence, approximate event intervals, confidence, occlusion,
human intervention and video hashes are saved in [ai-review.json](ai-review.json) and
[ai-review.csv](ai-review.csv). No source labels, evaluation scores or human-review signoffs
were changed. The initially empty human annotation file remains separate.

An optional human or second-camera check can now focus on these cases:

| Episode | Evidence to resolve |
|---|---|
| `frog_cup/0` | Cup is visibly lifted; at 18.267–19.267 s it descends together with the hand. The sampled view does not establish accidental slip versus deliberate tilted lowering. |
| `plant_pot/1` | Pot is visibly lifted at 16–18 s, then lowered gradually with the hand through 18–20 s. No decisive visible free fall establishes the source failure. |
| `metal_scoop_small/1` | Hand hides the scoop for much of the attempt. It tips around 29.767–30 s and remains on the table as the empty hand rises at 30.267 s; a stable lift is not demonstrated, but occlusion limits certainty. |

These are unresolved cases, not proposed label flips. Work can continue with the original
labels while retaining these flags. Independent human adjudication would improve confidence
in the benchmark, but is not needed to complete the AI first-pass annotations.
Local evidence sheets and their actual sample timestamps are under
`data/acceptance/hrdexdb-resume/quality/ai-review/`.

## Review packet and reproduction

The generated local packet is
`data/acceptance/hrdexdb-resume/quality/review.html`. Clips appear in a deterministic shuffled
order with source labels initially hidden. It links complete original videos and sampled
overviews. Enter judgments in the accompanying `annotations.csv`; the page does not save them.
`audit.json` maps review IDs to source episodes and mechanical checks. Rerunning preserves an
existing annotations file.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/audit_hrdexdb_quality.py
```

A separate fixed two-condition diagnostic compares the original task with an explicit stable
grasp instruction, using released FLORA without force on the current validation manifest.
It changes neither labels nor the source adapter. The exact planned suffix and input hashes
are written before model generation by `scripts/cluster/probe_hrdexdb_criterion.py`.
This is a development experiment motivated by inspected validation errors, not a new test set.

The completed comparison used the same current manifest
`8c8139b633eb5a68e53643518f7f1c4dc258823789ba8476ea2c21a14e677281`, checkpoint, frame
sampling and expected-score success threshold (4.0):

| Released FLORA, no force | Accuracy | AUROC | MAE × 100 |
|---|---:|---:|---:|
| Original task | 75% (12/16) | 0.859375 | 131.25 |
| Explicit stable-grasp task | 68.75% (11/16) | 0.796875 | 150.0 |

Both conditions produced 16 parsable answers. The explicit criterion corrected `cactus3/0`
and `potato_masher/5` but regressed `cactus3/1`, `lotion/3`, and `metal_scoop_small/0`.
Thus more explicit instructions did not improve this model overall; they traded some false
successes for false failures. This strengthens the case for human adjudication before
trusting a VLM relabeling pass. It does not establish that a different teacher would fail.
Full raw answers, expected scores and the pre-generation plan are preserved in
[criterion-result.json](criterion-result.json). Slurm job 6315 completed in 6 minutes 17 seconds
on two MI210s. Thirty relevant adapter, harness, prompt and inference tests passed;
both new scripts passed Ruff lint and formatting checks, and a stub run exercised both
conditions before the real inference run.

To reproduce (inside an allocation with sufficient GPU memory):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  data/cluster/rocm-venv/bin/python scripts/cluster/probe_hrdexdb_criterion.py \
  --model-id /path/to/pinned/Flora-Rewarder/checkpoint
```
