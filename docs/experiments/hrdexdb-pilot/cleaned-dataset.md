# Cleaned HRDexDB pilot v1

The working dataset is now `data/acceptance/hrdexdb-clean-v1/`: **25 training episodes and
13 validation episodes**. The three previously flagged validation clips are excluded.
This pass covers all 48 locally downloaded pilot clips, not the full 2,104-row release
catalog or PHANTOM, SharpaDex and other datasets in the workspace.

Codex inspected timestamped 1 Hz frame sequences spanning all 48 clips, with additional
4 Hz event sequences for four validation clips. All 38,933 source video frames decoded
while creating the sheets; visual judgments use the sampled images, not continuous playback.
Source labels were known. This is an AI curation pass, not independent human adjudication.

| Split | Original | Kept | Excluded | Kept success / failure |
|---|---:|---:|---:|---:|
| Train | 32 | 25 | 7 | 16 / 9 |
| Validation | 16 | 13 | 3 | 8 / 5 |

The selection keeps clear source-consistent outcomes and removes unresolved outcomes or clips
without an observable completed attempt. Difficult but clear failures remain, including slips,
topples and unsuccessful lifts. The pass found no sufficiently certain label correction to
justify flipping a source label. Every keep/exclude decision has an explanation and video hash
in [cleaning-review.json](cleaning-review.json) and [cleaning-review.csv](cleaning-review.csv).

## Excluded episodes

All IDs below have the prefix `inspire_dftp/`.

| Split | Episode | Reason |
|---|---|---|
| Validation | `frog_cup/0` | Lift and lowering visible; accidental slip versus deliberate lowering unresolved. |
| Validation | `plant_pot/1` | Lift and gradual lowering visible; failure criterion unresolved from this view. |
| Validation | `metal_scoop_small/1` | Hand occludes most of the attempt; stable lift versus unstable contact unresolved. |
| Train | `frame_oak/0` | Lift and upright return visible; source failure versus deliberate pose change unresolved. |
| Train | `pastel_light_green_bowl/1` | Clear lift and upright return; no decisive visible failure establishes source label. |
| Train | `rectangular_frying_pan/3` | Large rotation with handle close to table; full clearance and intended motion unresolved. |
| Train | `spam_can/1` | Lift and upright return visible; no decisive loss of control establishes source failure. |
| Train | `grey_pot/4` | No visible grasp attempt or outcome captured. |
| Train | `pringles/3` | Hand stays above object; no visible contact/lift in the clip. |
| Train | `yellow_wash_tub/3` | Setup and approach visible, but clip ends without an observable completed grasp attempt. |

The three no-attempt clips are excluded for this grasp-outcome training subset; the images do
not prove that their original failure flags are incorrect. They can remain useful for a
separately defined no-action detection task.

## Dataset artifacts and checks

The clean root contains filtered JSON splits, combined/train/validation Parquet manifests,
source/evaluation/training configs, the decision ledger and `summary.json`. Raw videos are
referenced through a symlink to the original files. Original manifests, source labels and
saved scores remain intact. The original holdout file is copied in full, so excluded validation
episodes remain held out and cannot enter training.

Force normalization is refit using exactly the 25 retained training episodes. The training
launcher now defaults to the clean root and only prepares episodes present in its manifest.
The existing force training mode is retained, with a separate omitted-force control config.
This visual cleanup does not establish force/video synchronization: the previously observed
0.24–3.14 second coverage differences remain unresolved.

The subset was curated after inspecting validation examples and model errors. Its metrics
must be reported as a curated development subset, not as an improvement on the original
16-episode benchmark. Dropping three negative validation cases also changes class balance.
No saved original score has been rewritten as a cleaned score.

The exact counts, normalization fit IDs and new manifest hashes are in
[clean-summary.json](clean-summary.json). The builder refuses missing/duplicate decisions,
split changes, label changes, unresolved kept verdicts, overlapping objects, changed videos,
changed input splits and an existing output directory.

Validation completed: 47 relevant unit tests passed; Ruff lint/format and whitespace checks
passed. Both real-tokenizer dry runs built exactly 25 training samples. A fresh temporary
rebuild reproduced the selection summary, normalization version and all three manifest hashes.
Checks confirmed that all excluded IDs are absent from clean manifests, all original 16
validation IDs remain held out, and the original split file hashes are unchanged.

```sh
# Rebuild into a fresh version directory; the existing v1 is never overwritten.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/clean_hrdexdb_pilot.py \
  --output data/acceptance/hrdexdb-clean-v2

# Validate force training data without loading model weights or starting training.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/run_hrdexdb_sft.py --dry-run \
  --run-name clean-v1-force-dry-run

# Validate the matched omitted-force control.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/run_hrdexdb_sft.py --dry-run \
  --config data/acceptance/hrdexdb-clean-v1/config/train-control.yaml \
  --run-name clean-v1-control-dry-run
```

Use `--base data/acceptance/hrdexdb-pilot` to explicitly select the historical unfiltered
training root. The original `prepare_hrdexdb_sft.py` reconstructs the original frozen pilot;
use the cleaning builder for reviewed versions. CPU visual experiments can use their existing
`--base data/acceptance/hrdexdb-clean-v1 --output <new-output-directory>` options.
