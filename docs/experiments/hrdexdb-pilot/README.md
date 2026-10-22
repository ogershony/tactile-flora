# HRDexDB pilot diagnostic snapshot

The active working subset is now **25 train / 13 validation** episodes after an AI review of
all 48 local pilot clips. See [the cleaned dataset](cleaned-dataset.md) for exclusions,
manifests, provenance and runnable configs. The snapshot and scores below describe the
original 32/16 pilot; they have not been rewritten as cleaned-subset results.
Matched force/control training and completed evaluation are described in
[the current training experiment](clean-training.md) and [portable results](clean-result.json).
On the cleaned 13-clip development subset, neither adapter improves accuracy over its
matching released baseline (84.6% without force; 61.5% with real force).

This is an exploratory single-hand outcome experiment on 32 training episodes and 16
validation episodes. Training and validation have disjoint episodes and object names.
The snapshot contains the actual saved predictions, raw model answers, labels, training
losses, configs, normalization constants and manifest hashes, without raw videos or weights.
See [snapshot.json](snapshot.json).

The current validation manifest hash is
`8c8139b633eb5a68e53643518f7f1c4dc258823789ba8476ea2c21a14e677281`.
The original snapshot has two evaluations matching it:

| Checkpoint | Input | Accuracy | AUROC | MAE × 100 |
|---|---|---:|---:|---:|
| Released FLORA | Real force + video | 62.5% | 0.640625 | 168.75 |
| HRDexDB signed-force SFT | Real force + video | 56.25% | 0.65625 | 193.75 |

The signed-force adapter mostly predicts success for failed grasps. Its eight optimizer
steps reduced training loss from 2.608 to 1.373; this did not improve validation accuracy.
The saved released no-force result is 75% accuracy / 0.859375 AUROC, but uses the older
`bdcd8a53…` manifest. Other older results are also retained and explicitly marked
`matches_current_manifest: false`. They need rerunning before a controlled comparison.
These results do not establish a force benefit or full benchmark acceptance.

Continuation now includes an audit of all 48 pilot clips and a fixed two-prompt released
FLORA diagnostic. See [the data quality diagnosis](data-quality.md) for label checks,
visual failure cases, sensor coverage differences and the human/VLM review recommendation.
The new generic no-force run reproduces 75% accuracy / 0.859375 AUROC against the **current**
manifest, making that baseline usable alongside the current real-force results.
Explicit stable-grasp wording scored 68.75%, correcting two failures but regressing three
previously correct clips. Both conditions and all raw answers are saved in
[criterion-result.json](criterion-result.json); this validation diagnostic did not relabel
any data or improve the model.

The source adapter now decodes unsigned 16-bit force representations before taking absolute
values. Older runs predate this correction and/or train-only normalization. The original
eval-only source policy is retained in the source config; the pilot training launcher uses
an explicit experimental override for its frozen training split. It does not alter the
default pipeline policy.

The new sensor diagnostic script uses all six force and position channels, with model
selection by leave-one-training-object-out cross-validation. Features exclude episode IDs,
object names, absolute timestamps and label files. It fits only the training episodes and
scores the selected model on the 16 validation episodes. Install its optional dependency
and run from the project root:

```sh
uv pip install --python .venv/bin/python scikit-learn
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/probe_hrdexdb_features.py
```

It requires the existing local pilot files under `data/acceptance/hrdexdb-pilot/` and writes
to the separate `data/acceptance/hrdexdb-resume/sensors/` directory. The snapshot records
the split IDs needed to reconstruct those local manifests. Raw datasets, checkpoints and
generated experiment directories remain gitignored. Diagnostic scripts do not modify the
frozen splits, existing adapters or prior results.

The first completed sensor probe achieved 50% validation accuracy and 0.453125 AUROC.
Its model was selected using training objects only. Full candidate cross-validation results
and the selected model's evaluation predictions are in [sensor-probe.json](sensor-probe.json).
This sensor attempt did not improve the outcome predictor. A later CPU video classifier
reached **81.25% validation accuracy (13/16)** and **0.90625 AUROC** after selecting among
56 settings using validation accuracy. Classifiers fit only training episodes. This is a
validation-tuned development result from a separate visual classifier, not a FLORA adapter
improvement. See [the video predictor instructions](video-predictor.md) and
[full predictions and candidate scores](video-result.json).

Pre-commit validation: 333 tests passed; Ruff lint/format and Pyright passed. The managed
shell required the previously documented temporary selector polling workaround for pytest
and an explicit virtualenv config for Pyright. Neither workaround is committed.
