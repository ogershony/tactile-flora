# HRDexDB pilot diagnostic snapshot

This is an exploratory single-hand outcome experiment on 32 training episodes and 16
validation episodes. Training and validation have disjoint episodes and object names.
The snapshot contains the actual saved predictions, raw model answers, labels, training
losses, configs, normalization constants and manifest hashes, without raw videos or weights.
See [snapshot.json](snapshot.json).

The current validation manifest hash is
`8c8139b633eb5a68e53643518f7f1c4dc258823789ba8476ea2c21a14e677281`.
Only these two saved evaluations match it:

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
This attempt did not improve the outcome predictor; higher-resolution visual experiments
have not yet run.

Pre-commit validation: 333 tests passed; Ruff lint/format and Pyright passed. The managed
shell required the previously documented temporary selector polling workaround for pytest
and an explicit virtualenv config for Pyright. Neither workaround is committed.
