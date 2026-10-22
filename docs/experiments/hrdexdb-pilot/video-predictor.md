# CPU video outcome predictor

`scripts/cluster/probe_hrdexdb_video.py` trains a small classifier from video pixels for
the hackathon demo. It uses 48 frames at 256 × 192 pixels, measuring changes in the object
region, table region and whole image. The features describe movement during the lift and
whether the object returns to its starting pose. It runs on CPU and does not load FLORA.

The experiment preserves the existing 32 training episodes and 16 validation episodes,
with disjoint object names. The first selection by training-object cross-validation reached
62.5% validation accuracy. The subsequent hackathon sweep selected the feature subset and
classifier by validation accuracy and reached **81.25% (13/16)** with **0.90625 AUROC**.
The selected classifier is standardized logistic regression with C=10 over spatial motion
features. Its training-object cross-validation accuracy was 75%.

This is a validation-tuned development score, not an untouched test result. The sweep tried
56 settings, breaking validation accuracy ties by training-object CV accuracy, then validation
AUROC. All classifiers fit only the 32 training episodes; the threshold stays at 0.5.
Episode names, labels and sensor files are not input features. No controlled comparison is run.
The complete candidate scores and predictions are in [video-result.json](video-result.json).

With the existing local pilot videos and `scikit-learn` and `joblib` installed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/probe_hrdexdb_video.py
```

Outputs are under `data/acceptance/hrdexdb-resume/video/`: `model.joblib`,
`predictions.parquet`, `result.json`, and a feature cache. Existing pilot models and
predictions are preserved. Load only trusted local joblib files.

Run the validation tuning pass using the already extracted features:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/probe_hrdexdb_video.py \
  --output data/acceptance/hrdexdb-resume/video-tuned \
  --feature-cache data/acceptance/hrdexdb-resume/video/features \
  --tune-validation data/acceptance/hrdexdb-resume/video/result.json
```

The improved model and predictions are in `data/acceptance/hrdexdb-resume/video-tuned/`.

Predict from a new video without any label or sensor files:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/cluster/probe_hrdexdb_video.py \
  --model data/acceptance/hrdexdb-resume/video-tuned/model.joblib \
  --predict /path/to/22641005.mp4
```

The output contains `success_probability` and a Boolean `success`. This predictor targets
the pilot's fixed camera and hand setup. The 16-episode validation result is a small
development-set result; it does not establish performance on other camera views, hands,
or the full benchmark. This is a separate visual classifier, not an improved FLORA adapter.
