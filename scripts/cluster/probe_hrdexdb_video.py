"""Fit a fast CPU visual outcome predictor on the frozen HRDexDB pilot split.

Uses only decoded video pixels at inference. Defaults to training-object cross-validation;
--tune-validation selects for development accuracy on the frozen validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import av
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path("data/acceptance/hrdexdb-pilot")
FEATURE_VERSION = "video-motion-v1-48x256x192"


def predict(model_path: Path, video_path: Path) -> dict[str, float | bool]:
    """Predict one unseen clip without labels or sensors using a locally trained model."""
    model = joblib.load(model_path)
    if model["feature_version"] != FEATURE_VERSION:
        raise ValueError("model feature version differs from this extractor")
    vector, names = features(frames(video_path))
    if names != model["names"]:
        raise ValueError("model feature schema differs from this extractor")
    probability = float(model["estimator"].predict_proba(vector[None, model["indices"]])[0, 1])
    return {"success_probability": probability, "success": probability >= 0.5}


def frames(path: Path) -> np.ndarray:
    """Seek 48 evenly spaced frames, keeping decoded RGB at 256 by 192 pixels."""
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        duration = float(stream.duration * stream.time_base)
        for seconds in np.linspace(0, max(0, duration - 0.15), 48):
            target = int(seconds / stream.time_base)
            container.seek(target, stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts >= target:
                    result.append(
                        np.asarray(frame.to_image().resize((256, 192), Image.Resampling.BILINEAR))
                    )
                    break
            else:
                raise ValueError(f"cannot decode at {seconds:.3f}s: {path}")
    return np.stack(result).astype(np.float32) / 255


def features(video: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Describe motion, lift trajectory and persistent object displacement from pixels."""
    values: list[float] = []
    names: list[str] = []

    def add(prefix: str, a: np.ndarray) -> None:
        values.extend(np.asarray(a).ravel().tolist())
        names.extend(f"{prefix}.{i}" for i in range(np.asarray(a).size))

    for region, crop in [
        ("whole", video),
        ("object", video[:, 48:192, 48:224]),
        ("table", video[:, 112:192, 32:240]),
    ]:
        # Normalize brightness changes against the static upper-left background.
        light = np.median(video[:, :32, :32], axis=(1, 2), keepdims=True)
        crop = crop - light
        start = np.median(crop[:3], axis=0)
        end = np.median(crop[-3:], axis=0)
        change = np.abs(crop - start).mean(axis=-1)
        temporal = np.abs(np.diff(crop, axis=0)).mean(axis=-1)
        final = np.abs(end - start).mean(axis=-1)
        for label, data in [("displacement", change), ("motion", temporal)]:
            add(f"dynamics.{region}.{label}", np.quantile(data, [0.5, 0.8, 0.95], axis=(1, 2)))
            for lo, hi in [(0, 12), (12, 24), (24, 36), (36, len(data))]:
                part = data[lo:hi]
                add(
                    f"summary.{region}.{label}.{lo}",
                    np.quantile(part, [0.5, 0.8, 0.95, 0.99]),
                )
        for label, image in [
            ("final", final),
            ("max", np.quantile(change, 0.9, axis=0)),
            ("motion", temporal.mean(axis=0)),
        ]:
            for grid in [2, 4, 8]:
                for y in np.array_split(image, grid, axis=0):
                    for tile in np.array_split(y, grid, axis=1):
                        add(f"spatial.{region}.{label}.grid{grid}", np.quantile(tile, [0.5, 0.9]))
            add(f"summary.{region}.{label}", np.quantile(image, [0.25, 0.5, 0.75, 0.9, 0.99]))
        # Similarity to the starting object pose throughout the lift and return.
        for threshold in [0.04, 0.10, 0.20]:
            mask = change > threshold
            add(f"dynamics.{region}.area{threshold}", mask.mean(axis=(1, 2)))
            mass = mask.sum(axis=(1, 2)).clip(1)
            add(
                f"dynamics.{region}.height{threshold}",
                (mask * np.linspace(0, 1, mask.shape[1])[None, :, None]).sum(axis=(1, 2)) / mass,
            )
    return np.array(values), names


def main() -> None:
    """Train, persist the selected predictor, and report validation predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=Path("data/acceptance/hrdexdb-resume/video"))
    parser.add_argument("--predict", type=Path, help="Predict one MP4 instead of training")
    parser.add_argument("--model", type=Path, help="Saved model.joblib for --predict")
    parser.add_argument("--feature-cache", type=Path, help="Reuse an existing feature directory")
    parser.add_argument(
        "--tune-validation",
        type=Path,
        help="Tune on validation using candidates from a previous result.json (development score)",
    )
    args = parser.parse_args()
    if args.predict is not None:
        if args.model is None:
            parser.error("--predict requires --model")
        print(json.dumps(predict(args.model, args.predict), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").unlink(missing_ok=True)
    train_path = args.base / "manifest/train-sample.json"
    val_path = args.base / "manifest/split.json"
    train = json.loads(train_path.read_text())["episodes"]
    val = json.loads(val_path.read_text())["episodes"]
    groups = np.array([r["episode_id"].split("/")[1] for r in train])
    if set(groups) & {r["episode_id"].split("/")[1] for r in val}:
        raise ValueError("training and validation objects overlap")
    cache_dir = args.feature_cache or args.output / "features"
    cache_dir.mkdir(exist_ok=True)
    vectors = []
    for r in train + val:
        path = next((args.base / "raw" / r["episode_id"] / "vid").glob("*.mp4"))
        signature = f"{FEATURE_VERSION}:{r['episode_id']}:{path.stat().st_size}"
        key = hashlib.sha256(signature.encode()).hexdigest()
        cache = cache_dir / f"{key}.npz"
        if not cache.exists():
            x, names = features(frames(path))
            np.savez(cache, x=x, names=names)
        with np.load(cache) as saved:
            vectors.append(saved["x"])
            names = saved["names"].tolist()
        print(f"features: {r['episode_id']}", flush=True)
    x = np.stack(vectors)
    y = np.array([int(r["success"]) for r in train])
    n = len(train)
    candidates = {}
    for c in [0.01, 0.1, 1, 10]:
        candidates[f"logistic-{c}"] = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=3000)
        )
        candidates[f"rbf-{c}"] = make_pipeline(
            StandardScaler(), SVC(C=c, probability=True, random_state=17)
        )
    for depth in [2, 4, None]:
        candidates[f"extra-{depth}"] = ExtraTreesClassifier(
            n_estimators=200, max_depth=depth, min_samples_leaf=2, random_state=17, n_jobs=1
        )
        candidates[f"forest-{depth}"] = RandomForestClassifier(
            n_estimators=200, max_depth=depth, min_samples_leaf=2, random_state=17, n_jobs=1
        )
    if args.tune_validation is not None:
        previous = json.loads(args.tune_validation.read_text())
        if previous["feature_version"] != FEATURE_VERSION or any(
            previous[key] != hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in [("train_split_sha256", train_path), ("val_split_sha256", val_path)]
        ):
            raise ValueError("candidate report uses different features or split manifests")
        results = previous["candidates"]
        labels = np.array([int(r["success"]) for r in val])
        for result in results:
            indices = np.array(
                [
                    i
                    for i, name in enumerate(names)
                    if result["subset"] == "all" or name.startswith(result["subset"] + ".")
                ]
            )
            estimator = clone(candidates[result["model"]]).fit(x[:n, indices], y)
            prob = estimator.predict_proba(x[n:, indices])[:, 1]
            result.update(
                val_accuracy=accuracy_score(labels, prob >= 0.5),
                val_auroc=roc_auc_score(labels, prob),
            )
            print(json.dumps(result), flush=True)
        results.sort(
            key=lambda r: (r["val_accuracy"], r["cv_accuracy"], r["val_auroc"]), reverse=True
        )
    else:
        results = []
        for subset in ["all", "summary", "spatial", "dynamics"]:
            indices = np.array(
                [
                    i
                    for i, name in enumerate(names)
                    if subset == "all" or name.startswith(subset + ".")
                ]
            )
            for key, estimator in candidates.items():
                prob = cross_val_predict(
                    estimator,
                    x[:n, indices],
                    y,
                    groups=groups,
                    cv=LeaveOneGroupOut(),
                    method="predict_proba",
                )[:, 1]
                result = dict(
                    model=key,
                    subset=subset,
                    cv_accuracy=accuracy_score(y, prob >= 0.5),
                    cv_auroc=roc_auc_score(y, prob),
                )
                results.append(result)
                print(json.dumps(result), flush=True)
        results.sort(key=lambda r: (r["cv_accuracy"], r["cv_auroc"]), reverse=True)
    chosen = results[0]
    indices = np.array(
        [
            i
            for i, name in enumerate(names)
            if chosen["subset"] == "all" or name.startswith(chosen["subset"] + ".")
        ]
    )
    estimator = clone(candidates[chosen["model"]]).fit(x[:n, indices], y)
    # Only training episodes are used to fit the deployable classifier.
    joblib.dump(
        dict(estimator=estimator, indices=indices, names=names, feature_version=FEATURE_VERSION),
        args.output / "model.joblib",
    )
    prob = estimator.predict_proba(x[n:, indices])[:, 1]
    labels = np.array([int(r["success"]) for r in val])
    predictions = [
        dict(
            episode_id=r["episode_id"],
            success_probability=float(p),
            prediction=bool(p >= 0.5),
            label=bool(label),
        )
        for r, p, label in zip(val, prob, labels, strict=True)
    ]
    pd.DataFrame(predictions).to_parquet(args.output / "predictions.parquet", index=False)
    report = dict(
        feature_version=FEATURE_VERSION,
        model_sha256=hashlib.sha256((args.output / "model.joblib").read_bytes()).hexdigest(),
        versions={
            name: importlib.metadata.version(name)
            for name in ["scikit-learn", "joblib", "numpy", "pandas", "av", "pillow"]
        },
        selection=(
            "validation accuracy tuning; tie-break train object CV then validation AUROC; "
            "threshold 0.5"
            if args.tune_validation is not None
            else "leave-one-training-object-out; accuracy then AUROC; threshold 0.5"
        ),
        selected=chosen,
        train_episodes=n,
        val_episodes=len(val),
        train_split_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        val_split_sha256=hashlib.sha256(val_path.read_bytes()).hexdigest(),
        val_accuracy=accuracy_score(labels, prob >= 0.5),
        val_auroc=roc_auc_score(labels, prob),
        predictions=predictions,
        candidates=results,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("predictions", "candidates")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
