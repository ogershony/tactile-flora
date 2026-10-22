"""Train-only grouped model selection for reversible HRDexDB sensor experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tactile_flora.sources.hrdexdb import _signed_force

BASE = Path("data/acceptance/hrdexdb-pilot")


def features(trial: Path) -> tuple[np.ndarray, list[str]]:
    """Summarize measured signals without paths, IDs, outcomes or absolute timestamps."""
    force = _signed_force(np.load(trial / "raw/hand/force.npy"))
    position = np.load(trial / "raw/hand/position.npy").astype(float)
    values: list[float] = []
    names: list[str] = []
    for name, signal in [("force", force), ("position", position)]:
        # Absolute amplitudes, changes from initial state, and dimensionless dynamics.
        baseline = np.median(signal[: max(1, len(signal) // 10)], axis=0)
        for variant, a in [("raw", signal), ("delta", signal - baseline)]:
            for phase, lo, hi in [
                ("all", 0, 1),
                ("start", 0, 0.1),
                ("middle", 0.3, 0.7),
                ("end", 0.8, 1),
            ]:
                part = a[int(len(a) * lo) : max(int(len(a) * lo) + 1, int(len(a) * hi))]
                stats = np.concatenate(
                    [
                        np.quantile(part, [0.1, 0.5, 0.9], axis=0),
                        np.std(part, axis=0, keepdims=True),
                    ],
                    axis=0,
                )
                for stat, row in zip(["q10", "median", "q90", "std"], stats, strict=True):
                    values.extend(row.tolist())
                    names.extend(f"{name}.{variant}.{phase}.{stat}.{i}" for i in range(len(row)))
        amplitude = np.maximum(np.ptp(signal, axis=0), 1)
        for stat, row in [
            (
                "end_fraction",
                (np.median(signal[-max(1, len(signal) // 10) :], axis=0) - baseline) / amplitude,
            ),
            ("motion", np.mean(np.abs(np.diff(signal, axis=0)), axis=0) / amplitude),
        ]:
            values.extend(row.tolist())
            names.extend(f"{name}.{stat}.{i}" for i in range(len(row)))
    return np.asarray(values), names


def main() -> None:
    """Select on training objects and score the selected sensor model on held-out objects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/acceptance/hrdexdb-resume/sensors")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = json.loads((BASE / "manifest/train-sample.json").read_text())["episodes"]
    val = json.loads((BASE / "manifest/split.json").read_text())["episodes"]
    groups = np.array([r["episode_id"].split("/")[1] for r in train])
    if set(groups) & {r["episode_id"].split("/")[1] for r in val}:
        raise ValueError("training and evaluation objects overlap")
    xy = [features(BASE / "raw" / r["episode_id"]) for r in train + val]
    x = np.stack([a for a, _ in xy])
    names = xy[0][1]
    y = np.array([int(r["success"]) for r in train + val])
    n = len(train)
    candidates = {}
    for c in [0.001, 0.01, 0.1, 1, 10]:
        candidates[f"logistic-{c}"] = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=3000)
        )
        candidates[f"rbf-{c}"] = make_pipeline(
            StandardScaler(), SVC(C=c, probability=True, random_state=17)
        )
    for depth in [2, 4, None]:
        candidates[f"extra-{depth}"] = ExtraTreesClassifier(
            n_estimators=300, max_depth=depth, min_samples_leaf=2, random_state=17, n_jobs=1
        )
        candidates[f"forest-{depth}"] = RandomForestClassifier(
            n_estimators=300, max_depth=depth, min_samples_leaf=2, random_state=17, n_jobs=1
        )
    results = []
    for subset in ["all", "force", "position"]:
        indices = np.array(
            [i for i, name in enumerate(names) if subset == "all" or name.startswith(subset + ".")]
        )
        for key, estimator in candidates.items():
            prob = cross_val_predict(
                estimator,
                x[:n, indices],
                y[:n],
                groups=groups,
                cv=LeaveOneGroupOut(),
                method="predict_proba",
            )[:, 1]
            result = dict(
                model=key,
                subset=subset,
                cv_accuracy=accuracy_score(y[:n], prob >= 0.5),
                cv_auroc=roc_auc_score(y[:n], prob),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    # Select using training objects only, before any evaluation labels are used in scoring.
    results.sort(key=lambda r: (r["cv_accuracy"], r["cv_auroc"]), reverse=True)
    chosen = results[0]
    indices = np.array(
        [
            i
            for i, name in enumerate(names)
            if chosen["subset"] == "all" or name.startswith(chosen["subset"] + ".")
        ]
    )
    estimator = clone(candidates[chosen["model"]]).fit(x[:n, indices], y[:n])
    prob = estimator.predict_proba(x[n:, indices])[:, 1]
    predictions = pd.DataFrame(
        {"episode_id": [r["episode_id"] for r in val], "success_probability": prob, "label": y[n:]}
    )
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    np.savez(args.output / "features.npz", train=x[:n], val=x[n:], names=names)
    import joblib

    joblib.dump(
        {"estimator": estimator, "indices": indices, "feature_names": names},
        args.output / "model.joblib",
    )
    report = dict(
        selection="leave-one-training-object-out; accuracy then AUROC",
        selected=chosen,
        train_episodes=n,
        val_episodes=len(val),
        train_objects=sorted(set(groups)),
        val_accuracy=accuracy_score(y[n:], prob >= 0.5),
        val_auroc=roc_auc_score(y[n:], prob),
        candidates=results,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
