"""Stacked head: a logistic regression over VLM scores and the force-only classifier.

Owns: joining per-cell ``predictions.parquet`` files on window id, the force-classifier
probability per window (recomputed through the same adapter -> resample -> normalize ->
``ForceScorer.score`` chain the summary line uses), fitting a standardized L2 logistic
regression on one split, scoring another, and object-grouped out-of-fold predictions so the
fitting split's own number is never in-sample. Never fits on the split it reports as "test".
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.eval.metrics import SUCCESS_THRESHOLD
from tactile_flora.force import normalize
from tactile_flora.force.classifier import load_scorer
from tactile_flora.force.resample import resample
from tactile_flora.labels.windows import outcome_frame_indices
from tactile_flora.verify import manifest as manifest_mod

FEATURES = ("tactile_E", "control_E", "classifier_logit")


class StackModel(BaseModel):
    """A fitted stack: standardization, weights and the provenance of the fitting rows."""

    model_config = ConfigDict(extra="forbid")

    features: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    l2: float
    fit_split: str
    n_fit: int
    fit_window_ids_sha256: str


def _label(truth_target: pd.Series) -> pd.Series:
    score = truth_target.str.extract(r"Score:\s*(\d)")[0].astype(int).to_numpy()
    return pd.Series((score >= SUCCESS_THRESHOLD).astype(int), index=truth_target.index)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def classifier_probabilities(
    base: Path, manifest_path: Path, force_cfg: ForceConfig, source_cfg: SourceConfig
) -> pd.Series:
    """Force-only classifier P(success) per window id, exactly as the summary line prints it."""
    from tactile_flora.sources.hrdexdb import HrdexdbAdapter

    consts = normalize.load(base / "verified/hrdexdb/norm.json")
    scorer = load_scorer(force_cfg.render.classifier_json or "")
    adapter = HrdexdbAdapter(source_cfg)
    adapter.discover(Path(source_cfg.root))
    out: dict[str, float] = {}
    for row in manifest_mod.read_manifest(manifest_path):
        ep = adapter.load(row.episode_id)
        cam = ep.cameras["scene"]
        times = [i / cam.fps for i in outcome_frame_indices(cam.n_frames, cam.fps)]
        feats = normalize.apply(
            resample(ep.force, ep.gripper, times, force_cfg.resample), consts, force_cfg.render
        )
        out[row.window_id] = scorer.score(feats, ep.episode_id)
    return pd.Series(out, name="classifier_p")


def stack_frame(
    tactile: pd.DataFrame, control: pd.DataFrame, classifier_p: pd.Series | None = None
) -> pd.DataFrame:
    """Join the two cells on window id with the label and stack features; raise on mismatch."""
    t = tactile.set_index("window_id") if "window_id" in tactile.columns else tactile
    c = control.set_index("window_id") if "window_id" in control.columns else control
    ids = t.index.intersection(c.index)
    if len(ids) != len(t) or len(ids) != len(c):
        raise ValueError("tactile and control cells cover different windows")
    t, c = t.loc[ids], c.loc[ids]
    if not (t["truth_target"] == c["truth_target"]).all():
        raise ValueError("truth targets differ between cells")
    frame = pd.DataFrame(
        {
            "episode_id": t["episode_id"],
            "object": t["object"]
            if "object" in t.columns
            else t["episode_id"].str.split("/").str[1],
            "label": _label(t["truth_target"]),
            "tactile_E": t["expected_score"].astype(float),
            "control_E": c["expected_score"].astype(float),
        },
        index=ids,
    )
    p = t["force_classifier_p"] if "force_classifier_p" in t.columns else classifier_p
    if p is None:
        raise ValueError("no force classifier probability available")
    frame["classifier_logit"] = _logit(pd.Series(p).reindex(ids).to_numpy())
    dropped = int(frame[list(FEATURES)].isna().any(axis=1).sum())
    frame = frame.dropna(subset=list(FEATURES))
    frame.attrs["dropped"] = dropped
    return frame


def fit_stack(frame: pd.DataFrame, *, l2: float = 1.0, fit_split: str) -> StackModel:
    """Standardized L2 logistic regression (sklearn), deterministic."""
    from sklearn.linear_model import LogisticRegression

    X = frame[list(FEATURES)].to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    model = LogisticRegression(C=1.0 / l2, max_iter=1000)
    model.fit((X - mean) / std, frame["label"].to_numpy(dtype=int))
    ids = "".join(sorted(frame.index.astype(str)))
    return StackModel(
        features=list(FEATURES),
        mean=mean.tolist(),
        std=std.tolist(),
        weights=np.asarray(model.coef_)[0].tolist(),
        bias=float(np.asarray(model.intercept_)[0]),
        l2=l2,
        fit_split=fit_split,
        n_fit=int(len(frame)),
        fit_window_ids_sha256=hashlib.sha256(ids.encode()).hexdigest(),
    )


def predict(model: StackModel, frame: pd.DataFrame) -> np.ndarray:
    """P(success) of the stack for every row of ``frame``."""
    X = frame[model.features].to_numpy(dtype=float)
    z = ((X - np.asarray(model.mean)) / np.asarray(model.std)) @ np.asarray(model.weights)
    return 1.0 / (1.0 + np.exp(-(z + model.bias)))


def oof_predict(frame: pd.DataFrame, *, n_folds: int = 5, l2: float = 1.0) -> np.ndarray:
    """Object-grouped out-of-fold P(success): fit on the other folds, score this one."""
    objects = sorted(
        frame["object"].unique(), key=lambda o: hashlib.sha256(str(o).encode()).hexdigest()
    )
    fold_of = {o: i % n_folds for i, o in enumerate(objects)}
    folds = frame["object"].map(fold_of).to_numpy()
    out = np.zeros(len(frame))
    for k in range(n_folds):
        train = frame[folds != k]
        if train["label"].nunique() < 2 or not (folds == k).any():
            out[folds == k] = train["label"].mean() if len(train) else 0.5
            continue
        out[folds == k] = predict(fit_stack(train, l2=l2, fit_split="oof"), frame[folds == k])
    return out


def stack_metrics(p: np.ndarray, label: Sequence[int]) -> dict[str, float]:
    """Accuracy at 0.5, AUROC, failure recall."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(label, dtype=int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    return {
        "accuracy": float((pred == y).mean()),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
        "failure_recall": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "n": float(len(y)),
    }


def load_source_cfg(base: Path) -> SourceConfig:
    """The HRDexDB source config of an experiment base."""
    return load_config(base / "config/sources/hrdexdb.yaml", SourceConfig)


__all__: list[str] = [
    "FEATURES",
    "StackModel",
    "classifier_probabilities",
    "fit_stack",
    "load_source_cfg",
    "oof_predict",
    "predict",
    "stack_frame",
    "stack_metrics",
]
