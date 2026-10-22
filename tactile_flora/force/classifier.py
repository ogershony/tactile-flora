"""A small force-only outcome classifier whose probability can be shown in the force block.

Owns: FEATURE_NAMES, extract_features (pure function of normalized ResampledForce), a
JSON-portable gradient-boosted tree ensemble (``TreeEnsemble``) evaluated in numpy, fit_cv
(sklearn, scripts only) and ``ForceScorer`` (full model plus per-fold models for out-of-fold
scores on training episodes).
Must never: read labels at prediction time, import sklearn at module import time, or render.

Why: HRDexDB analysis (2026-10-22) found the outcome signal in force is trial-relative and
nonlinear (object-grouped CV AUROC about 0.78 for boosted trees against about 0.55-0.72 for
a linear model), while the stacked video-plus-force score lifted development accuracy from
83% to 88%. A VLM fine-tuned on 350 clips is unlikely to learn that combination from 64 raw
lines, so the probability is written into the summary line. Training episodes get the score
of the fold model that never saw them, so the fine-tuned model learns how far to trust it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from tactile_flora.schema import ResampledForce

FEATURE_NAMES: tuple[str, ...] = (
    "contacts",
    "longest_s",
    "total_s",
    "peak",
    "hold_s",
    "hold_median",
    "late_early",
    "low_frac",
    "f_mean",
    "f_std",
    "f_p25",
    "f_p75",
    "f_p90",
    "contact_frac",
    "hold_std_rel",
    "hold_slope",
    "hold_df_max",
    "hold_df_min",
    "hold_df_big",
    "g_max",
    "g_mean",
    "g_cycles",
    "g_at_peak",
    "t_peak_frac",
    "d_hold_mean",
    "d_hold_absmean",
    "d_hold_std",
    "raw_hold_std_rel",
    "raw_hold_jump_max",
    "raw_hold_jumps",
    "span_frac",
)


def _runs(flags: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def extract_features(features: ResampledForce, arm: str = "R") -> np.ndarray:
    """Feature vector (FEATURE_NAMES order) for one arm of normalized features.

    An absent arm or unnormalized features raise. Frame-grid statistics mirror
    ``render.summary_lines``; raw-slice statistics use ``raw_groups`` (full-rate samples per
    0.5 s group) within the hold phase and are zero when no raw groups are present.
    """
    af = features.arms.get(arm)  # type: ignore[arg-type]
    if af is None or af.c is None:
        raise ValueError(f"arm {arm} missing or not normalized")
    times = np.asarray(features.frame_times, dtype=np.float64)
    n = len(times)
    step = float(np.median(np.diff(times))) if n > 1 else 0.0
    F = np.nan_to_num(np.asarray(af.F, dtype=np.float64))
    dF = np.nan_to_num(np.asarray(af.dF, dtype=np.float64))
    D = np.nan_to_num(np.asarray(af.delta if af.delta is not None else [0.0] * n))
    c = [int(v) for v in af.c]
    runs = _runs(c)
    seconds = [times[b] - times[a] + step for a, b in runs]
    if af.g is not None and max(af.g) > 0:
        g = np.asarray(af.g, dtype=np.float64)
        hold = g >= 0.8 * float(g.max())
    else:
        g = np.zeros(n)
        hold = np.asarray(c, dtype=bool)
    idx = np.flatnonzero(hold)
    peak_i = int(F.argmax()) if n else 0
    out: dict[str, float] = {
        "contacts": float(len(runs)),
        "longest_s": float(max(seconds, default=0.0)),
        "total_s": float(sum(seconds)),
        "peak": float(F.max()) if n else 0.0,
        "f_mean": float(F.mean()) if n else 0.0,
        "f_std": float(F.std()) if n else 0.0,
        "f_p25": float(np.percentile(F, 25)) if n else 0.0,
        "f_p75": float(np.percentile(F, 75)) if n else 0.0,
        "f_p90": float(np.percentile(F, 90)) if n else 0.0,
        "contact_frac": float(np.mean(c)) if n else 0.0,
        "g_max": float(g.max()) if n else 0.0,
        "g_mean": float(g.mean()) if n else 0.0,
        "g_at_peak": float(g[peak_i]) if n else 0.0,
        "t_peak_frac": float(peak_i / max(n - 1, 1)),
        "span_frac": float((times[-1] - times[0]) / max(times[-1], 1e-9)) if n > 1 else 0.0,
    }
    # closure cycles: rises above 80% of max after having dropped below 50% of max
    cycles, armed = 0, True
    gmax = float(g.max()) if n else 0.0
    for v in g:
        if armed and gmax > 0 and v >= 0.8 * gmax:
            cycles += 1
            armed = False
        elif not armed and gmax > 0 and v < 0.5 * gmax:
            armed = True
    out["g_cycles"] = float(cycles)
    if idx.size:
        hf = F[idx]
        median = float(np.median(hf))
        q = max(1, idx.size // 4)
        early = float(np.median(hf[:q]))
        out["hold_s"] = float(times[idx[-1]] - times[idx[0]] + step)
        out["hold_median"] = median
        out["late_early"] = float(np.median(hf[-q:]) / early) if early > 0 else 0.0
        out["low_frac"] = float(np.mean(hf < 0.2 * median)) if median > 0 else 1.0
        out["hold_std_rel"] = float(hf.std() / median) if median > 0 else 0.0
        out["hold_slope"] = (
            float(np.polyfit(times[idx], hf, 1)[0]) if idx.size > 2 and step > 0 else 0.0
        )
        hd = dF[idx]
        out["hold_df_max"] = float(hd.max())
        out["hold_df_min"] = float(hd.min())
        out["hold_df_big"] = float(np.sum(np.abs(hd) > 1.0))
        dd = D[idx]
        out["d_hold_mean"] = float(dd.mean())
        out["d_hold_absmean"] = float(np.abs(dd).mean())
        out["d_hold_std"] = float(dd.std())
        raw = [
            v
            for k in sorted({int(i) // 2 for i in idx})
            if k < len(af.raw_groups)
            for v in af.raw_groups[k]
            if not math.isnan(v)
        ]
        if len(raw) > 2 and median > 0:
            arr = np.asarray(raw, dtype=np.float64)
            jumps = np.abs(np.diff(arr))
            scale = max(float(np.median(arr)), 1e-9)
            out["raw_hold_std_rel"] = float(arr.std() / scale)
            out["raw_hold_jump_max"] = float(jumps.max() / scale)
            out["raw_hold_jumps"] = float(np.sum(jumps > 0.25 * scale))
        else:
            out["raw_hold_std_rel"] = out["raw_hold_jump_max"] = out["raw_hold_jumps"] = 0.0
    else:
        for k in (
            "hold_s",
            "hold_median",
            "late_early",
            "hold_std_rel",
            "hold_slope",
            "hold_df_max",
            "hold_df_min",
            "hold_df_big",
            "d_hold_mean",
            "d_hold_absmean",
            "d_hold_std",
            "raw_hold_std_rel",
            "raw_hold_jump_max",
            "raw_hold_jumps",
        ):
            out[k] = 0.0
        out["low_frac"] = 1.0
    return np.asarray([out[k] for k in FEATURE_NAMES], dtype=np.float64)


class Tree(BaseModel):
    """One regression tree in sklearn's array layout (children -1 at leaves)."""

    model_config = ConfigDict(extra="forbid")

    feature: list[int]
    threshold: list[float]
    left: list[int]
    right: list[int]
    value: list[float]

    def predict(self, x: np.ndarray) -> float:
        """Leaf value for feature vector ``x``."""
        node = 0
        while self.left[node] != -1:
            node = (
                self.left[node]
                if x[self.feature[node]] <= self.threshold[node]
                else self.right[node]
            )
        return self.value[node]


class TreeEnsemble(BaseModel):
    """Binary gradient-boosted trees: p = sigmoid(prior + rate * sum of tree values)."""

    model_config = ConfigDict(extra="forbid")

    feature_names: list[str]
    prior: float
    learning_rate: float
    trees: list[Tree]

    def predict_proba(self, x: np.ndarray) -> float:
        """Probability of success for one feature vector."""
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise ValueError("classifier features differ from FEATURE_NAMES")
        raw = self.prior + self.learning_rate * sum(t.predict(x) for t in self.trees)
        return float(1.0 / (1.0 + math.exp(-raw)))


def export_sklearn(model: Any) -> TreeEnsemble:
    """Convert a fitted sklearn GradientBoostingClassifier (binary, log loss) to JSON form."""
    init = model.init_
    prob = float(init.predict_proba(np.zeros((1, len(FEATURE_NAMES))))[0, 1])
    prior = math.log(prob / (1.0 - prob))
    trees: list[Tree] = []
    for stage in model.estimators_:
        t = stage[0].tree_
        trees.append(
            Tree(
                feature=[int(v) for v in t.feature],
                threshold=[float(v) for v in t.threshold],
                left=[int(v) for v in t.children_left],
                right=[int(v) for v in t.children_right],
                value=[float(v) for v in t.value.reshape(-1)],
            )
        )
    return TreeEnsemble(
        feature_names=list(FEATURE_NAMES),
        prior=prior,
        learning_rate=float(model.learning_rate),
        trees=trees,
    )


class ForceScorer(BaseModel):
    """Full model plus per-fold models; training episodes score with the fold that skipped them."""

    model_config = ConfigDict(extra="forbid")

    source: str
    arm: str = "R"
    model: TreeEnsemble
    fold_models: list[TreeEnsemble]
    fold_of_episode: dict[str, int]
    cv_metrics: dict[str, float]
    params: dict[str, Any]

    def score(self, features: ResampledForce, episode_id: str | None = None) -> float:
        """Success probability; out-of-fold when ``episode_id`` was a training episode."""
        x = extract_features(features, self.arm)
        if episode_id is not None and episode_id in self.fold_of_episode:
            return self.fold_models[self.fold_of_episode[episode_id]].predict_proba(x)
        return self.model.predict_proba(x)

    def save(self, path: Path | str) -> None:
        """Write the scorer as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=1) + "\n")


def load_scorer(path: Path | str) -> ForceScorer:
    """Read a scorer written by ``ForceScorer.save``."""
    return ForceScorer.model_validate_json(Path(path).read_text())


def fit_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    episode_ids: Sequence[str],
    *,
    source: str,
    n_folds: int = 5,
    params: dict[str, Any] | None = None,
) -> ForceScorer:
    """Fit the full model and ``n_folds`` object-grouped fold models (sklearn, scripts only).

    ``cv_metrics`` holds out-of-fold accuracy and AUROC at the 0.5 cut, which is what the
    fine-tuned model will see on its own training episodes.
    """
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-not-found]
    from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore[import-not-found]
    from sklearn.model_selection import GroupKFold  # type: ignore[import-not-found]

    hp = {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_samples_leaf": 5,
        "random_state": 0,
        **(params or {}),
    }
    oof = np.zeros(len(y))
    fold_models: list[TreeEnsemble] = []
    fold_of: dict[str, int] = {}
    for k, (trn, tst) in enumerate(GroupKFold(n_folds).split(X, y, groups)):
        m = GradientBoostingClassifier(**hp).fit(X[trn], y[trn])
        oof[tst] = m.predict_proba(X[tst])[:, 1]
        fold_models.append(export_sklearn(m))
        for i in tst:
            fold_of[str(episode_ids[i])] = k
    full = export_sklearn(GradientBoostingClassifier(**hp).fit(X, y))
    return ForceScorer(
        source=source,
        model=full,
        fold_models=fold_models,
        fold_of_episode=fold_of,
        cv_metrics={
            "oof_accuracy": float(accuracy_score(y, oof >= 0.5)),
            "oof_auroc": float(roc_auc_score(y, oof)),
            "n": float(len(y)),
            "positive_rate": float(np.mean(y)),
        },
        params={k: v for k, v in hp.items()},
    )


__all__ = [
    "FEATURE_NAMES",
    "ForceScorer",
    "Tree",
    "TreeEnsemble",
    "export_sklearn",
    "extract_features",
    "fit_cv",
    "load_scorer",
]
