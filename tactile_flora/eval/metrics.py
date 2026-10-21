"""Metric functions (MAE, macro F1, balanced accuracy, bad-action F1, failure recall, AUROC).

Owns: the paper's evaluation definitions as pure functions of two aligned DataFrames
(``pred``, ``truth``), the per-object breakdown helper and the dispatcher compute_metrics.
Must never: run models, read files, or know about manifests.

Frame schemas (rows are aligned by an inner join on the key columns when both frames carry
them, otherwise positionally, which then requires equal lengths):
- dense: key ``episode_id`` (the harness uses ``source/episode_id``), ``t`` (1 Hz step); value
  ``quality`` in {1, 2, 3}. Pool all steps of all episodes (the paper pools over steps).
- outcome: key ``episode_id``; ``score`` (integer 1-5; the harness replaces unparsable answers
  by 3, the middle of the scale, so they stay in MAE; NaN scores are still dropped); pred
  ``expected_score`` (E[s]); truth ``label`` (binary, 1 = success) for accuracy / AUROC. Truth
  may carry ``object`` for the per-object breakdown.
SoftVTBench extras: dense truth may carry a boolean ``over_compression`` column (the episode has
over-compression event spans) and outcome truth a ``dsr`` column (1 if success and the peak
deformation is within tolerance, else 0, NaN when unknown); see ``over_compression_recall`` and
``mae_vs_dsr``.
Undefined metrics (an empty class) are NaN; metrics.json stores them as null.
Macro F1 averages over the classes present in truth or prediction; balanced accuracy over the
classes present in truth.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

DENSE_KEYS = ("episode_id", "t")
OUTCOME_KEYS = ("episode_id",)
SCORE_SCALE = 100.0  # paper reports MAE on the 1-5 scale times 100
# One success cut for both sides: a truth score >= this is a success (binary label) and a
# predicted E[s] >= this is a predicted success (accuracy).
SUCCESS_THRESHOLD = 4.0
# Unparsable outcome answers are replaced by the middle of the 1-5 scale (the outcome analogue
# of the dense all-normal replacement) and counted in parse_failed.
OUTCOME_PARSE_FAIL_SCORE = 3
OVERCOMPRESS_RATIONALE = "excessive force on the object"
DENSE_METRICS = (
    "macro_f1",
    "balanced_acc",
    "bad_action_f1",
    "failure_recall",
    "over_compression_recall",
)
OUTCOME_METRICS = ("mae", "accuracy", "auroc", "mae_vs_dsr")

Metric = Callable[[pd.DataFrame, pd.DataFrame], float]


def _align(
    pred: pd.DataFrame, truth: pd.DataFrame, keys: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all(k in pred.columns and k in truth.columns for k in keys):
        merged = pred.merge(truth, on=list(keys), how="inner", suffixes=("_p", "_t"))
        p = pd.DataFrame({c: merged[c + "_p" if c + "_p" in merged else c] for c in pred.columns})
        t = pd.DataFrame({c: merged[c + "_t" if c + "_t" in merged else c] for c in truth.columns})
        return p, t
    if len(pred) != len(truth):
        raise ValueError("pred and truth must have equal length when no key columns are shared")
    return pred.reset_index(drop=True), truth.reset_index(drop=True)


def _quality(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p, t = _align(pred, truth, DENSE_KEYS)
    return p["quality"].to_numpy(dtype=int), t["quality"].to_numpy(dtype=int)


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else math.nan


def macro_f1(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Unweighted mean of per-class F1 over the dense classes present in pred or truth."""
    p, t = _quality(pred, truth)
    scores: list[float] = []
    for c in sorted(set(p.tolist()) | set(t.tolist())):
        scores.append(_f1(int(((p == c) & (t == c)).sum()), int(((p == c) & (t != c)).sum()),
                          int(((p != c) & (t == c)).sum())))  # fmt: skip
    return float(np.mean(scores)) if scores else math.nan


def balanced_accuracy(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Mean per-class recall over the dense classes present in truth."""
    p, t = _quality(pred, truth)
    recalls = [float((p[t == c] == c).mean()) for c in sorted(set(t.tolist()))]
    return float(np.mean(recalls)) if recalls else math.nan


def bad_action_f1(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """F1 of the non-normal class (quality 1 or 2 is positive)."""
    p, t = _quality(pred, truth)
    pb, tb = p != 3, t != 3
    return _f1(int((pb & tb).sum()), int((pb & ~tb).sum()), int((~pb & tb).sum()))


def failure_recall(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Recall of class 1 (failure): predicted 1 among truth 1; NaN without any truth failure."""
    p, t = _quality(pred, truth)
    return float((p[t == 1] == 1).mean()) if (t == 1).any() else math.nan


def over_compression_recall(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Failure recall restricted to episodes whose truth carries over-compression spans.

    Needs a boolean ``over_compression`` column on ``truth`` (per step, constant per episode);
    NaN without it or without any failure step in those episodes.
    """
    p, t = _align(pred, truth, DENSE_KEYS)
    if "over_compression" not in t:
        return math.nan
    keep = t["over_compression"].astype(bool).to_numpy()
    pq, tq = p["quality"].to_numpy(dtype=int)[keep], t["quality"].to_numpy(dtype=int)[keep]
    return float((pq[tq == 1] == 1).mean()) if (tq == 1).any() else math.nan


def mae_vs_dsr(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """MAE x 100 between (E[s] - 1) / 4 (score mapped to [0, 1]) and the DSR in ``truth["dsr"]``.

    Rows with an unknown DSR or E[s] are dropped; NaN when none remain.
    """
    p, t = _align(pred, truth, OUTCOME_KEYS)
    if "dsr" not in t:
        return math.nan
    err = (((p["expected_score"].astype(float) - 1.0) / 4.0) - t["dsr"].astype(float)).abs()
    err = err.dropna()
    return float(err.mean() * SCORE_SCALE) if len(err) else math.nan


def outcome_mae(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Mean |score - truth score| on the 1-5 scale times 100 (NaN scores are dropped)."""
    p, t = _align(pred, truth, OUTCOME_KEYS)
    err = (p["score"].astype(float) - t["score"].astype(float)).abs().dropna()
    return float(err.mean() * SCORE_SCALE) if len(err) else math.nan


def outcome_accuracy(
    pred: pd.DataFrame, truth: pd.DataFrame, threshold: float = SUCCESS_THRESHOLD
) -> float:
    """Accuracy of ``expected_score >= threshold`` against the binary truth ``label``."""
    p, t = _align(pred, truth, OUTCOME_KEYS)
    ok = p["expected_score"].notna()
    if not ok.any():
        return math.nan
    hit = (p.loc[ok, "expected_score"] >= threshold).astype(int) == t.loc[ok, "label"].astype(int)
    return float(hit.mean())


def auroc(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """AUROC of ``expected_score`` against the binary ``label`` (Mann-Whitney rank statistic).

    Ties get average ranks. NaN when one class is empty.
    """
    p, t = _align(pred, truth, OUTCOME_KEYS)
    ok = p["expected_score"].notna()
    s = p.loc[ok, "expected_score"].astype(float).reset_index(drop=True)
    y = t.loc[ok, "label"].astype(int).reset_index(drop=True)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = s.rank(method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def parse_failed_count(pred: pd.DataFrame) -> int:
    """Count the rows with ``parse_failed`` true."""
    return int(pred["parse_failed"].astype(bool).sum()) if "parse_failed" in pred else 0


METRICS: dict[str, Metric] = {
    "mae": outcome_mae,
    "macro_f1": macro_f1,
    "balanced_acc": balanced_accuracy,
    "bad_action_f1": bad_action_f1,
    "failure_recall": failure_recall,
    "accuracy": outcome_accuracy,
    "auroc": auroc,
    "over_compression_recall": over_compression_recall,
    "mae_vs_dsr": mae_vs_dsr,
}


def per_object(
    metric: Metric, pred: pd.DataFrame, truth: pd.DataFrame, key: Sequence[str]
) -> dict[str, float]:
    """Apply ``metric`` to each object's rows; ``truth`` must carry an ``object`` column.

    ``key`` are the join columns of the metric's frame kind (DENSE_KEYS or OUTCOME_KEYS).
    """
    p, t = _align(pred, truth, key)
    out: dict[str, float] = {}
    for obj, idx in t.groupby("object").groups.items():
        out[str(obj)] = metric(p.loc[idx].reset_index(drop=True), t.loc[idx].reset_index(drop=True))
    return out


def compute_metrics(
    names: Sequence[str],
    dense: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    outcome: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """Compute the named metrics from the dense and/or outcome (pred, truth) pairs.

    Dense metrics need ``dense``, outcome metrics ``outcome``; a metric whose frames are absent
    is skipped. Unknown names raise KeyError.
    """
    out: dict[str, float] = {}
    for name in names:
        fn = METRICS[name]
        pair = dense if name in DENSE_METRICS else outcome
        if pair is not None:
            out[name] = fn(*pair)
    return out
