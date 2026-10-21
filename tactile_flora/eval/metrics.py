"""Metric functions (MAE, macro F1, balanced accuracy, bad-action F1, failure recall, AUROC).

Owns: metric functions (MAE, macro F1, balanced accuracy, bad-action F1, failure recall, AUROC).
Must never: running models or reading files.
"""

from __future__ import annotations

import pandas as pd


def compute_metrics(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Public entry point of this module (milestone 10)."""
    raise NotImplementedError("milestone 10")
