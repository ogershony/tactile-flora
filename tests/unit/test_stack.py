"""Tests for the stacked head over VLM scores and the force classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tactile_flora.eval import stack


def _cells(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    ids = [f"w{i:03d}" for i in range(n)]
    objects = [f"obj{i % 12}" for i in range(n)]
    tactile_e = rng.uniform(1, 5, n)
    control_e = np.clip(tactile_e + rng.normal(0, 1.0, n), 1, 5)
    p_force = rng.uniform(0.05, 0.95, n)
    logit = np.log(p_force / (1 - p_force))
    z = 2.0 * (tactile_e - 3.0) + 1.5 * logit
    label = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z))).astype(int)
    truth = [f"Rationale: no rationale\n\nScore: {5 if y else 1}" for y in label]

    def cell(e: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window_id": ids,
                "episode_id": [f"hand/{o}/{i}" for i, o in enumerate(objects)],
                "object": objects,
                "expected_score": e,
                "truth_target": truth,
            }
        )

    return cell(tactile_e), cell(control_e), pd.Series(p_force, index=ids)


def test_stack_fits_predicts_and_round_trips() -> None:
    tactile, control, p = _cells()
    frame = stack.stack_frame(tactile, control, p)
    assert len(frame) == 300 and frame.attrs["dropped"] == 0
    model = stack.fit_stack(frame, fit_split="val")
    assert model.weights[0] > 0 and model.weights[2] > 0  # tactile score and classifier logit
    in_sample = stack.stack_metrics(stack.predict(model, frame), frame["label"].to_numpy())
    oof = stack.stack_metrics(stack.oof_predict(frame), frame["label"].to_numpy())
    majority = max(frame["label"].mean(), 1 - frame["label"].mean())
    assert oof["accuracy"] > majority and oof["accuracy"] <= in_sample["accuracy"] + 0.05
    assert 0 < in_sample["auroc"] <= 1
    again = stack.StackModel.model_validate_json(model.model_dump_json())
    np.testing.assert_allclose(stack.predict(again, frame), stack.predict(model, frame))


def test_stack_frame_rejects_mismatches_and_prefers_the_column() -> None:
    tactile, control, p = _cells(40)
    frame = stack.stack_frame(tactile.assign(force_classifier_p=0.5), control, p)
    assert np.allclose(frame["classifier_logit"], 0.0)
    with pytest.raises(ValueError, match="different windows"):
        stack.stack_frame(tactile.iloc[:-1], control, p)
    bad = control.copy()
    bad.loc[0, "truth_target"] = "Rationale: no rationale\n\nScore: 3"
    with pytest.raises(ValueError, match="truth targets differ"):
        stack.stack_frame(tactile, bad, p)
    with pytest.raises(ValueError, match="no force classifier"):
        stack.stack_frame(tactile, control, None)
    nan = tactile.copy()
    nan.loc[0, "expected_score"] = np.nan
    assert stack.stack_frame(nan, control, p).attrs["dropped"] == 1
