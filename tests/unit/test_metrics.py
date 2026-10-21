import math

import pandas as pd
import pytest

from tactile_flora.eval import metrics as M


def dense(q: list[int], ep: str = "e") -> pd.DataFrame:
    return pd.DataFrame({"episode_id": ep, "t": range(len(q)), "quality": q})


# truth classes: 1 at t3,t4; 2 at t2; 3 at t0,t1,t5. pred disagrees at t1 and t4.
TRUTH = dense([3, 3, 2, 1, 1, 3])
PRED = dense([3, 2, 2, 1, 3, 3])


def test_dense_metrics_hand_computed() -> None:
    # class F1: class1 tp1 fp0 fn1 -> 2/3; class2 tp1 fp1 fn0 -> 2/3; class3 tp2 fp1 fn1 -> 2/3
    assert M.macro_f1(PRED, TRUTH) == pytest.approx(2 / 3)
    # recalls 1/2, 1, 2/3
    assert M.balanced_accuracy(PRED, TRUTH) == pytest.approx((0.5 + 1 + 2 / 3) / 3)
    # non-normal: truth {2,3,4}, pred {1,2,3}: tp2 fp1 fn1
    assert M.bad_action_f1(PRED, TRUTH) == pytest.approx(2 / 3)
    assert M.failure_recall(PRED, TRUTH) == pytest.approx(0.5)


def test_dense_alignment_by_key_and_pooling() -> None:
    shuffled = PRED.sample(frac=1.0, random_state=1)
    assert M.failure_recall(shuffled, TRUTH) == pytest.approx(0.5)
    p = pd.concat([PRED, dense([1, 1], "f")], ignore_index=True)
    t = pd.concat([TRUTH, dense([1, 3], "f")], ignore_index=True)
    assert M.failure_recall(p, t) == pytest.approx(2 / 3)  # pooled: 2 of 3 failures hit


def test_undefined_metrics_are_nan() -> None:
    assert math.isnan(M.failure_recall(dense([3, 3]), dense([3, 3])))
    assert math.isnan(M.bad_action_f1(dense([3, 3]), dense([3, 3])))
    with pytest.raises(ValueError):
        M.macro_f1(dense([3]), dense([3, 3]).rename(columns={"t": "u"}).drop(columns="episode_id"))


def outcome_pred(scores: list[float], es: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"episode_id": list("abcd")[: len(scores)], "score": scores, "expected_score": es}
    )


def test_outcome_mae_accuracy_auroc() -> None:
    pred = outcome_pred([5, 4, 2], [4.5, 3.0, 2.0])
    truth = pd.DataFrame(
        {
            "episode_id": list("abc"),
            "score": [5, 2, 1],
            "label": [1, 1, 0],
            "object": ["x", "x", "y"],
        }
    )
    assert M.outcome_mae(pred, truth) == pytest.approx(100.0)  # (0 + 2 + 1) / 3 * 100
    assert M.outcome_accuracy(pred, truth) == pytest.approx(2 / 3)  # E >= 4 -> 1,0,0 vs 1,1,0
    assert M.per_object(M.outcome_mae, pred, truth, M.OUTCOME_KEYS) == {
        "x": pytest.approx(100.0),
        "y": pytest.approx(100.0),
    }


def test_auroc_ties_and_degenerate() -> None:
    pred = outcome_pred([5, 5, 1, 1], [4.5, 3.0, 2.0, 3.0])
    truth = pd.DataFrame({"episode_id": list("abcd"), "score": [5, 5, 1, 1], "label": [1, 1, 0, 0]})
    # positives {4.5, 3.0}, negatives {2.0, 3.0}: 1 + 1 + 1 + 0.5 over 4 pairs
    assert M.auroc(pred, truth) == pytest.approx(0.875)
    truth["label"] = 1
    assert math.isnan(M.auroc(pred, truth))


def test_mae_drops_unparsed_and_counts_failures() -> None:
    pred = outcome_pred([5, math.nan], [5.0, math.nan]).assign(parse_failed=[False, True])
    truth = pd.DataFrame({"episode_id": ["a", "b"], "score": [4, 1], "label": [1, 0]})
    assert M.outcome_mae(pred, truth) == pytest.approx(100.0)
    assert M.parse_failed_count(pred) == 1


def test_compute_metrics_dispatch() -> None:
    out = M.compute_metrics(["failure_recall", "mae"], dense=(PRED, TRUTH), outcome=None)
    assert out == {"failure_recall": pytest.approx(0.5)}
    with pytest.raises(KeyError):
        M.compute_metrics(["nope"], dense=(PRED, TRUTH))


def test_success_threshold_is_shared() -> None:
    assert M.SUCCESS_THRESHOLD == 4.0 and M.OUTCOME_PARSE_FAIL_SCORE == 3
    pred = outcome_pred([4], [4.0])
    truth = pd.DataFrame({"episode_id": ["a"], "score": [4], "label": [1]})
    assert M.outcome_accuracy(pred, truth) == 1.0  # 4.0 counts as success on both sides


def test_over_compression_recall() -> None:
    truth = pd.concat([dense([1, 1, 3], "a"), dense([1, 3, 3], "b")], ignore_index=True)
    truth["over_compression"] = truth["episode_id"] == "a"
    pred = pd.concat([dense([1, 3, 3], "a"), dense([3, 3, 3], "b")], ignore_index=True)
    assert M.over_compression_recall(pred, truth) == pytest.approx(0.5)  # b's miss is excluded
    assert M.failure_recall(pred, truth) == pytest.approx(1 / 3)
    assert math.isnan(M.over_compression_recall(pred, truth.drop(columns="over_compression")))


def test_mae_vs_dsr() -> None:
    pred = outcome_pred([5, 1, 3], [5.0, 1.0, 3.0])
    truth = pd.DataFrame(
        {
            "episode_id": list("abc"),
            "score": [5, 5, 5],
            "label": [1, 1, 1],
            "dsr": [1.0, 1.0, math.nan],
        }
    )
    # (E-1)/4 = 1, 0, .5 vs dsr 1, 1, NaN (dropped): (0 + 1) / 2 * 100
    assert M.mae_vs_dsr(pred, truth) == pytest.approx(50.0)
    assert M.compute_metrics(["mae_vs_dsr"], outcome=(pred, truth)) == {"mae_vs_dsr": 50.0}
