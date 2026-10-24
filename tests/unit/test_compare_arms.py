"""Statistical comparisons require identical complete populations and unchanged truth."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")
from writeup.hrdexdb.compare_arms import align, load, paired  # noqa: E402


def cell(tmp_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "window_id": ["a", "b", "c", "d"],
            "source": ["hrdexdb"] * 4,
            "episode_id": ["one/0", "one/1", "two/0", "two/1"],
            "object": ["one", "one", "two", "two"],
            "truth_target": ["Score: 1", "Score: 5"] * 2,
            "expected_score": [1.5, 4.5, 2.0, 4.0],
        }
    )
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path)
    loaded = load(path)
    assert loaded is not None
    return loaded


def test_alignment_preserves_all_windows_and_reorders(tmp_path: Path) -> None:
    data = cell(tmp_path)
    a, b = align(data, data.iloc[::-1])
    pd.testing.assert_frame_equal(a, b)
    result = paired(data, b, resamples=20)
    assert "wins 0 losses 0 McNemar p=1.000" in result
    assert "acc diff +0.000" in result
    assert "AUROC diff +0.000" in result


@pytest.mark.parametrize("change", ["missing", "duplicate", "truth", "object", "episode"])
def test_mismatched_pairs_are_refused(tmp_path: Path, change: str) -> None:
    data = cell(tmp_path)
    other = data.copy()
    if change == "missing":
        other = other.iloc[:-1]
    elif change == "duplicate":
        other = pd.concat([other, other.iloc[:1]])
    else:
        column = {"truth": "truth_target", "object": "object", "episode": "episode_id"}[change]
        other.loc["a", column] = "changed"
    with pytest.raises(ValueError):
        paired(data, other, resamples=20)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 0.0, 6.0])
def test_invalid_scores_are_not_silently_dropped(tmp_path: Path, score: float) -> None:
    data = cell(tmp_path).reset_index()
    data.loc[0, "expected_score"] = score
    path = tmp_path / "bad.parquet"
    data.to_parquet(path)
    with pytest.raises(ValueError, match="expected scores"):
        load(path)
