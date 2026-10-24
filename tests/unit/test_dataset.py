from pathlib import Path

import pytest

from tactile_flora.config import ForceMode
from tactile_flora.schema import ManifestRow
from tactile_flora.train.dataset import (
    DENSE_FAILURE_MAX,
    FAILURE_SCORE_MAX,
    IGNORE_INDEX,
    WindowItem,
    answer_labels,
    build_samples,
    estimate_wallclock,
    is_dense_failure_target,
    is_failure_target,
    token_histogram,
)
from tests.fixtures.train_data import StubTokenizer, windows_source, write_manifest
from tests.fixtures.verify_ctx import FORCE_CFG

DENSE_TARGET = "Worst: 1\nQuality:\n" + "1" * 32 + "\nRationales:\n1-32 crushed it"


def _build(tmp_path: Path, mode: ForceMode):
    mp = write_manifest(tmp_path / "m.parquet")
    return build_samples(
        mp, mode, FORCE_CFG, tmp_path, tokenizer=StubTokenizer(), windows_source=windows_source
    )


def test_answer_only_labels(tmp_path: Path) -> None:
    for s in _build(tmp_path, "real"):
        ans = [i for i, lab in enumerate(s.labels) if lab != IGNORE_INDEX]
        assert ans == list(range(len(s.labels) - len(ans), len(s.labels)))
        n_answer = len(s.target.split()) + 1  # words plus <eos>
        assert len(ans) == n_answer
        assert [s.labels[i] for i in ans] == [s.input_ids[i] for i in ans]
        assert len(s.input_ids) == len(s.labels) == s.n_tokens


def test_split_filter_and_sorted_by_length(tmp_path: Path) -> None:
    samples = _build(tmp_path, "real")
    assert len(samples) == 3  # the val row is excluded
    lengths = [s.n_tokens for s in samples]
    assert lengths == sorted(lengths) and len(set(lengths)) == 3


def test_force_differs_from_control_only_in_block(tmp_path: Path) -> None:
    force = {s.window_id: s for s in _build(tmp_path, "real")}
    control = {s.window_id: s for s in _build(tmp_path, "omitted")}
    assert force.keys() == control.keys()
    for wid, f in force.items():
        c = control[wid]
        assert f.force_block is not None and c.force_block is None
        fsys = f.messages[0]["content"][0]["text"]
        csys = c.messages[0]["content"][0]["text"]
        assert fsys.startswith(csys) and fsys != csys
        fafter = f.messages[1]["content"][2]["text"]
        cafter = c.messages[1]["content"][2]["text"]
        assert fafter == f"\n{f.force_block}{cafter}"
        assert f.messages[1]["content"][:2] == c.messages[1]["content"][:2]
        assert f.target == c.target and f.n_tokens > c.n_tokens


def test_zero_mode_changes_block(tmp_path: Path) -> None:
    real = {s.window_id: s for s in _build(tmp_path, "real")}
    zero = {s.window_id: s for s in _build(tmp_path, "zero")}
    assert all(zero[k].force_block != real[k].force_block for k in real)


def test_missing_features_modes(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "m.parquet")

    def no_feats(row):  # type: ignore[no-untyped-def]
        return windows_source(row).model_copy(update={"features": None})

    with pytest.raises(ValueError, match="needs resampled features"):
        build_samples(
            mp, "zero", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(), windows_source=no_feats
        )
    out = build_samples(
        mp, "real", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(), windows_source=no_feats
    )
    assert all(s.force_block for s in out)


def test_histogram_and_wallclock(tmp_path: Path) -> None:
    samples = _build(tmp_path, "omitted")
    h = token_histogram(samples)
    n = [s.n_tokens for s in samples]
    assert h["min"] == min(n) and h["max"] == max(n) and h["total"] == sum(n)
    assert h["min"] <= h["p50"] <= h["p90"] <= h["max"]
    assert h["p50"] == n[1]
    assert estimate_wallclock(samples, 9.0) == 27.0
    assert token_histogram([])["total"] == 0.0


def test_is_failure_target_follows_the_outcome_grammar() -> None:
    assert FAILURE_SCORE_MAX == 2
    assert is_failure_target("Score: 1") and is_failure_target("Score: 2")
    assert is_failure_target("Rationale: slipped out of the gripper\n\nScore: 2")
    assert not is_failure_target("Score: 3") and not is_failure_target("Score: 5")
    assert not is_failure_target("Rationale: clean insert\n\nScore: 5")
    # dense answers and anything outside the grammar are never failures
    assert not is_failure_target(DENSE_TARGET)
    assert not is_failure_target("Score: ok 1") and not is_failure_target("")


def _retargeted(targets: dict[str, str], prompt_kind: str = "outcome"):  # type: ignore[no-untyped-def]
    """windows_source whose train windows carry ``targets[episode_id]`` (and prompt_kind)."""

    def src(row: ManifestRow) -> WindowItem:
        item = windows_source(row)
        update: dict[str, object] = {"target": targets.get(row.episode_id, item.window.target)}
        if prompt_kind == "dense":
            update["prompt_kind"] = "dense"
            update["frame_times"] = tuple(i * 0.25 for i in range(64))
        return item.model_copy(update={"window": item.window.model_copy(update=update)})

    return src


def test_failure_weight_marks_failed_outcomes_only(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "m.parquet")
    targets = {
        "pad-0": "Score: 1",
        "pad-1": "Rationale: dropped the part\n\nScore: 2",
        "pad-2": "Rationale: clean insert\n\nScore: 5",
    }
    out = build_samples(
        mp, "real", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=_retargeted(targets), failure_weight=2.2,
    )  # fmt: skip
    assert {s.target: s.weight for s in out} == {
        "Score: 1": 2.2,
        "Rationale: dropped the part\n\nScore: 2": 2.2,
        "Rationale: clean insert\n\nScore: 5": 1.0,
    }
    # the default leaves every sample at 1.0, failures included
    flat = build_samples(
        mp, "real", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=_retargeted(targets),
    )  # fmt: skip
    assert {s.weight for s in flat} == {1.0}
    # the weight changes nothing else about the sample
    for a, b in zip(out, flat, strict=True):
        assert a.model_dump(exclude={"weight"}) == b.model_dump(exclude={"weight"})


def test_failure_weight_leaves_dense_windows_alone(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "m.parquet")
    src = _retargeted({eid: DENSE_TARGET for eid in ("pad-0", "pad-1", "pad-2")}, "dense")
    out = build_samples(
        mp, "omitted", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=src, failure_weight=2.2,
    )  # fmt: skip
    assert len(out) == 3 and all(s.target == DENSE_TARGET for s in out)
    assert {s.weight for s in out} == {1.0}


NORMAL_DENSE = "Worst: 3\nQuality:\n" + "3" * 32 + "\nRationales:"
SUBOPTIMAL_DENSE = "Worst: 2\nQuality:\n" + "3" * 30 + "22" + "\nRationales:\n31-32 hesitated"


def test_is_dense_failure_target_follows_the_dense_grammar() -> None:
    assert DENSE_FAILURE_MAX == 2
    assert is_dense_failure_target(DENSE_TARGET) and is_dense_failure_target(SUBOPTIMAL_DENSE)
    assert not is_dense_failure_target(NORMAL_DENSE)
    # outcome answers and a Worst that disagrees with the digits are never dense failures
    assert not is_dense_failure_target("Score: 1")
    assert not is_dense_failure_target("Worst: 1\nQuality:\n" + "3" * 32 + "\nRationales:")


def test_dense_failure_weight_marks_non_normal_dense_windows_only(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "m.parquet")
    targets = {"pad-0": DENSE_TARGET, "pad-1": SUBOPTIMAL_DENSE, "pad-2": NORMAL_DENSE}
    out = build_samples(
        mp, "omitted", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=_retargeted(targets, "dense"), failure_weight=2.2, dense_failure_weight=3.58,
    )  # fmt: skip
    assert {s.target: s.weight for s in out} == {
        DENSE_TARGET: 3.58,
        SUBOPTIMAL_DENSE: 3.58,
        NORMAL_DENSE: 1.0,
    }
    # outcome failures keep failure_weight, untouched by the dense weight
    outcome = build_samples(
        mp, "real", FORCE_CFG, tmp_path, tokenizer=StubTokenizer(),
        windows_source=_retargeted({"pad-0": "Score: 1"}), dense_failure_weight=3.58,
    )  # fmt: skip
    assert {s.target: s.weight for s in outcome}["Score: 1"] == 1.0


def test_answer_labels_rejects_non_prefix() -> None:
    assert answer_labels([1, 2], [1, 2, 3]) == [IGNORE_INDEX, IGNORE_INDEX, 3]
    with pytest.raises(ValueError):
        answer_labels([1, 2], [1, 9, 3])
    with pytest.raises(ValueError):
        answer_labels([1, 2], [1, 2])
