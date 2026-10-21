from pathlib import Path

import pytest

from tactile_flora.config import ForceMode
from tactile_flora.train.dataset import (
    IGNORE_INDEX,
    answer_labels,
    build_samples,
    estimate_wallclock,
    token_histogram,
)
from tests.fixtures.train_data import StubTokenizer, windows_source, write_manifest
from tests.fixtures.verify_ctx import FORCE_CFG


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


def test_answer_labels_rejects_non_prefix() -> None:
    assert answer_labels([1, 2], [1, 2, 3]) == [IGNORE_INDEX, IGNORE_INDEX, 3]
    with pytest.raises(ValueError):
        answer_labels([1, 2], [1, 9, 3])
    with pytest.raises(ValueError):
        answer_labels([1, 2], [1, 2])
