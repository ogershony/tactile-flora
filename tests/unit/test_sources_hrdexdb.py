"""Tests for the hrdexdb adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.base import get_source
from tactile_flora.sources.hrdexdb import HrdexdbAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "hrdexdb" / "data"
EID = "allegro_v5/retry_mug/20260101_000000"


def _cfg() -> SourceConfig:
    return SourceConfig(
        name="hrdexdb",
        root=str(ROOT),
        license_allowed=["CC-BY-NC-4.0"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=0.5,
        seed=0,
    )


def test_registered_and_discover() -> None:
    assert get_source("hrdexdb") is HrdexdbAdapter
    assert HrdexdbAdapter(_cfg()).discover(ROOT) == [EID]


def test_load_on_episode_clock() -> None:
    ep = HrdexdbAdapter(_cfg()).load(EID)
    assert ep.session == "allegro_v5"
    assert ep.license == "CC-BY-NC-4.0"
    assert ep.force.channel == "pad"
    assert set(ep.force.arms) == {"R"}
    assert ep.meta["clock_offset_s"] == pytest.approx(100.0)
    assert ep.force.t[0] == pytest.approx(0.0)
    assert ep.meta["tactile_channels"] == [1, 3]
    arm = ep.force.arms["R"]
    assert arm.kind == "pad"
    assert arm.f_a[4] == 500 and arm.f_b[4] == 200
    assert ep.gripper is not None and set(ep.gripper.arms) == {"R"}
    g = ep.gripper.arms["R"]
    assert len(g) == len(ep.force.t) and min(g) == 0.0 and max(g) == 1.0
    assert list(ep.cameras) == ["scene"]
    assert ep.cameras["scene"].n_frames == 10
    assert ep.duration_s == pytest.approx(10 / ep.cameras["scene"].fps)


def test_outcome_words_stripped() -> None:
    ep = HrdexdbAdapter(_cfg()).load(EID)
    assert "retry" not in ep.task.lower()
    assert ep.task == "grasp and lift the mug"


def test_splits_and_labels() -> None:
    a = HrdexdbAdapter(_cfg())
    assert a.splits() == {"benchmark": [EID]}
    labels = a.native_labels(EID)
    assert len(labels) == 1
    assert labels[0].kind == "outcome" and labels[0].outcome == 5
    assert labels[0].origin == "source"


def test_unsupported_hands_raise(tmp_path: Path) -> None:
    for hand in ("inspire_f1", "robotiq_2f85", "human"):
        trial = tmp_path / hand / "obj" / "s1"
        trial.mkdir(parents=True)
        (trial / "grasp_result.json").write_text("{}")
    a = HrdexdbAdapter(_cfg())
    ids = a.discover(tmp_path)
    assert len(ids) == 3
    for eid in ids:
        with pytest.raises(NotImplementedError, match="not loadable"):
            a.load(eid)
    assert a.native_labels(ids[0]) == []
    assert np.isclose(1.0, 1.0)
