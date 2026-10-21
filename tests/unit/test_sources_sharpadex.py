"""Tests for the sharpadex adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.sharpadex import SharpadexAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "sharpadex"
EID = "pick_cup/season_7_train/0"


def _cfg() -> SourceConfig:
    return SourceConfig(
        name="sharpadex",
        root=str(ROOT),
        license_allowed=["CC-BY-NC-4.0"],
        max_windows=600,
        max_episodes=200,
        eval_only=False,
        expected_label_rate=0.3,
        seed=0,
    )


def test_discover_finds_fixture() -> None:
    assert SharpadexAdapter(_cfg()).discover(ROOT) == [EID]


def test_load_on_episode_clock() -> None:
    a = SharpadexAdapter(_cfg())
    a.discover(ROOT)
    ep = a.load(EID)
    assert ep.session == "7" and ep.license == "CC-BY-NC-4.0" and ep.meta["season"] == "7"
    assert ep.force.channel == "pad" and set(ep.force.arms) == {"R"} and ep.force.rate_hz == 30
    assert ep.force.t[0] == 0.0 and ep.meta["clock_offset_s"] == 0.0
    arm = ep.force.arms["R"]
    assert arm.kind == "pad"
    assert max(arm.f_a) == pytest.approx(5.0) and max(arm.f_b) == pytest.approx(2.5)
    assert ep.gripper is not None
    g = ep.gripper.arms["R"]
    assert min(g) == 0.0 and max(g) == pytest.approx(1.0) and ep.meta["closure_state_index"] == 40
    assert set(ep.cameras) == {"scene", "wrist_left", "wrist_right"}
    assert ep.cameras["scene"].fps == 30 and ep.cameras["scene"].n_frames == 10
    assert ep.cameras["scene"].path.endswith(".mp4")
    assert ep.duration_s == pytest.approx(9 / 30)


def test_subtasks_and_outcome_words() -> None:
    a = SharpadexAdapter(_cfg())
    a.discover(ROOT)
    ep = a.load(EID)
    assert ep.task == "pick up the cup, on"
    assert ep.meta["subtasks"] == [
        {"start_step": 0, "end_step": 4, "text": "reach the cup", "task_skill": "reach"},
        {"start_step": 5, "end_step": 9, "text": "grasp the cup", "task_skill": "grasp"},
    ]


def test_splits_by_season_and_no_labels() -> None:
    a = SharpadexAdapter(_cfg())
    a.discover(ROOT)
    s = a.splits()
    assert sum(len(v) for v in s.values()) == 1 and set(s) == {"train", "val", "test"}
    assert a.native_labels(EID) == []
