"""Tests for the rh20t adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.rh20t import Rh20tAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "rh20t"
EID = "rh20t_cfg3/0"


def _cfg() -> SourceConfig:
    return SourceConfig(
        name="rh20t",
        root=str(ROOT),
        license_allowed=["CC-BY-SA-4.0", "CC-BY-NC-4.0"],
        max_windows=300,
        max_episodes=300,
        eval_only=False,
        expected_label_rate=0.3,
        seed=0,
    )


def test_discover_finds_fixture() -> None:
    assert Rh20tAdapter(_cfg()).discover(ROOT) == [EID]


def test_load_on_episode_clock() -> None:
    a = Rh20tAdapter(_cfg())
    a.discover(ROOT)
    ep = a.load(EID)
    assert ep.session == "scene_0006" and ep.license == "CC-BY-NC-4.0"
    assert ep.meta["rating"] == 7 and ep.meta["scene"] == 6 and ep.meta["license"] == "CC-BY-NC-4.0"
    assert ep.force.channel == "wrist" and set(ep.force.arms) == {"R"} and ep.force.rate_hz == 10
    assert ep.force.t[0] == 0.0 and ep.meta["clock_offset_s"] == 0.0
    arm = ep.force.arms["R"]
    assert arm.kind == "wrist" and max(arm.fz) == pytest.approx(10.0)
    assert ep.gripper is not None
    assert max(ep.gripper.arms["R"]) == pytest.approx(100 / 110)
    assert set(ep.cameras) == {"scene", "extra_f0002"}
    assert ep.cameras["scene"].fps == 10 and ep.cameras["scene"].n_frames == 10


def test_outcome_words_stripped_and_labels() -> None:
    a = Rh20tAdapter(_cfg())
    a.discover(ROOT)
    assert a.load(EID).task == "put the cup on the plate, then if"
    assert a.native_labels(EID) == []
    s = a.splits()
    assert sum(len(v) for v in s.values()) == 1
