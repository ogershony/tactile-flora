"""Tests for the airoa adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.airoa import AiroaAdapter, wrench_key

ROOT = Path(__file__).parents[1] / "fixtures" / "airoa"


def _cfg() -> SourceConfig:
    return SourceConfig(
        name="airoa",
        root=str(ROOT),
        license_allowed=["CC-BY-NC-SA-4.0"],
        max_windows=100,
        max_episodes=200,
        eval_only=False,
        expected_label_rate=0.3,
        seed=0,
    )


def test_discover_finds_fixture() -> None:
    assert AiroaAdapter(_cfg()).discover(ROOT) == ["0"]


def test_load_on_episode_clock() -> None:
    a = AiroaAdapter(_cfg())
    a.discover(ROOT)
    ep = a.load("0")
    assert ep.session == "hsr12_2026-03-14" and ep.license == "CC-BY-NC-SA-4.0"
    assert ep.force.channel == "wrist" and set(ep.force.arms) == {"R"} and ep.force.rate_hz == 10
    assert ep.force.t[0] == 0.0 and ep.meta["clock_offset_s"] == 0.0
    arm = ep.force.arms["R"]
    assert arm.kind == "wrist" and max(arm.fz) == pytest.approx(10.0)
    assert ep.gripper is not None and max(ep.gripper.arms["R"]) == pytest.approx(1.0)
    assert set(ep.cameras) == {"scene", "wrist_right"}
    assert ep.cameras["scene"].fps == 10 and ep.cameras["scene"].n_frames == 10
    for k, v in (
        ("task_success", True),
        ("primitive_action", "pick"),
        ("task_type", "short_horizon"),
    ):
        assert ep.meta[k] == v
    assert ep.meta["label"] == "pick cup"


def test_outcome_words_stripped_and_splits() -> None:
    a = AiroaAdapter(_cfg())
    a.discover(ROOT)
    assert a.load("0").task == "pick up the cup and do not"
    assert sum(len(v) for v in a.splits().values()) == 1
    assert a.native_labels("0") == []


def test_wrench_key_fallbacks() -> None:
    assert wrench_key(["a", "observation.wrist.wrench"]) == "observation.wrist.wrench"
    assert wrench_key(["observation.state", "observation.wrist_ft"]) == "observation.wrist_ft"
    assert wrench_key(["observation.state", "observation.lift"]) is None
