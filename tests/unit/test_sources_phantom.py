"""Tests for the phantom (HapticWAM) adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.phantom import PhantomAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "phantom"
EID = "ep_egg_fail_20260801_000"


def _cfg() -> SourceConfig:
    return SourceConfig(
        name="phantom",
        root=str(ROOT),
        license_allowed=["CC-BY-4.0"],
        max_windows=300,
        max_episodes=100,
        eval_only=False,
        expected_label_rate=0.3,
        seed=0,
    )


def test_discover_finds_fixture() -> None:
    assert PhantomAdapter(_cfg()).discover(ROOT) == [EID]


def test_load_on_episode_clock() -> None:
    a = PhantomAdapter(_cfg())
    a.discover(ROOT)
    ep = a.load(EID)
    assert ep.source == "phantom" and ep.session == "20260801" and ep.license == "CC-BY-4.0"
    assert ep.meta["clock_offset_s"] == pytest.approx(1000.0)
    assert ep.duration_s == pytest.approx(9 / 15)
    assert ep.force.channel == "pad" and set(ep.force.arms) == {"R"}
    assert ep.force.rate_hz == 10.0 and ep.force.t[0] == 0.0
    arm = ep.force.arms["R"]
    assert arm.kind == "pad"
    assert max(arm.f_a) == pytest.approx(10.0) and max(arm.f_b) == pytest.approx(8.0)
    assert ep.gripper is not None and set(ep.gripper.arms) == {"R"}
    assert ep.gripper.t[0] == pytest.approx(-0.02) and max(ep.gripper.arms["R"]) == 1.0
    cam = ep.cameras["scene"]
    assert cam.fps == 15 and cam.n_frames == 10 and cam.path.endswith("camera_scene_color")
    for key in (
        "success",
        "failure_demo",
        "task_folder",
        "tactile_contact",
        "object",
        "peak_force_N",
    ):
        assert key in ep.meta
    assert ep.meta["failure_demo"] is True and ep.meta["task_folder"] == "egg_fail"


def test_outcome_words_stripped() -> None:
    a = PhantomAdapter(_cfg())
    a.discover(ROOT)
    task = a.load(EID).task
    assert "failure" not in task.lower() and task.startswith("Pick up the egg")


def test_splits_force_failure_demos_to_train() -> None:
    a = PhantomAdapter(_cfg())
    a.discover(ROOT)
    # manifest says val, but this is a failure demo
    assert a.splits() == {"train": [EID], "val": [], "test": []}


def test_native_labels_shape() -> None:
    a = PhantomAdapter(_cfg())
    a.discover(ROOT)
    assert all(lb.origin == "source" for lb in a.native_labels(EID))
