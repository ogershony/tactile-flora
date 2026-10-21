"""Tests for the baxter adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.base import get_source
from tactile_flora.sources.baxter import BaxterAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "baxter" / "data"
EID = "episode_000000"


def _cfg() -> SourceConfig:
    return SourceConfig(
        name="baxter",
        root=str(ROOT),
        license_allowed=["internal"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=0.35,
        seed=0,
    )


def test_registered_and_discover() -> None:
    assert get_source("baxter") is BaxterAdapter
    assert BaxterAdapter(_cfg()).discover(ROOT) == [EID]


def test_load_on_episode_clock() -> None:
    ep = BaxterAdapter(_cfg()).load(EID)
    assert ep.session == "2026-11-02" and ep.license == "internal"
    assert ep.force.channel == "pad" and set(ep.force.arms) == {"L", "R"}
    assert ep.meta["clock_offset_s"] == pytest.approx(50.1)
    assert ep.force.t[0] == pytest.approx(-0.1)  # force began 0.1 s before the first frame
    assert ep.force.rate_hz == pytest.approx(200.0)
    assert len(ep.force.t) == 60
    assert ep.force.arms["L"].kind == "pad"
    assert ep.force.arms["L"].f_b[-1] == pytest.approx(3.0 * 1.0)
    assert ep.gripper is not None and set(ep.gripper.arms) == {"L", "R"}
    assert ep.gripper.t[0] == 0.0 and ep.gripper.rate_hz == pytest.approx(30.0)
    assert set(ep.cameras) == {"scene", "wrist_left", "wrist_right"}
    assert ep.cameras["scene"].n_frames == 10
    assert ep.meta["policy"]["gripper_force"] == 30.0


def test_outcome_words_stripped() -> None:
    assert BaxterAdapter(_cfg()).load(EID).task == "pick up the soda can"


def test_splits_and_labels() -> None:
    a = BaxterAdapter(_cfg())
    assert a.splits() == {"benchmark": [EID]}
    labels = a.native_labels(EID)
    assert [lb.kind for lb in labels] == ["outcome", "dense"]
    assert all(lb.origin == "source" for lb in labels)
    assert labels[0].outcome == 2
    spans = [(s.t0, s.t1, s.quality, s.rationale) for s in labels[1].spans]
    assert spans == [(2.0, 4.0, 2, ""), (4.0, 5.0, 1, "")]
