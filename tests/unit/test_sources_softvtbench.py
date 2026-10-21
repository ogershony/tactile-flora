"""Tests for the softvtbench adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.base import get_source
from tactile_flora.sources.softvtbench import SoftvtbenchAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "softvtbench" / "data"
EID = "object-soft/libero_object/LIBERO_Object_pick_egg/demo_0"


def _cfg(**rules: Any) -> SourceConfig:
    return SourceConfig(
        name="softvtbench",
        root=str(ROOT),
        license_allowed=["Apache-2.0"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=0.5,
        split_rules=dict(rules),
        seed=0,
    )


def test_registered_and_discover() -> None:
    assert get_source("softvtbench") is SoftvtbenchAdapter
    ids = SoftvtbenchAdapter(_cfg()).discover(ROOT)
    assert ids == [EID, EID.replace("demo_0", "demo_1")]


def test_load_on_episode_clock() -> None:
    ep = SoftvtbenchAdapter(_cfg()).load(EID)
    assert ep.license == "Apache-2.0"
    assert ep.session == "object-soft/3"
    assert ep.force.channel == "pad" and ep.force.rate_hz == 20.0
    assert ep.force.t[0] == 0.0 and ep.meta["clock_offset_s"] == 0.0
    arm = ep.force.arms["R"]
    assert arm.kind == "pad"
    assert arm.f_a[3] == pytest.approx(10.0) and arm.f_b[3] == pytest.approx(20.0)
    assert ep.gripper is not None and ep.gripper.arms["R"][3] == 1.0
    assert set(ep.cameras) == {"scene", "wrist_right"}
    assert ep.meta["peak_fem_kabsch_max_pct"] == pytest.approx(12.0)
    assert ep.meta["soft_task_kind"] == "pick"
    assert ep.meta["dsr_tolerance_pct"] is None


def test_outcome_words_stripped() -> None:
    assert SoftvtbenchAdapter(_cfg()).load(EID).task == "pick up the egg"


def test_splits_and_labels_without_tolerance() -> None:
    a = SoftvtbenchAdapter(_cfg())
    assert a.splits() == {"benchmark": a.discover(ROOT)}
    labels = a.native_labels(EID)
    assert [(lb.kind, lb.outcome, lb.origin) for lb in labels] == [("outcome", 5, "source")]


def test_overcompression_event_with_tolerance() -> None:
    a = SoftvtbenchAdapter(_cfg(dsr_tolerance_pct=10.0))
    labels = a.native_labels(EID)
    assert [lb.kind for lb in labels] == ["outcome", "event"]
    spans = labels[1].spans
    assert len(spans) == 1
    assert spans[0].t0 == pytest.approx(4 / 20) and spans[0].t1 == pytest.approx(6 / 20)
    assert spans[0].quality == 1
    assert spans[0].rationale == "excessive force on the object"
    assert [
        lb.kind for lb in SoftvtbenchAdapter(_cfg(dsr_tolerance_pct=50.0)).native_labels(EID)
    ] == ["outcome"]
