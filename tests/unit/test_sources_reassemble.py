"""Tests for the reassemble adapter against the synthetic fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.reassemble import ReassembleAdapter

ROOT = Path(__file__).parents[1] / "fixtures" / "reassemble"
EID = "2025-01-11-14-45-48"


def _cfg(root: Path = ROOT) -> SourceConfig:
    return SourceConfig(
        name="reassemble",
        root=str(root),
        license_allowed=["CC-BY-4.0"],
        max_windows=150,
        max_episodes=300,
        eval_only=False,
        expected_label_rate=0.3,
        seed=0,
    )


def test_discover_finds_fixture() -> None:
    assert ReassembleAdapter(_cfg()).discover(ROOT) == [EID]


def test_load_on_episode_clock() -> None:
    a = ReassembleAdapter(_cfg())
    a.discover(ROOT)
    ep = a.load(EID)
    assert ep.session == "2025-01-11" and ep.license == "CC-BY-4.0"
    assert ep.meta["clock_offset_s"] == pytest.approx(500.0)
    assert ep.force.channel == "wrist" and set(ep.force.arms) == {"R"}
    assert ep.force.t[0] == pytest.approx(-0.05) and ep.force.rate_hz == pytest.approx(100.0)
    assert ep.force.arms["R"].kind == "wrist" and max(ep.force.arms["R"].fz) == pytest.approx(10.0)
    assert ep.gripper is not None and max(ep.gripper.arms["R"]) == pytest.approx(1.0)
    assert set(ep.cameras) == {"scene", "scene_2", "wrist_right"}
    assert ep.cameras["scene"].n_frames == 10 and ep.cameras["scene"].fps == pytest.approx(10)
    assert ep.duration_s == pytest.approx(0.9)
    assert ep.meta["known_issues"] == ["ft_invalid_last_action"]
    assert ep.meta["action_success"] == [True, False]
    assert ep.meta["segments"][1] == {
        "t0": pytest.approx(0.4),
        "t1": pytest.approx(0.9),
        "success": False,
        "text": "Insert square peg 3.",
    }
    assert ep.meta["low_level_segments"][0]["parent"] == 0
    assert ep.task == "Pick square peg 3."


def test_labels_and_splits() -> None:
    a = ReassembleAdapter(_cfg())
    a.discover(ROOT)
    (label,) = a.native_labels(EID)
    assert label.kind == "event" and label.origin == "source"
    assert [s.quality for s in label.spans] == [3, 1]
    assert sum(len(v) for v in a.splits().values()) == 1


def test_missing_extracted_video_raises_with_command(tmp_path: Path) -> None:
    shutil.copy(ROOT / f"{EID}.h5", tmp_path / f"{EID}.h5")
    a = ReassembleAdapter(_cfg(tmp_path))
    a.discover(tmp_path)
    with pytest.raises(FileNotFoundError, match="h5py"):
        a.load(EID)
