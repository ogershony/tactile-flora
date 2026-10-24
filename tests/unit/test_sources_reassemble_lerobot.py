"""Tests for the reassemble_lerobot adapter against the synthetic port fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.sources.base import get_source, make_adapter
from tactile_flora.sources.reassemble_lerobot import (
    ReassembleLerobotAdapter,
    frame_segments,
    holdout_split,
    is_idle,
    port_split,
    segment_labels,
)
from tactile_flora.verify import manifest, run_manifest, run_verify
from tactile_flora.verify.windowing import episode_windows
from tactile_flora.video.readers import AutoReader, PyAVReader
from tests.fixtures.reassemble_lerobot import make_fixture as fx

ROOT = Path(__file__).parents[1] / "fixtures" / "reassemble_lerobot"
FORCE_CFG = load_config("configs/force.yaml", ForceConfig)


def _cfg(root: Path = ROOT, **rules: float) -> SourceConfig:
    return SourceConfig(
        name="reassemble_lerobot",
        root=str(root),
        license_allowed=["CC-BY-4.0"],
        max_windows=0,
        max_episodes=0,
        eval_only=False,
        expected_label_rate=0.88,
        split_rules=dict(rules),
        seed=0,
    )


def _adapter(root: Path = ROOT) -> ReassembleLerobotAdapter:
    a = ReassembleLerobotAdapter(_cfg(root))
    a.discover(root)
    return a


def test_registered_and_config_loads() -> None:
    assert get_source("reassemble_lerobot") is ReassembleLerobotAdapter
    a = make_adapter("reassemble_lerobot")
    assert a.name == "reassemble_lerobot"
    cfg = load_config("configs/sources/reassemble_lerobot.yaml", SourceConfig)
    assert cfg.license_allowed == ["CC-BY-4.0"] and cfg.root == "data/raw/reassemble_lerobot"


def test_discover_lists_episodes_present_on_disk() -> None:
    # episode 2 is in the episodes table and splits.json but its data parquet is missing
    assert ReassembleLerobotAdapter(_cfg()).discover(ROOT) == ["0", "1"]


def test_discover_without_dataset_is_empty(tmp_path: Path) -> None:
    assert ReassembleLerobotAdapter(_cfg(tmp_path)).discover(tmp_path) == []


def test_load_on_episode_clock() -> None:
    ep = _adapter().load("0")
    assert ep.source == "reassemble_lerobot" and ep.episode_id == "0"
    assert ep.session == "2025-01-11" and ep.license == "CC-BY-4.0"
    assert ep.meta["recording"] == "2025-01-11-14-45-48" and ep.meta["port_split"] == "train"
    assert ep.meta["known_issues"] == ["ft_invalid_last_action"]
    assert ep.meta["clock_offset_s"] == 0.0 and ep.duration_s == pytest.approx(29 / 30)
    assert ep.force.channel == "wrist" and set(ep.force.arms) == {"R"} and ep.force.rate_hz == 30
    assert ep.force.t == pytest.approx([k / 30 for k in range(30)])
    arm = ep.force.arms["R"]
    # force comes from the gravity-compensated column (the raw one carries RAW_BIAS)
    assert ep.meta["force_keys"] == ["observation.force.base", "observation.torque"]
    assert arm.kind == "wrist" and max(arm.fz) == pytest.approx(10.0)
    assert max(arm.fx) == pytest.approx(2.0) and min(arm.fz) == pytest.approx(0.0)
    assert max(arm.ty) == 0.0 and max(arm.tz) == 0.0  # sensor torque, not torque.base
    assert ep.gripper is not None and ep.gripper.t == ep.force.t
    g = ep.gripper.arms["R"]
    # width is float32 in the port, so fully open comes out at 1 - 0.04f / 0.04 ~ 2e-8
    assert min(g) == pytest.approx(0.0, abs=1e-6) and max(g) == pytest.approx(1.0)
    assert set(ep.cameras) == {"scene", "scene_2", "wrist_right"}
    for ref in ep.cameras.values():
        assert ref.fps == 30 and ref.n_frames == 30 and Path(ref.path).exists()
    assert ep.cameras["scene"].path.endswith("observation.images.hama1/chunk-000/file-000.mp4")
    assert ep.meta["video_from_s"] == {"scene": 0.0, "scene_2": 0.0, "wrist_right": 0.0}
    assert ep.task == "Pick square peg 3."


def test_segments_tile_the_episode_and_keep_idle_time() -> None:
    ep = _adapter().load("0")
    segs = ep.meta["segments"]
    assert [s["text"] for s in segs] == [
        "No action.",
        "Pick square peg 3.",
        "Insert square peg 3.",
        "No action.",
    ]
    assert [s["idle"] for s in segs] == [True, False, False, True]
    assert [s["success"] for s in segs] == [True, True, False, True]
    assert [(s["t0"], s["t1"]) for s in segs] == pytest.approx(
        [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    )
    assert [(s["frame0"], s["frame1"]) for s in segs] == [(0, 6), (6, 15), (15, 24), (24, 30)]
    assert ep.meta["segment_boundaries_s"] == pytest.approx([0.0, 0.2, 0.5, 0.8, 1.0])
    assert ep.meta["action_success"] == [True, False]


def test_failure_labels_mirror_the_raw_mapping() -> None:
    a = _adapter()
    ep = a.load("0")
    assert [lab.kind for lab in ep.labels] == ["outcome", "dense", "event"]
    assert all(lab.origin == "source" for lab in ep.labels)
    outcome, dense, event = ep.labels
    assert outcome.outcome == 1 and outcome.spans == []
    (span,) = dense.spans
    assert (span.t0, span.t1, span.quality, span.rationale) == (
        pytest.approx(0.5),
        pytest.approx(0.8),
        1,
        "Insert square peg 3.",
    )
    assert [(s.quality, s.rationale) for s in event.spans] == [
        (3, "Pick square peg 3."),
        (1, "Insert square peg 3."),
    ]
    assert a.native_labels("0") == ep.labels


def test_idle_segments_are_excluded_from_labels() -> None:
    ep = _adapter().load("1")
    # the idle segment is flagged failed in the data; it must not fail the recording
    assert ep.meta["segments"][0]["idle"] and ep.meta["segments"][0]["success"] is False
    assert ep.meta["action_success"] == [True]
    assert [lab.kind for lab in ep.labels] == ["outcome", "event"]
    assert ep.labels[0].outcome == 5
    assert [s.quality for s in ep.labels[1].spans] == [3]
    assert ep.task == "Pick square peg 3."


def test_second_episode_in_concatenated_video_is_on_its_own_clock() -> None:
    ep = _adapter().load("1")
    assert ep.meta["video_from_s"] == {"scene": 1.0, "scene_2": 1.0, "wrist_right": 1.0}
    assert (
        ep.force.t[0] == 0.0 and len(ep.force.t) == 12 and ep.duration_s == pytest.approx(11 / 30)
    )
    assert ep.force.t == pytest.approx([k / 30 for k in range(12)])
    assert ep.gripper is not None and ep.gripper.t == ep.force.t
    assert ep.meta["segment_boundaries_s"] == pytest.approx([0.0, 0.1, 0.4])
    assert ep.cameras["scene"].n_frames == 12
    assert PyAVReader().frame_count(ep, "scene") == 12
    assert ep.session == "2025-01-13" and ep.meta["port_split"] == "test"


def test_splits_follow_the_port_and_hold_out_test() -> None:
    s = _adapter().splits()
    assert s["train"] == ["0"]
    assert sorted(s["val"] + s["test"]) == ["1"]
    cfg = _cfg()
    assert port_split("train", "2025-01-11", cfg) == "train"
    assert port_split("unassigned", "2025-01-10", cfg) == "train"
    assert port_split(None, "2025-01-10", cfg) == "train"
    held = port_split("test", "2025-01-13", cfg)
    assert held in {"val", "test"} and held == port_split("test", "2025-01-13", cfg)
    assert holdout_split("2025-01-13", _cfg(val_fraction=1.0, test_fraction=0.0)) == "val"
    assert holdout_split("2025-01-13", _cfg(val_fraction=0.0, test_fraction=1.0)) == "test"
    assert holdout_split("2025-01-13", _cfg(val_fraction=0.0, test_fraction=0.0)) == "test"
    dates = [f"2025-01-{d:02d}" for d in range(1, 31)]
    picks = {holdout_split(d, cfg) for d in dates}
    assert picks == {"val", "test"}


def test_frame_segments_are_runs_in_frame_order() -> None:
    t = np.arange(5) / 10.0
    index = np.array([0, 0, 1, 1, 0])
    ok = np.array([True, True, True, False, True])
    task = np.array([7, 7, 8, 8, 7])
    segs = frame_segments(index, ok, task, t, 10.0)
    assert [(s["index"], s["task_index"], s["success"]) for s in segs] == [
        (0, 7, True),
        (1, 8, False),
        (0, 7, True),
    ]
    assert [(s["t0"], s["t1"]) for s in segs] == pytest.approx([(0.0, 0.2), (0.2, 0.4), (0.4, 0.5)])
    assert frame_segments(np.array([]), np.array([]), np.array([]), np.array([]), 10.0) == []


def test_idle_text_and_all_idle_recording() -> None:
    assert is_idle("No action.") and is_idle("no action") and is_idle(" NO ACTION! ")
    assert not is_idle("Pick USB.") and not is_idle("")
    idle_only = [{"t0": 0.0, "t1": 1.0, "success": False, "text": "No action.", "idle": True}]
    assert segment_labels(idle_only) == []


def test_windows_from_segment_labels(tmp_path: Path) -> None:
    fx.write_long(tmp_path)
    ep = _adapter(tmp_path).load("0")
    assert ep.duration_s == pytest.approx(479 / 30)
    wins = episode_windows(ep, FORCE_CFG, None, "train")
    dense = [w for w in wins if w.prompt_kind == "dense"]
    (outcome,) = [w for w in wins if w.prompt_kind == "outcome"]
    assert outcome.target == "Rationale: no rationale\n\nScore: 1"
    assert outcome.label_origin == "source" and len(outcome.frame_times) == 64
    assert [w.anchor_s for w in dense] == pytest.approx([0.0, 479 / 30 - 9.75])
    # the failed segment covers 8..14 s: groups 21-32 of the first window, 9-20 of the second
    assert dense[0].target == (
        "Worst: 1\nQuality:\n" + "3" * 20 + "1" * 12 + "\nRationales:\n21-32 Insert square peg 3."
    )
    assert dense[1].target.split("\n")[2] == "3" * 8 + "1" * 12 + "3" * 12
    assert {w.label_origin for w in dense} == {"source"}


@pytest.mark.parametrize("recording", ["2025-01-13-18-06-08", "2025-01-13-18-16-38"])
def test_measured_ft_invalid_recordings_are_flagged_by_stem(tmp_path: Path, recording: str) -> None:
    from tactile_flora.sources.reassemble import KNOWN_ISSUES

    fx.write_long(tmp_path, recording=recording)
    ep = _adapter(tmp_path).load("0")
    assert ep.meta["recording"] == recording and ep.session == "2025-01-13"
    assert ep.meta["known_issues"] == ["ft_invalid_last_action"] == KNOWN_ISSUES[recording]
    # the flag is metadata only: the episode still loads with its labels
    assert [lab.kind for lab in ep.labels] == ["outcome", "dense", "event"]


def test_force_falls_back_to_the_raw_columns(tmp_path: Path) -> None:
    fx.write_long(tmp_path, compensated=False)
    ep = _adapter(tmp_path).load("0")
    assert ep.meta["force_keys"] == ["observation.force", "observation.torque"]
    arm = ep.force.arms["R"]
    assert arm.kind == "wrist"
    fz = arm.fz
    assert max(fz) == pytest.approx(10.0 + fx.RAW_BIAS[2], abs=0.3)
    assert min(fz) == pytest.approx(fx.RAW_BIAS[2], abs=0.3)


def test_verify_and_manifest_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "reassemble_lerobot"
    fx.write_long(root)
    cfg = _cfg(root)
    data = tmp_path / "data"
    reader = AutoReader(".")
    first = run_verify(cfg, FORCE_CFG, data, reader)
    n = len(list((data / "verified" / cfg.name / "spot").glob("*.png")))
    summary = run_verify(cfg, FORCE_CFG, data, reader, spot_pass=n)
    assert summary.source_ok, (first.episode_failures, summary.source_verdicts)
    rows = manifest.read_manifest(run_manifest(cfg, FORCE_CFG, data, name="t", report_version="rv"))
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r.prompt_kind] = kinds.get(r.prompt_kind, 0) + 1
    assert kinds.get("outcome") == 1 and kinds.get("dense", 0) >= 2
    assert {r.split for r in rows} == {"train"} and {r.source for r in rows} == {cfg.name}
    assert {r.label_origin for r in rows if r.prompt_kind == "outcome"} == {"source"}
