"""Tests for the hrdexdb adapter against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.sources.base import get_source
from tactile_flora.sources.hrdexdb import HrdexdbAdapter, _signed_force, video_clock

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


def test_unsigned_force_bit_patterns_are_decoded_before_absolute_value() -> None:
    encoded = np.array([0, 19, 32767, 32768, 65517, 65535], dtype=np.int64)
    np.testing.assert_array_equal(_signed_force(encoded), [0, 19, 32767, -32768, -19, -1])
    # Arrays already represented as signed int16 remain signed.
    signed = np.array([-19, 0, 19], dtype=np.int16)
    np.testing.assert_array_equal(_signed_force(signed), [-19, 0, 19])


def test_recorded_clock_corrects_offset_drift_and_missing_timestamp(tmp_path: Path) -> None:
    directory = tmp_path / "raw/timestamps"
    directory.mkdir(parents=True)
    # Capture is 2x faster than the encoded video, starts two seconds after the sensor,
    # and the middle frame's timestamp is missing. Frame IDs are one-based upstream.
    np.save(directory / "timestamp.npy", [102.0, 102.5, 103.5, 104.0])
    np.save(directory / "frame_id.npy", [1, 2, 4, 5])
    sensor = np.array([100, 102, 102.25, 103, 104, 105])
    mask, mapped, info = video_clock(tmp_path, sensor, fps=1, n_frames=5)
    np.testing.assert_array_equal(mask, [False, True, True, True, True, False])
    np.testing.assert_allclose(mapped, [0, 0.5, 2, 4])
    assert info["video_start_minus_sensor_start_s"] == 2
    assert info["missing_frame_timestamps"] == 1


def test_adapter_aligns_force_and_gripper_on_the_same_video_clock(tmp_path: Path) -> None:
    root = tmp_path / "data"
    copytree(ROOT, root)
    directory = root / EID / "raw/timestamps"
    directory.mkdir()
    capture = np.linspace(100.05, 100.275, 10)
    np.save(directory / "timestamp.npy", capture)
    np.save(directory / "frame_id.npy", np.arange(1, 11))
    legacy = HrdexdbAdapter(_cfg()).load(EID)
    cfg = _cfg().model_copy(update={"root": str(root), "hrdexdb_use_video_timestamps": True})
    aligned = HrdexdbAdapter(cfg).load(EID)
    sensor_time = np.load(root / EID / "raw/hand/time.npy")
    mask = (sensor_time >= capture[0]) & (sensor_time <= capture[-1])
    np.testing.assert_allclose(
        aligned.force.t, np.interp(sensor_time[mask], capture, np.arange(10) / 30)
    )
    aligned_arm = aligned.force.arms["R"]
    legacy_arm = legacy.force.arms["R"]
    assert aligned_arm.kind == "pad"
    assert legacy_arm.kind == "pad"
    np.testing.assert_array_equal(aligned_arm.f_a, np.asarray(legacy_arm.f_a)[mask])
    assert aligned.gripper is not None and legacy.gripper is not None
    np.testing.assert_array_equal(aligned.gripper.t, aligned.force.t)
    np.testing.assert_array_equal(
        aligned.gripper.arms["R"], np.asarray(legacy.gripper.arms["R"])[mask]
    )
    assert aligned.meta["clock_offset_s"] == pytest.approx(capture[0])


@pytest.mark.parametrize(
    "timestamps,frames",
    [
        ([102, 101], [1, 5]),
        ([102, 103], [0, 4]),
        ([102, 103], [1, 4]),
        ([102, float("nan")], [1, 5]),
        ([102, 103], [1, 1]),
        ([102], [1, 5]),
    ],
)
def test_recorded_clock_refuses_invalid_timelines(tmp_path, timestamps, frames) -> None:
    directory = tmp_path / "raw/timestamps"
    directory.mkdir(parents=True)
    np.save(directory / "timestamp.npy", timestamps)
    np.save(directory / "frame_id.npy", frames)
    with pytest.raises(ValueError, match="invalid or incomplete"):
        video_clock(tmp_path, np.array([102, 103]), fps=1, n_frames=5)


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
