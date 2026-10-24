"""Tests for the baxter_rollouts adapter against a generated LeRobot v3 layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.schema import PadArmForce
from tactile_flora.sources.base import get_source
from tactile_flora.sources.baxter_rollouts import BaxterRolloutsAdapter
from tactile_flora.video.readers import PyAVReader
from tests.fixtures._video import write_mp4

FPS = 15
LENGTHS = (6, 4)  # two episodes concatenated in one mp4


def _write_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": FPS,
        "features": {
            "observation.state": {"dtype": "float32", "names": ["left_s0", "left_gripper"]},
            "observation.tactile": {"dtype": "float32", "names": ["L_a", "L_b", "R_a", "R_b"]},
            "observation.images.head": {"dtype": "video"},
            "observation.images.wrist": {"dtype": "video"},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    rows, ep_rows, start = [], [], 0
    for e, n in enumerate(LENGTHS):
        for i in range(n):
            closed = i >= n // 2
            rows.append(
                {
                    "timestamp": np.float32(i / FPS),
                    "frame_index": i,
                    "episode_index": e,
                    "observation.state": [0.0, 0.37 if closed else 0.96],
                    "observation.tactile": [400.0 if closed else 92.0, 92.0, 0.0, 0.0],
                }
            )
        ep_rows.append(
            {
                "episode_index": e,
                "length": n,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "videos/observation.images.head/chunk_index": 0,
                "videos/observation.images.head/file_index": 0,
                "videos/observation.images.head/from_timestamp": start / FPS,
                "videos/observation.images.wrist/chunk_index": 0,
                "videos/observation.images.wrist/file_index": 0,
                "videos/observation.images.wrist/from_timestamp": start / FPS,
            }
        )
        start += n
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet")
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(ep_rows).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    for key in ("head", "wrist"):
        write_mp4(
            root / f"videos/observation.images.{key}/chunk-000/file-000.mp4",
            n=sum(LENGTHS),
            fps=FPS,
        )
    index = {}
    for e, n in enumerate(LENGTHS):
        d = root / "raw" / f"ep{e:04d}"
        d.mkdir(parents=True)
        t = np.arange(0, n / FPS, 1 / 200.0)
        la = np.where(t >= (n // 2) / FPS, 400.0, 92.0)
        pd.DataFrame(
            {
                "t": t,
                "teensy_micros": 0,
                "counts_L_a": la,
                "counts_L_b": 92.0,
                "counts_R_a": 0.0,
                "counts_R_b": 0.0,
            }
        ).to_csv(d / "tactile_200hz.csv", index=False)
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "task": "pick up the red cube",
                    "label": "success" if e == 0 else "fail",
                    "label_source": "operator",
                    "stop_reason": "picked",
                    "note": "test",
                    "started": "2026-10-24T03:04:09",
                }
            )
        )
        index[str(e)] = f"rollouts/20261024/ep{e + 1:04d}"
    (root / "raw/index.json").write_text(json.dumps(index))


def _cfg(root: Path, **extra: object) -> SourceConfig:
    return SourceConfig(
        name="baxter_rollouts",
        root=str(root),
        license_allowed=["apache-2.0"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=1.0,
        seed=0,
        **extra,  # type: ignore[arg-type]
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write_dataset(tmp_path)
    return tmp_path


def test_registered_and_discover(root: Path) -> None:
    assert get_source("baxter_rollouts") is BaxterRolloutsAdapter
    assert BaxterRolloutsAdapter(_cfg(root)).discover(root) == ["ep0000", "ep0001"]


def test_load_second_episode_on_its_own_clock(root: Path) -> None:
    a = BaxterRolloutsAdapter(_cfg(root, baxter_rollouts_arm="R"))
    a.discover(root)
    ep = a.load("ep0001")
    assert ep.session == "20261024" and ep.license == "apache-2.0"
    assert ep.task == "pick up the red cube" and ep.meta["object"] == "red cube"
    assert set(ep.force.arms) == {"R"} and ep.force.channel == "pad"
    assert ep.force.t[0] == pytest.approx(0.0) and ep.force.rate_hz == pytest.approx(200.0)
    arm = ep.force.arms["R"]
    assert isinstance(arm, PadArmForce) and arm.f_a[-1] == pytest.approx(400.0)
    assert ep.gripper is not None and set(ep.gripper.arms) == {"R"}
    g = ep.gripper.arms["R"]
    assert g[0] == pytest.approx(0.0) and g[-1] == pytest.approx((0.96 - 0.37) / 0.92)
    assert set(ep.cameras) == {"scene"}  # wrist not requested
    assert ep.cameras["scene"].n_frames == 4 and ep.cameras["scene"].fps == FPS
    assert ep.meta["video_from_s"]["scene"] == pytest.approx(6 / FPS)
    assert ep.duration_s == pytest.approx(4 / FPS)
    # The reader honours the in-file offset: the first frame of ep0001 is file frame 6.
    frames = PyAVReader(root).read_many(ep, "scene", [0, 3])
    assert frames[0].shape[-1] == 3 and int(frames[0].mean()) > int(frames[1].mean()) - 200


def test_wrist_camera_opt_in(root: Path) -> None:
    a = BaxterRolloutsAdapter(_cfg(root, baxter_rollouts_cameras=["head", "wrist"]))
    a.discover(root)
    assert set(a.load("ep0000").cameras) == {"scene", "wrist_left"}


def test_labels_recorder_then_override(root: Path) -> None:
    a = BaxterRolloutsAdapter(_cfg(root))
    a.discover(root)
    assert [lab.outcome for lab in a.native_labels("ep0000")] == [5]
    assert [lab.outcome for lab in a.native_labels("ep0001")] == [1]
    assert a.native_labels("ep0001")[0].origin == "source"
    (root / "labels.json").write_text(json.dumps({"ep0001": {"outcome": 5, "note": "operator"}}))
    lab = a.native_labels("ep0001")[0]
    assert lab.outcome == 5 and lab.origin == "human"
    assert a.splits() == {"benchmark": ["ep0000", "ep0001"]}
