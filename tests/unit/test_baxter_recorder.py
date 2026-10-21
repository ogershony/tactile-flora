"""Recorder gate and episode writing, then the to_lerobot round trip through the adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tactile_flora.baxter.ros.recorder import Recorder, run_episode
from tactile_flora.baxter.ros.transport import FakeTransport
from tactile_flora.baxter.to_lerobot import convert
from tactile_flora.config import (
    BaxterConfig,
    ForceConfig,
    SourceConfig,
    load_config,
)
from tactile_flora.schema import VerifyContext
from tactile_flora.sources.base import get_source
from tactile_flora.sources.baxter import BaxterAdapter
from tactile_flora.verify.checks import check_sync

CONFIGS = Path(__file__).parents[2] / "configs"
CFG = load_config(CONFIGS / "baxter.yaml", BaxterConfig)
FCFG = load_config(CONFIGS / "force.yaml", ForceConfig)
T0 = 50.0
HEALTHY = {"variance": [4.0] * 4, "saturation": [0.0] * 4, "baseline_drift": [0.0] * 4}


def _setup(
    tmp_path: Path, health: dict[str, list[float]] | None = None
) -> tuple[FakeTransport, Recorder]:
    tr = FakeTransport(t0=T0)
    tr.advertise(CFG.topics.zero, lambda _r: {"baseline": [500.0, 490.0, 505.0, 515.0]})
    rec = Recorder(tr, CFG, tmp_path / "raw", "cal-2026-11-01", today=lambda: "2026-11-02")
    tr.publish(CFG.topics.health, health or HEALTHY, T0)
    for arm in ("L", "R"):
        tr.publish(CFG.topics.gripper_state[arm], {"position": 0.0, "command": 0.0}, T0)
    return tr, rec


def _drive(tr: FakeTransport, seconds: float = 2.0, t_start: float = 100.0) -> None:
    """Publish synthetic cameras (30 fps), gripper (30 Hz) and force (200 Hz) for ``seconds``."""
    n = int(seconds * 30)
    for k in range(n):
        t = t_start + k / 30
        img = np.full((32, 32, 3), 10 * (k % 20), dtype=np.uint8)
        for topic in CFG.topics.cameras.values():
            tr.publish(topic, {"image": img}, t)
        closure = float(np.clip((k / 30 - 0.5) / 0.5, 0, 1))
        tr.publish(CFG.topics.gripper_state["L"], {"position": closure, "command": closure}, t)
        tr.publish(CFG.topics.gripper_state["R"], {"position": 0.0, "command": 0.0}, t)
    for j in range(int(seconds * 200)):
        t = t_start + j / 200
        ramp = float(np.clip((j / 200 - 0.5) / 0.5, 0, 1))
        tr.publish(CFG.topics.force, {"data": [3 * ramp, 2 * ramp, 0.01, 0.02]}, t)


def _record(tmp_path: Path) -> Path:
    tr, rec = _setup(tmp_path)
    tr.publish(
        CFG.topics.policy_status,
        {
            "object": "paper cup",
            "firmness": "compliant",
            "grasp_height": 0.12,
            "lateral_offset": 0.01,
            "gripper_force": 25.0,
            "occlusion": True,
        },
        T0,
    )
    rec.start_episode("synthetic")
    _drive(tr)
    tr.publish(CFG.topics.policy_status, {"state": "done"}, T0)
    assert rec.policy_done
    return rec.stop_episode()


def test_run_episode_times_out_and_stops(tmp_path: Path) -> None:
    _tr, rec = _setup(tmp_path)
    d = run_episode(rec, "n", timeout_s=0.5)
    assert (d / "meta.json").is_file() and not rec.recording


@pytest.mark.parametrize(
    "health,msg",
    [
        ({**HEALTHY, "variance": [4.0, 0.0, 4.0, 4.0]}, "flat"),
        ({**HEALTHY, "saturation": [0.0, 0.0, 0.5, 0.0]}, "saturated"),
    ],
)
def test_refuses_flat_or_saturated(
    tmp_path: Path, health: dict[str, list[float]], msg: str
) -> None:
    _tr, rec = _setup(tmp_path, health)
    with pytest.raises(RuntimeError, match=msg):
        rec.start_episode()
    assert not (tmp_path / "raw").exists()


def test_refuses_closed_gripper_stale_health_and_no_health(tmp_path: Path) -> None:
    tr, rec = _setup(tmp_path)
    tr.publish(CFG.topics.gripper_state["R"], {"position": 0.8, "command": 1.0}, T0)
    with pytest.raises(RuntimeError, match="not open"):
        rec.start_episode()
    tr.publish(CFG.topics.gripper_state["R"], {"position": 0.0, "command": 0.0}, T0)
    tr.sleep(60)
    with pytest.raises(RuntimeError, match="stale"):
        rec.start_episode()
    tr2 = FakeTransport()
    with pytest.raises(RuntimeError, match="health"):
        Recorder(tr2, CFG, tmp_path / "x").start_episode()


def test_episode_directory(tmp_path: Path) -> None:
    d = _record(tmp_path)
    assert d.name == "episode_000000"
    for role in ("scene", "wrist_left", "wrist_right"):
        assert (d / f"{role}.mp4").stat().st_size > 0
    meta = json.loads((d / "meta.json").read_text())
    assert meta["object"] == "paper cup" and meta["occlusion"] is True
    assert meta["policy"] == {"grasp_height": 0.12, "lateral_offset": 0.01, "gripper_force": 25.0}
    assert meta["recording_day"] == "2026-11-02" and meta["calibration_version"] == "cal-2026-11-01"
    assert meta["video_t0_s"] == pytest.approx(100.0) and meta["n_frames"]["scene"] == 60
    assert meta["baseline"] == [500.0, 490.0, 505.0, 515.0]
    force = pd.read_parquet(d / "force.parquet")
    assert list(force.columns) == ["t", "L_a", "L_b", "R_a", "R_b"] and len(force) == 400
    assert force["t"].iloc[0] == pytest.approx(100.0)


def test_round_trip_through_adapter(tmp_path: Path) -> None:
    _record(tmp_path)
    out = convert(tmp_path / "raw", tmp_path / "lerobot", FCFG.resample)
    cfg = SourceConfig(
        name="baxter",
        root=str(out),
        license_allowed=["internal"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=0.35,
        seed=0,
    )
    assert get_source("baxter") is BaxterAdapter
    adapter = BaxterAdapter(cfg)
    assert adapter.discover(out) == ["episode_000000"]
    ep = adapter.load("episode_000000")
    assert ep.session == "2026-11-02" and set(ep.cameras) == {"scene", "wrist_left", "wrist_right"}
    assert ep.meta["object"] == "paper cup" and ep.meta["clock_offset_s"] == pytest.approx(100.0)
    assert ep.cameras["scene"].n_frames == 60 and ep.duration_s == pytest.approx(2.0)
    assert len(ep.force.t) == 400 and ep.force.t[0] == pytest.approx(0.0)
    assert ep.force.arms["L"].n_samples() == 400
    frames = pd.read_parquet(out / "data" / "chunk-000" / "episode_000000.parquet")
    obs = np.stack(frames["observation.force"].to_numpy())
    assert obs.shape == (60, 4) and obs[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert obs[-1, 0] == pytest.approx(3.0, abs=1e-6) and obs[-1, 1] == pytest.approx(2.0, abs=1e-6)
    assert frames["observation.gripper_left"].iloc[-1] == pytest.approx(1.0)
    assert not (out / "labels.json").exists()
    sync = check_sync(ep, VerifyContext(source="baxter", resample_cfg=FCFG.resample))
    assert sync.passed and abs(sync.value or 0.0) < 0.1
