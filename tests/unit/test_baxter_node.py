"""FsrNode against a recorded-line fake serial and the FakeTransport."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.baxter.calibration import CalibrationFile, fit_pad
from tactile_flora.baxter.ros.fsr_node import FsrNode
from tactile_flora.baxter.ros.transport import FakeTransport
from tactile_flora.config import BaxterConfig, load_config

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT.parent / "configs" / "baxter.yaml", BaxterConfig)
CAL = CalibrationFile(
    date="2026-11-01",
    r_ref_ohm=4700.0,
    sensor_confirmed=False,
    pads={c: fit_pad([0, 4095], [0.0, 40.0]) for c in CFG.channels},
)


class FakeSerial:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)
        self.written: list[bytes] = []
        self.on_write: list[bytes] = []

    def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    def write(self, data: bytes) -> int:
        self.written.append(data)
        if data == b"Z\n":
            self.lines = self.on_write + self.lines
        return len(data)


def _recorded() -> list[bytes]:
    text = (ROOT / "fixtures" / "baxter" / "serial_lines.txt").read_text()
    return [line.encode() + b"\n" for line in text.splitlines()]


def test_node_publishes_stamped_raw_force_and_health() -> None:
    tr = FakeTransport(t0=100.0)
    serial = FakeSerial(_recorded())
    log: list[str] = []
    cfg = CFG.model_copy(update={"health_period_s": 0.1})
    node = FsrNode(tr, serial, cfg, CAL, echo=log.append)
    assert log and "UNCONFIRMED" in log[0]
    # deliver lines as if they arrived in real time (200 Hz) so the clock fit has a time base
    while serial.lines:
        node.step(max_lines=1)
        tr.sleep(0.005)
    raw = tr.messages(CFG.topics.raw)
    force = tr.messages(CFG.topics.force)
    assert len(raw) == len(force) == node.n_samples == 44 and node.n_bad_lines == 3
    assert raw[0][0]["data"][0] == pytest.approx(500, abs=3)
    assert force[0][0]["data"][0] == pytest.approx(raw[0][0]["data"][0] * 40 / 4095, rel=1e-4)
    stamps = [s for _, s in raw]
    assert stamps == sorted(stamps)
    health = tr.messages(CFG.topics.health)
    assert health and set(health[0][0]) >= {"variance", "saturation", "baseline_drift"}
    assert node.baseline == (512, 498, 505, 520)  # recorded Z replies set the baseline


def test_zero_service_sends_z_and_returns_baseline() -> None:
    tr = FakeTransport()
    serial = FakeSerial([])
    serial.on_write = [b"Z,10,20,30,40\n"]
    FsrNode(tr, serial, CFG, CAL, echo=lambda _m: None)
    assert tr.call(CFG.topics.zero, {}) == {"baseline": [10, 20, 30, 40]}
    assert serial.written == [b"Z\n"]


def test_health_flags_saturation() -> None:
    tr = FakeTransport()
    lines = [f"{1000 + i * 5000},4095,500,500,500\n".encode() for i in range(300)]
    node = FsrNode(tr, FakeSerial(lines), CFG, CAL, echo=lambda _m: None)
    for _ in range(300):
        node.step(max_lines=1)
        tr.sleep(0.005)
    msg = tr.messages(CFG.topics.health)[-1][0]
    assert msg["saturation"][0] == 1.0 and msg["saturation"][1] == 0.0
