"""Calibration math and the interactive flow against a fake serial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tactile_flora.baxter.calibrate import G, calibrate
from tactile_flora.baxter.calibration import CalibrationFile, fit_pad, sensor_warning
from tactile_flora.config import BaxterConfig, load_config

CONFIGS = Path(__file__).parents[2] / "configs"


def _cfg(**kw: object) -> BaxterConfig:
    cfg = load_config(CONFIGS / "baxter.yaml", BaxterConfig)
    return cfg.model_copy(update={"calibration_samples": 5, **kw})


def test_fit_is_monotone_and_interpolates() -> None:
    fit = fit_pad([100, 200, 300, 400], [0.0, 2.0, 1.0, 5.0])  # one violation
    assert all(b >= a for a, b in zip(fit.newtons, fit.newtons[1:], strict=False))
    assert fit.newtons[1] == pytest.approx(1.5) and fit.newtons[2] == pytest.approx(1.5)
    assert fit.apply(350)[()] == pytest.approx(3.25)
    assert fit.apply([0, 1000]).tolist() == [0.0, 5.0]  # clamped


def test_fit_averages_duplicate_counts() -> None:
    fit = fit_pad([10, 10, 20], [0.0, 2.0, 4.0])
    assert fit.counts == [10.0, 20.0] and fit.newtons == [1.0, 4.0]
    with pytest.raises(ValueError):
        fit_pad([5, 5], [1, 2])


def test_file_roundtrip(tmp_path: Path) -> None:
    cal = CalibrationFile(
        date="2026-11-01",
        r_ref_ohm=4700.0,
        sensor_confirmed=False,
        pads={"L_a": fit_pad([0, 10], [0, 1])},
    )
    path = cal.save(tmp_path / "c.json")
    assert CalibrationFile.load(path) == cal
    assert cal.version == "cal-2026-11-01-unconfirmed"


def test_sensor_warning_follows_flag() -> None:
    assert sensor_warning(_cfg()) is not None
    assert sensor_warning(_cfg(sensor_confirmed=True)) is None


class _PressSerial:
    """Serial whose counts follow whatever the operator 'pressed' last (set by the prompt)."""

    def __init__(self) -> None:
        self.counts = [500, 500, 500, 500]

    def readline(self) -> bytes:
        return ("1000," + ",".join(str(c) for c in self.counts) + "\n").encode()

    def write(self, data: bytes) -> int:
        return len(data)


def test_calibrate_flow(tmp_path: Path) -> None:
    cfg = _cfg(calibration_weights_g=[100.0, 500.0, 2000.0])
    serial = _PressSerial()
    log: list[str] = []

    def prompt(msg: str) -> None:
        pad = cfg.channels.index(msg[1:4])
        base = [500] * 4
        if "press" in msg:
            grams = float(msg.split("press ")[1].split(" g")[0])
            base[pad] = int(500 + grams * 0.5)
        serial.counts = base

    out = calibrate(cfg, serial, prompt, tmp_path / "cal.json", echo=log.append, today="2026-11-01")
    cal = CalibrationFile.load(out)
    assert log[0].startswith("WARNING") and set(cal.pads) == set(cfg.channels)
    assert cal.pads["R_b"].apply(1500)[()] == pytest.approx(2.0 * G)
    assert np.all(np.diff(cal.pads["L_a"].newtons) >= 0)
