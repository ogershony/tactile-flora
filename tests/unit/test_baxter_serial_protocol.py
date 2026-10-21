"""Host-side serial parser and clock fit, fed with recorded Teensy lines."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.baxter.serial_protocol import (
    BaselineReply,
    ClockFit,
    Sample,
    parse_line,
)

LINES = (Path(__file__).parents[1] / "fixtures" / "baxter" / "serial_lines.txt").read_text()


def test_recorded_lines() -> None:
    parsed = [parse_line(line + "\n") for line in LINES.splitlines()]
    samples = [p for p in parsed if isinstance(p, Sample)]
    zs = [p for p in parsed if isinstance(p, BaselineReply)]
    bad = [p for p in parsed if p is None]
    assert len(samples) == 44 and len(zs) == 2 and len(bad) == 3
    assert zs[0].baseline == (512, 498, 505, 520)
    assert samples[0].counts[2] == pytest.approx(505, abs=3)
    micros = [s.micros for s in samples]
    assert micros == sorted(micros)


def test_partial_and_bytes() -> None:
    assert parse_line(b"1000,1,2,3,4\n") == Sample(1000, (1, 2, 3, 4))
    assert parse_line(b"1000,1,2,3,4") is None  # no newline: partial
    assert parse_line("1000,1,2\n") is None
    assert parse_line(b"\xff\xfe,1,2,3,4\n") is None
    assert parse_line("Z,1,2,3\n") is None
    assert parse_line("") is None and parse_line("\n") is None
    assert parse_line("1000,-1,2,3,4\n") is None


def test_clock_fit_offset_and_drift() -> None:
    fit = ClockFit(refit_s=5.0)
    slope, offset = 1.0001, 1000.0
    for i in range(2000):  # 10 s at 200 Hz
        micros = (4_000_000_000 + i * 5000) % (1 << 32)  # crosses the 32-bit wrap
        fit.add(micros, offset + slope * (4000.0 + i * 0.005))
    assert fit.n_fits >= 2
    assert fit.slope == pytest.approx(slope, abs=1e-5)
    last = (4_000_000_000 + 1999 * 5000) % (1 << 32)
    assert fit.to_host_time(last) == pytest.approx(
        offset + slope * (4000.0 + 1999 * 0.005), abs=1e-3
    )


def test_clock_fit_provisional() -> None:
    fit = ClockFit(refit_s=5.0)
    fit.add(1_000_000, 50.0)
    assert fit.to_host_time(1_200_000) == pytest.approx(50.2)
