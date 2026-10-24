"""Live text readout of the FSR stream for bench bring-up (no ROS needed).

Owns: ``monitor`` (read lines from any ``SerialLike``, print the sample rate, the latest counts per
channel, newtons when a calibration is given, and per-board status for the UDP link).
Must never: publish, record, calibrate, or write files.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import cast

from tactile_flora.baxter.calibration import CalibrationFile
from tactile_flora.baxter.serial_protocol import BaselineReply, Sample, SerialLike, parse_line


def _board_text(st: dict[str, object]) -> str:
    text = f"{st['name']}[{st['slots']}] {'up' if st['live'] else 'DOWN'} gaps {st['gaps']}"
    for key, label in (
        ("vbat_mv", "vbat"),
        ("rail_mv", "3v3"),
        ("dropped", "dropped"),
        ("read_us", "read_us"),
    ):
        if key in st:
            text += f" {label} {st[key]}"
    return text


def monitor(
    serial: SerialLike,
    channels: Sequence[str],
    calibration: CalibrationFile | None = None,
    echo: Callable[[str], None] = print,
    period_s: float = 0.5,
    duration_s: float | None = None,
    zero: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Print one status line every ``period_s`` until ``duration_s`` (forever if None).

    ``zero`` sends ``Z`` first (gripper open, nothing touching the pads). Returns the number of
    samples read.
    """
    if zero:
        serial.write(b"Z\n")
    t0 = last = clock()
    n = n_bad = total = 0
    latest: tuple[int, ...] | None = None
    while duration_s is None or clock() - t0 < duration_s:
        line = serial.readline()
        if line:
            parsed = parse_line(line)
            if isinstance(parsed, Sample):
                latest, n, total = parsed.counts, n + 1, total + 1
            elif isinstance(parsed, BaselineReply):
                echo(
                    "baseline "
                    + " ".join(f"{c} {v}" for c, v in zip(channels, parsed.baseline, strict=True))
                )
            else:
                n_bad += 1
        now = clock()
        if now - last < period_s:
            continue
        parts = [f"{n / (now - last):6.1f} Hz"]
        if latest is not None:
            newtons = calibration.apply(channels, list(map(float, latest))) if calibration else None
            for i, (name, count) in enumerate(zip(channels, latest, strict=True)):
                parts.append(f"{name} {count:4d}" + (f" ({newtons[i]:5.2f} N)" if newtons else ""))
        if n_bad:
            parts.append(f"bad {n_bad}")
        status = getattr(serial, "status", None)
        if callable(status):
            parts.extend(_board_text(st) for st in cast("list[dict[str, object]]", status()))
        echo("  ".join(parts))
        n = n_bad = 0
        last = now
    return total
