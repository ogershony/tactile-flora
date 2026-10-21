"""The calibration flow: per pad, press a known-weight stack and fit counts to newtons.

Owns: the operator-driven flow (prompts, reading samples from the serial transport, median
counts per weight, ``fit_pad``, writing the calibration JSON).
Must never: touch ROS or record episodes. Prompts and output go through injected callables so
the flow is testable with a fake serial.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as date_cls
from pathlib import Path

import numpy as np

from tactile_flora.baxter.calibration import CalibrationFile, fit_pad, sensor_warning
from tactile_flora.baxter.serial_protocol import Sample, SerialLike, parse_line
from tactile_flora.config import BaxterConfig

G = 9.80665
MAX_EMPTY_READS = 20000


def read_counts(serial: SerialLike, n: int) -> np.ndarray:
    """Read ``n`` valid samples and return an (n, 4) array of counts."""
    rows: list[tuple[int, int, int, int]] = []
    empty = 0
    while len(rows) < n:
        line = serial.readline()
        if not line:
            empty += 1
            if empty > MAX_EMPTY_READS:
                raise RuntimeError(f"serial timed out after {len(rows)}/{n} samples")
            continue
        parsed = parse_line(line)
        if isinstance(parsed, Sample):
            rows.append(parsed.counts)
    return np.asarray(rows, dtype=np.float64)


def calibrate(
    cfg: BaxterConfig,
    serial: SerialLike,
    prompt: Callable[[str], None],
    out: Path,
    echo: Callable[[str], None] = print,
    today: str | None = None,
) -> Path:
    """Run the interactive calibration and write the file at ``out``.

    For each pad: an unloaded reading (0 N), then one reading per weight in
    ``cfg.calibration_weights_g``; the median over ``cfg.calibration_samples`` samples per step.
    """
    warning = sensor_warning(cfg)
    if warning:
        echo(warning)
    pads = {}
    for ch_idx, channel in enumerate(cfg.channels):
        counts = []
        newtons = []
        prompt(f"[{channel}] remove all load from the pad, then continue")
        counts.append(float(np.median(read_counts(serial, cfg.calibration_samples)[:, ch_idx])))
        newtons.append(0.0)
        for grams in cfg.calibration_weights_g:
            prompt(f"[{channel}] press {grams:g} g through the puck and skin, then continue")
            med = float(np.median(read_counts(serial, cfg.calibration_samples)[:, ch_idx]))
            counts.append(med)
            newtons.append(grams / 1000.0 * G)
        pads[channel] = fit_pad(counts, newtons)
        echo(f"{channel}: fitted {len(pads[channel].counts)} knots")
    cal = CalibrationFile(
        date=today or date_cls.today().isoformat(),
        r_ref_ohm=cfg.r_ref_ohm,
        sensor_confirmed=cfg.sensor_confirmed,
        pads=pads,
    )
    return cal.save(out)
