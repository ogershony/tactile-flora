"""The calibration flow: push one pad at a time against a kitchen scale and type its reading.

Owns: the operator-driven flow (prompts, reading samples from the serial transport, median
counts per push, point screening, ``fit_pad``, merging into and writing the calibration JSON,
the per-pad report).
Must never: touch ROS or record episodes. Prompts and output go through injected callables so
the flow is testable with a fake serial.

Rig (tactile_flora/baxter/firmware/esp32_fsr/README.md, "Calibration rig"): a kitchen scale stands
on its edge, braced behind its base, and is tared there. One pad at a time is pushed against the
platform; the scale then reads the force through that pad, provided the puck is the pad's only
contact and nothing but the base touches the table or the brace. Pads are done in the order
given. Per push, the median of ``cfg.calibration_samples`` fresh samples of the pushed pad is
paired with the grams the operator reads off the scale while still holding. Points at the
unloaded floor or at saturation are dropped (they do not pin the map down); points above
``cfg.linear_max_counts`` are kept but flagged. The file is written after every pad, so an abort
keeps the pads already finished.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date as date_cls
from pathlib import Path

import numpy as np

from tactile_flora.baxter.calibration import (
    CalibrationFile,
    PadRecord,
    PiecewiseLinear,
    fit_pad,
    sensor_warning,
)
from tactile_flora.baxter.serial_protocol import Sample, SerialLike, parse_line
from tactile_flora.config import BaxterConfig

G = 9.80665
MAX_EMPTY_READS = 20000
METHOD = "scale_push"
MAX_SCALE_G = 10000.0  # largest believable kitchen-scale reading
# A push whose median is within this many counts (or 3 unloaded sd) of the unloaded median
# registered nothing: the ESP32 reads 0 V below ~0.15 V, so light force sits on the floor.
FLOOR_MARGIN_COUNTS = 2.0
# Capture sd above both of these means the force changed during the 1 s capture.
UNSTEADY_MIN_COUNTS = 5.0
UNSTEADY_FRAC = 0.03


def read_counts(serial: SerialLike, n: int) -> np.ndarray:
    """Read ``n`` fresh valid samples and return an (n, 4) array of counts.

    Lines buffered while the operator was setting up the push are dropped first (pyserial and
    ``UdpBoards`` both offer ``reset_input_buffer``; a UDP backlog can be seconds long).
    """
    flush = getattr(serial, "reset_input_buffer", None)
    if callable(flush):
        flush()
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


def _ask_grams(ask: Callable[[str], str], echo: Callable[[str], None], label: str) -> float | None:
    """Ask for the scale reading until it parses; None when the operator discards the push."""
    while True:
        reply = ask(f"{label} scale reading in grams (r discards this push)").strip().lower()
        if reply == "r":
            return None
        try:
            grams = float(reply.removesuffix("g"))
        except ValueError:
            grams = float("nan")
        if 0.0 < grams <= MAX_SCALE_G:
            return grams
        echo(f"not a reading in grams: {reply!r}")


def _existing(
    out: Path, r_ref_ohm: float, echo: Callable[[str], None]
) -> tuple[dict[str, PiecewiseLinear], dict[str, PadRecord]]:
    """Return the fits already in ``out`` if they were made at the same R_ref, else nothing."""
    if not out.exists():
        return {}, {}
    try:
        old = CalibrationFile.load(out)
    except (OSError, ValueError) as exc:
        echo(f"ignoring unreadable {out}: {exc}")
        return {}, {}
    if old.r_ref_ohm != r_ref_ohm:
        echo(
            f"discarding {', '.join(old.pads)} from {out}: fitted at R_ref {old.r_ref_ohm:g} ohm, "
            f"the config now says {r_ref_ohm:g}"
        )
        return {}, {}
    echo(f"existing fits in {out} (from {old.date}): {', '.join(old.pads)}; pads redone replace")
    return dict(old.pads), dict(old.records)


def report(cfg: BaxterConfig, pad: str, rec: PadRecord) -> list[str]:
    """Summarize one pad's points: the table, floor, clamp, saturation and an R_ref estimate.

    R_fsr = R_ref * (adc_max / counts - 1) assumes ratiometric counts (true for the ESP32
    firmware with its rail monitor, and for the Teensy); above ``linear_max_counts`` it is biased.
    """
    full = float(cfg.adc_max_counts)
    lines = [f"{pad}: unloaded {rec.unloaded_counts:.0f} counts, {len(rec.points)} pushes fitted"]
    for grams, c in sorted(rec.points):
        r_fsr = cfg.r_ref_ohm * (full / c - 1.0)
        flag = "  above the linear range" if c > cfg.linear_max_counts else ""
        newtons = grams / 1000 * G
        lines.append(
            f"  {grams:7.0f} g {newtons:7.2f} N {c:7.0f} counts  R_fsr {r_fsr:7.0f} ohm{flag}"
        )
    below = [g for g, _, why in rec.dropped if why == "below detection"]
    sat = [g for g, _, why in rec.dropped if why == "saturated"]
    if below:
        lines.append(f"  reads nothing up to {max(below):g} g")
    if rec.points:
        top = max(g for g, _ in rec.points)
        lines.append(f"  forces above {top:g} g read as {top / 1000 * G:.2f} N (the fit clamps)")
    if sat:
        lines.append(
            f"  saturated from {min(sat):g} g: R_ref {cfg.r_ref_ohm:g} ohm is too large for that"
        )
    linear = [(g, c) for g, c in rec.points if c <= cfg.linear_max_counts]
    if linear:
        g_lin, c_lin = max(linear)
        r_fsr = cfg.r_ref_ohm * (full / c_lin - 1.0)
        target = r_fsr * cfg.linear_max_counts / (full - cfg.linear_max_counts)
        heavier = bool(sat) or any(c > cfg.linear_max_counts for _, c in rec.points)
        lines.append(
            f"  R_ref that puts {g_lin:g} g at {cfg.linear_max_counts} counts: {target:.0f} ohm"
            + (" (an upper bound: the heavier pushes need less)" if heavier else "")
        )
    return lines


def _calibrate_pad(
    cfg: BaxterConfig,
    serial: SerialLike,
    ask: Callable[[str], str],
    echo: Callable[[str], None],
    pad: str,
    cal_date: str,
) -> tuple[PiecewiseLinear, PadRecord]:
    """Run one pad: its unloaded reading, then pushes until ``done`` with enough usable points."""
    col = cfg.channels.index(pad)
    n = cfg.calibration_samples
    ask(f"[{pad}] move the scale clear so nothing touches {pad}, tare the scale, press enter")
    rest = read_counts(serial, n)[:, col]
    unloaded = float(np.median(rest))
    floor = unloaded + max(FLOOR_MARGIN_COUNTS, 3.0 * float(rest.std()))
    echo(f"[{pad}] unloaded {unloaded:.0f} counts")

    points: list[tuple[float, float]] = []
    dropped: list[tuple[float, float, str]] = []
    k = 1
    while True:
        label = f"[{pad} push {k}]"
        reply = ask(
            f"{label} push {pad} against the scale, hold until it settles, press enter "
            "(done ends this pad)"
        )
        if reply.strip().lower() == "done":
            if len(points) >= cfg.calibration_min_points:
                break
            echo(
                f"[{pad}] has {len(points)} usable pushes, needs {cfg.calibration_min_points} "
                "(Ctrl-C aborts; pads already finished are saved)"
            )
            continue
        block = read_counts(serial, n)[:, col]
        c, sd = float(np.median(block)), float(block.std())
        echo(f"{label} captured {c:.0f} counts (sd {sd:.1f})")
        if sd > max(UNSTEADY_MIN_COUNTS, UNSTEADY_FRAC * (c - unloaded)):
            echo(f"{label} the force moved during the capture; r redoes it")
        grams = _ask_grams(ask, echo, label)
        if grams is None:
            echo(f"{label} discarded")
            continue
        if c >= cfg.saturation_counts:
            dropped.append((grams, c, "saturated"))
            echo(f"{label} saturated at {grams:g} g: dropped")
        elif c <= floor:
            dropped.append((grams, c, "below detection"))
            echo(f"{label} reads nothing at {grams:g} g: dropped")
        else:
            points.append((grams, c))
            if c > cfg.linear_max_counts:
                echo(f"{label} {c:.0f} counts is past the linear range: kept, flagged")
        k += 1

    fit = fit_pad([unloaded] + [c for _, c in points], [0.0] + [g / 1000.0 * G for g, _ in points])
    rec = PadRecord(
        date=cal_date, method=METHOD, unloaded_counts=unloaded, points=points, dropped=dropped
    )
    return fit, rec


def calibrate(
    cfg: BaxterConfig,
    serial: SerialLike,
    ask: Callable[[str], str],
    out: Path,
    pads: Sequence[str],
    echo: Callable[[str], None] = print,
    today: str | None = None,
) -> Path:
    """Calibrate each of ``pads`` in turn, one pad pushed against the scale at a time.

    ``ask`` shows a prompt and returns the operator's reply. Per pad: an unloaded reading (0 N),
    then pushes until the operator types ``done`` with at least ``cfg.calibration_min_points``
    usable pushes. ``out`` is rewritten after every pad; fits for channels already in it are
    kept when they were made with the same ``r_ref_ohm``, and a pad redone replaces its fit.
    """
    unknown = [p for p in pads if p not in cfg.channels]
    if not pads or unknown:
        raise ValueError(f"pads must be channels of {cfg.channels}, got {list(pads)}")
    warning = sensor_warning(cfg)
    if warning:
        echo(warning)
    cal_date = today or date_cls.today().isoformat()
    fits, records = _existing(out, cfg.r_ref_ohm, echo)
    for pad in pads:
        fits[pad], records[pad] = _calibrate_pad(cfg, serial, ask, echo, pad, cal_date)
        for line in report(cfg, pad, records[pad]):
            echo(line)
        CalibrationFile(
            date=cal_date,
            r_ref_ohm=cfg.r_ref_ohm,
            sensor_confirmed=cfg.sensor_confirmed,
            pads=fits,
            records=records,
        ).save(out)
        echo(f"[{pad}] saved to {out}")
    return out
