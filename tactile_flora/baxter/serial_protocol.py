"""Host side of the Teensy serial protocol: line parser, serial interface and the clock fit.

Owns: ``parse_line`` for ``<micros>,<a0>,<a1>,<a2>,<a3>`` samples and ``Z,<b0>..<b3>`` baseline
replies, the ``SerialLike`` Protocol the node and calibrator read from, ``open_serial`` (lazy
pyserial) and ``ClockFit`` (Teensy micros to host time, offset and drift, refit periodically).
Must never: calibrate, publish, or touch ROS.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

N_CHANNELS = 4
_WRAP = 1 << 32
MAX_COUNT = 65535


class SerialLike(Protocol):
    """The slice of pyserial the stack uses; tests feed recorded lines through a fake."""

    def readline(self) -> bytes:
        """Return one line including the newline, or b"" on timeout / no data."""
        ...

    def write(self, data: bytes) -> int | None:
        """Write bytes to the device."""
        ...


@dataclass(frozen=True)
class Sample:
    """One 200 Hz tick: Teensy micros plus four raw ADC counts."""

    micros: int
    counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class BaselineReply:
    """The Teensy's reply to ``Z``: the 1 s mean count per channel."""

    baseline: tuple[int, int, int, int]


def _ints(fields: list[str], hi: int = MAX_COUNT) -> list[int] | None:
    try:
        vals = [int(f) for f in fields]
    except ValueError:
        return None
    return vals if all(0 <= v <= hi for v in vals) else None


def parse_line(line: bytes | str) -> Sample | BaselineReply | None:
    """Parse one serial line; None for partial, corrupted or empty lines.

    A line must be newline-terminated (a read that timed out mid-line is a partial line).
    """
    if isinstance(line, bytes):
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError:
            return None
    else:
        text = line
    if not text.endswith("\n"):
        return None
    fields = text.strip().split(",")
    if fields[0] == "Z":
        vals = _ints(fields[1:])
        if vals is None or len(vals) != N_CHANNELS:
            return None
        return BaselineReply((vals[0], vals[1], vals[2], vals[3]))
    micros = _ints(fields[:1], _WRAP - 1)
    vals = _ints(fields[1:])
    if micros is None or vals is None or len(vals) != N_CHANNELS:
        return None
    return Sample(micros[0], (vals[0], vals[1], vals[2], vals[3]))


def open_serial(port: str, baud: int, timeout_s: float = 0.01) -> SerialLike:
    """Open a pyserial port (lazy import; needs the [baxter] extra)."""
    import importlib

    try:
        serial = importlib.import_module("serial")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyserial is not installed (uv sync --extra baxter)") from exc
    return serial.Serial(port, baud, timeout=timeout_s)  # type: ignore[no-any-return]


class ClockFit:
    """Linear map from Teensy micros to host time: ``host = slope * teensy_s + offset``.

    ``add`` records (micros, host arrival time) pairs; the line is refit at the first pair that
    makes a fit possible and then every ``refit_s`` of host time, which captures the offset and
    the oscillator drift. The offset is shifted to the lower envelope of the pairs (serial and
    scheduling latency only ever delay arrival). The 32-bit micros counter is unwrapped.
    """

    def __init__(self, refit_s: float, max_pairs: int = 4000) -> None:
        self.refit_s = refit_s
        self._pairs: deque[tuple[float, float]] = deque(maxlen=max_pairs)
        self._last_u = 0.0
        self._has_last = False
        self._slope = 1.0
        self._offset = 0.0
        self._last_fit_host = float("-inf")
        self.n_fits = 0

    def _unwrap(self, micros: int) -> float:
        if not self._has_last:
            return micros / 1e6
        last = self._last_u * 1e6
        k = round((last - micros) / _WRAP)
        return (micros + k * _WRAP) / 1e6

    @property
    def slope(self) -> float:
        """Fitted seconds of host time per Teensy second (1 + drift)."""
        return self._slope

    @property
    def offset(self) -> float:
        """Fitted host-time intercept in seconds."""
        return self._offset

    def add(self, micros: int, host_time: float) -> None:
        """Record one sample's arrival and refit if due."""
        u = self._unwrap(micros)
        self._last_u, self._has_last = u, True
        self._pairs.append((u, host_time))
        if len(self._pairs) == 1:
            self._offset = host_time - u  # provisional: slope 1, offset from the first pair
            return
        if self.n_fits == 0 or host_time - self._last_fit_host >= self.refit_s:
            self._refit(host_time)

    def _refit(self, host_time: float) -> None:
        arr = np.asarray(self._pairs, dtype=np.float64)
        u, h = arr[:, 0], arr[:, 1]
        if float(np.ptp(u)) < 1e-3:
            return
        slope, offset = np.polyfit(u - u[0], h, 1)
        offset = float(offset) + float(np.min(h - (slope * (u - u[0]) + offset)))
        self._slope = float(slope)
        self._offset = offset - self._slope * float(u[0])
        self._last_fit_host = host_time
        self.n_fits += 1

    def to_host_time(self, micros: int) -> float:
        """Map a Teensy micros value (near the most recent sample) to host time in seconds."""
        return self._slope * self._unwrap(micros) + self._offset
