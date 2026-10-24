r"""UDP link to the ESP32 FSR boards: discovery, keepalive, and merging boards into one line stream.

Owns: ``UdpBoards`` (a ``SerialLike`` over UDP that maps each board's micros to host time with its
own ``ClockFit``, merges the boards' channel slots onto one host-time grid and yields the same
``<micros>,<a0>,<a1>,<a2>,<a3>`` and ``Z,<b0>..<b3>`` lines one USB board would), the datagram
header parser and ``open_udp``.
Must never: calibrate, publish, or touch ROS.

Board to host, one datagram: ``#B,<name>,<slots>,<boot>[,<vbat_mv>,<rail_mv>,<dropped>,<read_us>]``
then zero or more protocol lines (four values; slots the board does not own are 0). ``<slots>``
lists the channel indices the board fills (``01`` = L_a, L_b; ``0123`` = all four). A header-only
datagram is the 1 Hz discovery beacon. Host to board: ``H\n`` (stream to this address; repeated
every ``KEEPALIVE_S``) and ``Z\n`` (1 s baseline; each board replies ``Z,..``, merged here).
Merged lines carry host-time micros (``round(host_s * 1e6) mod 2**32``), not board micros;
``to_host_time`` unwraps them, and ``FsrNode`` uses it instead of its own clock fit, so each
sample is stamped by exactly one fit (its board's). ``clock`` must be the transport's clock.
"""

from __future__ import annotations

import contextlib
import math
import select
import socket
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

from tactile_flora.baxter.serial_protocol import (
    N_CHANNELS,
    BaselineReply,
    ClockFit,
    Sample,
    parse_line,
)

DEFAULT_PORT = 9870
BOARD_TIMEOUT_S = 1.0  # a board silent this long leaves the merge (its slots read 0)
FORGET_S = 30.0  # stop sending keepalives to a board silent this long
KEEPALIVE_S = 0.5
STALE_S = 0.1  # a poll this long after the previous one is draining a backlog
HISTORY = 400  # samples kept per board (2 s at 200 Hz)
REORDER_US = 1_000_000  # micros this far behind the last sample are a late datagram, not a reboot
_WRAP = 1 << 32
_MAX_DATAGRAM = 2048

Counts = tuple[int, int, int, int]
Addr = tuple[str, int]


@dataclass(frozen=True)
class Header:
    """The first line of every board datagram."""

    name: str
    slots: tuple[int, ...]
    boot: str
    extra: tuple[int, ...] = ()  # vbat_mv, rail_mv, dropped, read_us when present


def parse_header(line: bytes) -> Header | None:
    """Parse ``#B,<name>,<slots>,<boot>[,<int>...]``; None if malformed."""
    try:
        fields = line.decode("ascii").strip().split(",")
    except UnicodeDecodeError:
        return None
    if len(fields) < 4 or fields[0] != "#B" or not fields[1] or not fields[2].isdigit():
        return None
    slots = tuple(int(c) for c in fields[2])
    if len(set(slots)) != len(slots) or max(slots) >= N_CHANNELS:
        return None
    try:
        extra = tuple(int(f) for f in fields[4:])
    except ValueError:
        return None
    return Header(fields[1], slots, fields[3], extra)


@dataclass
class _Board:
    header: Header
    addr: Addr
    clock: ClockFit
    samples: deque[tuple[float, Counts]] = field(default_factory=lambda: deque(maxlen=HISTORY))
    last_rx: float = -math.inf
    last_micros: int | None = None
    n_samples: int = 0
    n_gaps: int = 0
    zero: Counts | None = None

    def nearest(self, t: float) -> Counts:
        """Return the counts nearest host time ``t``, dropping samples the grid has passed."""
        s = self.samples
        while len(s) >= 2 and s[1][0] <= t:
            s.popleft()
        if len(s) >= 2 and s[1][0] - t < t - s[0][0]:
            return s[1][1]
        return s[0][1]


class UdpBoards:
    """``SerialLike`` over UDP for one or more ESP32 boards (see the module docstring)."""

    def __init__(
        self,
        sock: socket.socket,
        rate_hz: float = 200.0,
        timeout_s: float = 0.01,
        refit_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.sock = sock
        self.sock.setblocking(False)
        self.rate_hz = rate_hz
        self.timeout_s = timeout_s
        self.refit_s = refit_s
        self.clock = clock
        self.boards: dict[Addr, _Board] = {}
        self.n_bad_datagrams = 0
        self._out: deque[bytes] = deque()
        self._grid0: float | None = None
        self._k = 0
        self._zero_wait: set[Addr] | None = None
        self._last_keepalive = -math.inf
        self._last_poll = -math.inf

    # SerialLike -------------------------------------------------------------------------------

    def readline(self) -> bytes:
        """Return the next merged line, waiting up to ``timeout_s``; b"" if none."""
        if not self._out:
            self.poll(self.timeout_s)
        return self._out.popleft() if self._out else b""

    def write(self, data: bytes) -> int:
        """Forward commands to every live board; ``Z`` starts a merged baseline reply."""
        live = self._live(self.clock())
        if data.strip() == b"Z":
            self._zero_wait = {b.addr for b in live} or None
            for b in live:
                b.zero = None
        for b in live:
            self._send(data, b.addr)
        return len(data)

    def to_host_time(self, micros: int) -> float:
        """Host time of a merged line's micros (host-time micros, unwrapped nearest now)."""
        now_us = self.clock() * 1e6
        return (micros + round((now_us - micros) / _WRAP) * _WRAP) / 1e6

    def reset_input_buffer(self) -> None:
        """Drop everything received so far, so the next line is fresh (pyserial's name)."""
        self.poll(0.0)
        self._out.clear()
        for b in self.boards.values():
            b.samples.clear()
        self._grid0 = None

    # receive and merge -------------------------------------------------------------------------

    def poll(self, timeout_s: float = 0.0) -> None:
        """Wait up to ``timeout_s`` for datagrams, ingest every queued one, emit merged lines.

        Datagrams queued while nobody polled (a gap over ``STALE_S``) are a backlog: their arrival
        time is not their send time, so they are mapped with the current clock fit, not fed to it.
        """
        backlog = self.clock() - self._last_poll > STALE_S
        if not backlog:
            select.select([self.sock], [], [], timeout_s)
        while True:
            try:
                data, addr = self.sock.recvfrom(_MAX_DATAGRAM)
            except OSError:  # BlockingIOError when drained
                break
            self.ingest(data, (str(addr[0]), int(addr[1])), self.clock(), fit_clock=not backlog)
        now = self.clock()
        self._last_poll = now
        self._keepalive(now)
        self._emit(now)

    def ingest(self, data: bytes, addr: Addr, now: float, fit_clock: bool = True) -> None:
        """Parse one datagram received from ``addr`` at host time ``now``."""
        lines = data.split(b"\n")
        header = parse_header(lines[0])
        if header is None:
            self.n_bad_datagrams += 1
            return
        board = self.boards.get(addr)
        if board is None or (board.header.boot, board.header.slots) != (header.boot, header.slots):
            board = self._new_board(header, addr)
        board.header = header
        board.last_rx = now
        for raw in lines[1:-1]:  # a datagram's last element is "" or a partial line
            parsed = parse_line(raw + b"\n")
            if isinstance(parsed, Sample):
                if board.last_micros is not None:
                    back = (board.last_micros - parsed.micros) % _WRAP
                    if back < REORDER_US:
                        continue  # duplicate or reordered datagram: already past this sample
                    if back < _WRAP // 2:
                        board = self._new_board(header, addr)  # micros ran backwards: a reboot
                        board.last_rx = now
                self._add_sample(board, parsed, now, fit_clock)
            elif isinstance(parsed, BaselineReply):
                board.zero = parsed.baseline
        self._finish_zero()

    def _new_board(self, header: Header, addr: Addr) -> _Board:
        board = _Board(header, addr, ClockFit(self.refit_s, min_span_s=self.refit_s))
        self.boards[addr] = board
        self._send(b"H\n", addr)
        return board

    def _add_sample(self, board: _Board, s: Sample, now: float, fit_clock: bool) -> None:
        if board.last_micros is not None:
            if (s.micros - board.last_micros) % _WRAP > 1.5e6 / self.rate_hz:
                board.n_gaps += 1
        elif not fit_clock:
            return  # no fit yet, so a backlog sample cannot be placed in host time
        board.last_micros = s.micros
        if fit_clock:
            board.clock.add(s.micros, now)
        board.samples.append((board.clock.to_host_time(s.micros), s.counts))
        board.n_samples += 1

    def _finish_zero(self) -> None:
        if not self._zero_wait:
            return
        merged = [0] * N_CHANNELS
        for addr in self._zero_wait:
            board = self.boards.get(addr)
            if board is None or board.zero is None:
                return
            for slot in board.header.slots:
                merged[slot] = board.zero[slot]
        self._out.append(("Z," + ",".join(map(str, merged)) + "\n").encode("ascii"))
        self._zero_wait = None

    def _live(self, now: float) -> list[_Board]:
        return [b for b in self.boards.values() if now - b.last_rx <= BOARD_TIMEOUT_S]

    def _emit(self, now: float) -> None:
        """Emit one merged line per grid step that every live board's samples already cover."""
        live = [b for b in self._live(now) if b.samples]
        if not live:
            return
        step = 1.0 / self.rate_hz
        start = max(b.samples[0][0] for b in live)
        if self._grid0 is None:
            self._grid0, self._k = start, 0
        elif self._grid0 + self._k * step < start:  # a board joined or the stream resumed
            self._k = math.ceil((start - self._grid0) / step)
        horizon = min(b.samples[-1][0] for b in live)
        while (t := self._grid0 + self._k * step) <= horizon:
            counts = [0] * N_CHANNELS
            for b in live:
                c = b.nearest(t)
                for slot in b.header.slots:
                    counts[slot] = c[slot]
            micros = round(t * 1e6) % _WRAP
            self._out.append(f"{micros},{','.join(map(str, counts))}\n".encode("ascii"))
            self._k += 1

    def _keepalive(self, now: float) -> None:
        if now - self._last_keepalive < KEEPALIVE_S:
            return
        self._last_keepalive = now
        for b in self.boards.values():
            if now - b.last_rx <= FORGET_S:
                self._send(b"H\n", b.addr)

    def _send(self, data: bytes, addr: Addr) -> None:
        with contextlib.suppress(OSError):  # network down; the next keepalive retries
            self.sock.sendto(data, addr)

    def status(self) -> list[dict[str, object]]:
        """One dict per known board: identity, liveness, sample and gap counts, header extras."""
        now = self.clock()
        names = ("vbat_mv", "rail_mv", "dropped", "read_us")
        return [
            {
                "name": b.header.name,
                "slots": "".join(map(str, b.header.slots)),
                "addr": f"{b.addr[0]}:{b.addr[1]}",
                "live": now - b.last_rx <= BOARD_TIMEOUT_S,
                "samples": b.n_samples,
                "gaps": b.n_gaps,
                **dict(zip(names, b.header.extra, strict=False)),
            }
            for b in self.boards.values()
        ]

    def close(self) -> None:
        """Close the socket."""
        self.sock.close()


def open_udp(
    url: str,
    rate_hz: float = 200.0,
    timeout_s: float = 0.01,
    clock: Callable[[], float] = time.time,
) -> UdpBoards:
    """Bind ``udp://<host>:<port>`` (default ``0.0.0.0:9870``) and return the board link."""
    parsed = urlparse(url)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((parsed.hostname or "0.0.0.0", parsed.port or DEFAULT_PORT))
    return UdpBoards(sock, rate_hz=rate_hz, timeout_s=timeout_s, clock=clock)
