"""Reading the Teensy serial stream and publishing /tactile/raw, /tactile/force, /tactile/health.

Owns: ``FsrNode`` (serial lines to clock-fitted, calibrated, stamped topic messages, the 1 Hz
health summary and the zero/baseline service).
Must never: calibrate, record episodes, or import rospy (all ROS access is the Transport).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from tactile_flora.baxter.calibration import CalibrationFile, sensor_warning
from tactile_flora.baxter.ros.transport import Msg, Transport
from tactile_flora.baxter.serial_protocol import (
    BaselineReply,
    ClockFit,
    Sample,
    SerialLike,
    parse_line,
)
from tactile_flora.config import BaxterConfig

ZERO_TIMEOUT_S = 5.0


class FsrNode:
    """Bridge between the Teensy serial stream and the transport."""

    def __init__(
        self,
        transport: Transport,
        serial: SerialLike,
        cfg: BaxterConfig,
        calibration: CalibrationFile,
        echo: Callable[[str], None] = print,
    ) -> None:
        self.transport = transport
        self.serial = serial
        self.cfg = cfg
        self.calibration = calibration
        self.clock = ClockFit(cfg.clock_refit_s)
        self.baseline: tuple[int, ...] | None = None
        self.n_samples = 0
        self.n_bad_lines = 0
        self._window: deque[tuple[int, int, int, int]] = deque(
            maxlen=max(2, int(cfg.sample_rate_hz * cfg.health_period_s))
        )
        self._last_health = transport.now()
        self._reply: BaselineReply | None = None
        warning = sensor_warning(cfg)
        if warning:
            echo(warning)
        transport.advertise(cfg.topics.zero, self._on_zero)

    def _handle(self, line: bytes) -> None:
        parsed = parse_line(line)
        if parsed is None:
            self.n_bad_lines += 1
        elif isinstance(parsed, BaselineReply):
            self._reply = parsed
            self.baseline = parsed.baseline
        else:
            self._publish_sample(parsed)

    def _publish_sample(self, s: Sample) -> None:
        self.clock.add(s.micros, self.transport.now())
        stamp = self.clock.to_host_time(s.micros)
        counts = [float(c) for c in s.counts]
        forces = self.calibration.apply(self.cfg.channels, counts)
        self.transport.publish(self.cfg.topics.raw, {"data": counts}, stamp)
        self.transport.publish(self.cfg.topics.force, {"data": forces}, stamp)
        self._window.append(s.counts)
        self.n_samples += 1

    def _publish_health(self) -> None:
        if len(self._window) < 2:
            return
        arr = np.asarray(self._window, dtype=np.float64)
        drift = (
            (arr.mean(axis=0) - np.asarray(self.baseline, dtype=np.float64)).tolist()
            if self.baseline is not None
            else [0.0] * arr.shape[1]
        )
        msg: Msg = {
            "variance": arr.var(axis=0).tolist(),
            "saturation": (arr >= self.cfg.saturation_counts).mean(axis=0).tolist(),
            "baseline_drift": drift,
            "sensor_confirmed": self.cfg.sensor_confirmed,
        }
        self.transport.publish(self.cfg.topics.health, msg, self.transport.now())

    def step(self, max_lines: int = 64) -> int:
        """Read up to ``max_lines`` serial lines, publish; returns the number of lines read."""
        n = 0
        while n < max_lines:
            line = self.serial.readline()
            if not line:
                break
            n += 1
            self._handle(line)
        now = self.transport.now()
        if now - self._last_health >= self.cfg.health_period_s:
            self._publish_health()
            self._last_health = now
        return n

    def _on_zero(self, _request: Msg) -> Msg:
        """Service handler: send ``Z`` and wait for the baseline echo."""
        self._reply = None
        self.serial.write(b"Z\n")
        deadline = self.transport.now() + ZERO_TIMEOUT_S
        while self._reply is None and self.transport.now() < deadline:
            if self.step() == 0:
                self.transport.sleep(0.01)
        if self._reply is None:
            raise RuntimeError("no baseline reply from the Teensy")
        return {"baseline": list(self._reply.baseline)}

    def run(
        self, max_steps: int | None = None, is_shutdown: Callable[[], bool] = lambda: False
    ) -> None:
        """Loop until shutdown (or ``max_steps`` steps, for tests)."""
        steps = 0
        while not is_shutdown() and (max_steps is None or steps < max_steps):
            if self.step() == 0:
                self.transport.sleep(0.001)
            steps += 1


def main() -> None:  # pragma: no cover - needs ROS and hardware
    """Entry point: ``python -m tactile_flora.baxter.ros.fsr_node --config configs/baxter.yaml``."""
    import argparse

    from tactile_flora.baxter.ros.transport import RosTransport
    from tactile_flora.baxter.serial_protocol import open_serial
    from tactile_flora.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baxter.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, BaxterConfig)
    node = FsrNode(
        RosTransport("fsr_node"),
        open_serial(cfg.serial_port, cfg.baud),
        cfg,
        CalibrationFile.load(cfg.calibration_path),
    )
    node.run()


if __name__ == "__main__":
    main()
