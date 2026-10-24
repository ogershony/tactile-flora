#!/usr/bin/env python3
"""Fingertip tactile sensors: four FSR pads (two per gripper) read by a Teensy 4.0 over USB serial.

Driver: tactile-flora (github.com/ogershony/tactile-flora @533e4e6), in .venv-lerobot:
    pip install --no-deps --ignore-requires-python \\
        "tactile-flora @ git+https://github.com/ogershony/tactile-flora.git@533e4e6d9f35a68d648a140dfcbfd2bb668d7f55"
    pip install pyserial "pydantic>=2" typer rich
(--no-deps: its research stack -- zarr, h5py, tokenizers, pillow>=13 -- isn't needed to read the
sensors.) Its firmware (scripts/flash_tactile.sh) prints "<micros>,<a0>,<a1>,<a2>,<a3>" at 200 Hz:
raw 12-bit counts, rising with force, on A0..A3 = L_a, L_b, R_a, R_b. We use its line parser,
Teensy-clock fit and calibration, not its ROS node (that needs rospy; we only reach ROS through
rosbridge, and the Teensy plugs into this laptop anyway).

Newtons only once config/tactile_calibration.json exists (press known weights on each pad):
    .venv-lerobot/bin/tf baxter calibrate --config config/tactile.yaml

    python bridge/tactile.py              # wait for the Teensy, then show the pads live
    python bridge/tactile.py --check 5    # 5 s health report; exit 1 on no / bad data

    from tactile import TactileStream
    tac = TactileStream()
    counts, newtons, t = tac.latest()     # [4] counts, [4] N or None, t on time.monotonic()
    tac.start_log("logs/run_tactile.csv", t0)   # every 200 Hz sample, t relative to t0
"""

import argparse
import csv
import glob
import os
import sys
import threading
import time
from collections import deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "tactile.yaml")
TEENSY_VID = 0x16C0  # PJRC; the firmware's "USB Serial" type enumerates as 16c0:0483
# 2026-10-24: the Teensy can also come in through a Silicon Labs CP2104 USB-UART adapter
# (10c4:ea60, /dev/ttyUSB*), same lines at 921600 baud (config baud; native USB ignores it)
ADAPTER_VID = 0x10C4


def find_port():
    """The Teensy's serial device (its /dev/serial/by-id link when there is one), or None:
    native USB first, else the CP210x adapter."""
    from serial.tools import list_ports

    ports = list_ports.comports()
    for vid, kind in ((TEENSY_VID, "ttyACM"), (ADAPTER_VID, "ttyUSB")):
        for p in ports:
            if p.vid == vid and kind in p.device:
                for link in glob.glob("/dev/serial/by-id/*"):
                    if os.path.realpath(link) == p.device:
                        return link
                return p.device
    return None


def load_config(path=CONFIG):
    from tactile_flora.config import BaxterConfig, load_config as load

    return load(path, BaxterConfig)


def load_calibration(cfg):
    """The per-pad counts->N fit, or None while the pads are uncalibrated."""
    from tactile_flora.baxter.calibration import CalibrationFile

    path = os.path.join(ROOT, cfg.calibration_path)
    return CalibrationFile.load(path) if os.path.exists(path) else None


def clock_fit(refit_s):
    """tactile-flora's ClockFit (Teensy micros -> host time), except that its first fit waits for
    refit_s of samples. Upstream fits the first two samples, 5 ms apart, so their arrival jitter
    puts the slope anywhere until the next fit: simulated 1.5 ms USB/scheduler latency gave
    timestamps 0.9 s off (median; up to 9 s) for the first 5 s. Until then: slope 1 (a Teensy's
    crystal drifts ppm), offset from the first sample. Uses ClockFit internals (pinned 533e4e6)."""
    from tactile_flora.baxter.serial_protocol import ClockFit

    class Clock(ClockFit):
        def _refit(self, host_time):
            if self._pairs[-1][0] - self._pairs[0][0] >= self.refit_s:
                ClockFit._refit(self, host_time)

    return Clock(refit_s)


class TactileStream(object):
    """The Teensy read continuously on a thread; latest() never blocks.

    Waits for the Teensy to be plugged in and survives unplug / replug (the clock fit restarts:
    a replugged Teensy restarts its micros counter). Sample times are the Teensy's own clock
    fitted onto time.monotonic(), so they line up with camera frames and the control log.
    """

    def __init__(self, port=None, config=CONFIG):
        from tactile_flora.baxter.serial_protocol import Sample, open_serial, parse_line

        self._Sample, self._open, self._parse = Sample, open_serial, parse_line
        self.cfg = load_config(config)
        self.channels = list(self.cfg.channels)
        self.calibration = load_calibration(self.cfg)
        self.port_arg, self.port, self.connected, self.error = port, None, False, None
        self.n_samples = self.n_bad = self.n_lines = self.connects = 0
        self._latest = (None, None, None)
        self._hist = deque(maxlen=int(self.cfg.sample_rate_hz * 10))
        self._lock = threading.Lock()
        self._log = self._csv = None
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        ser = None
        while self._run:
            if ser is None:
                port = self.port_arg or find_port()
                if port is None:
                    time.sleep(0.5)
                    continue
                try:
                    ser = self._open(port, self.cfg.baud, timeout_s=0.05)
                    ser.reset_input_buffer()
                except OSError as e:  # serial.SerialException is an OSError
                    self.error = "%s: %s" % (port, e)
                    time.sleep(0.5)
                    continue
                self.port, self.connected, self.error = port, True, None
                self.clock = clock_fit(self.cfg.clock_refit_s)
                self.connects += 1
            try:
                line = ser.readline()
            except OSError as e:  # unplugged
                self.connected, self.error = False, "%s lost: %s" % (self.port, e)
                try:
                    ser.close()
                except OSError:
                    pass
                ser = None
                continue
            if not line:
                continue
            self.n_lines += 1
            parsed = self._parse(line)
            if isinstance(parsed, self._Sample):
                self._sample(parsed)
            elif parsed is None:  # partial / corrupt line (Z-baseline replies are ignored)
                self.n_bad += 1
        if ser is not None:
            ser.close()

    def _sample(self, s):
        self.clock.add(s.micros, time.monotonic())
        t = self.clock.to_host_time(s.micros)
        counts = np.array(s.counts, float)
        newtons = (
            np.array(self.calibration.apply(self.channels, counts)) if self.calibration else None
        )
        self._latest = (counts, newtons, t)
        with self._lock:
            self._hist.append((t, s.counts))
            if self._csv is not None:
                self._csv.writerow(
                    ["%.4f" % (t - self._t0), s.micros]
                    + list(s.counts)
                    + (["%.3f" % v for v in newtons] if newtons is not None else [])
                )
        self.n_samples += 1

    def latest(self):
        """(counts [4], newtons [4] or None, monotonic sample time), or (None, None, None)."""
        return self._latest

    def wait(self, timeout=None):
        """Block until a sample has arrived (True) or `timeout` s pass (False)."""
        end = None if timeout is None else time.monotonic() + timeout
        while self.n_samples == 0:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.05)
        return True

    def health(self, seconds=1.0):
        """Per-pad counts over the last `seconds` (mean/std/min/max, saturated fraction), the
        sample rate and problems (saturated; flat = no change at all). None without data."""
        with self._lock:
            hist = list(self._hist)
        hist = [h for h in hist if h[0] >= time.monotonic() - seconds]
        if len(hist) < 2:
            return None
        arr = np.array([c for _, c in hist], float)
        span = hist[-1][0] - hist[0][0]
        sat = (arr >= self.cfg.saturation_counts).mean(axis=0)
        problems = {}
        for i, ch in enumerate(self.channels):
            if sat[i] > self.cfg.saturation_frac_max:
                problems[ch] = "saturated %.0f%% of samples" % (100 * sat[i])
            elif arr[:, i].var() < self.cfg.flat_variance:
                problems[ch] = "flat at %.0f counts" % arr[:, i].mean()
        return {
            "rate": (len(hist) - 1) / span if span > 0 else 0.0,
            "n": len(hist),
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
            "saturated": sat,
            "problems": problems,
        }

    def start_log(self, path, t0):
        """Write every sample from now on to `path` (CSV), times relative to monotonic `t0`."""
        f = open(path, "w", newline="")
        out = csv.writer(f)
        out.writerow(
            ["t", "teensy_micros"]
            + ["counts_" + c for c in self.channels]
            + (["N_" + c for c in self.channels] if self.calibration else [])
        )
        with self._lock:
            self._log, self._csv, self._t0 = f, out, t0

    def stop_log(self):
        with self._lock:
            f, self._log, self._csv = self._log, None, None
        if f is not None:
            f.close()

    def describe(self):
        """One line: where it's reading from and whether forces are calibrated."""
        cal = (
            "calibrated (%s)" % self.calibration.version
            if self.calibration
            else "UNCALIBRATED: raw counts only (tf baxter calibrate --config config/tactile.yaml)"
        )
        return "tactile %s, %s" % (self.port or "not found", cal)

    def diagnose(self):
        """Why there is no data yet, as advice."""
        if self.connected:
            if self.n_lines == 0:
                return (
                    "%s is open but silent: flash the firmware (scripts/flash_tactile.sh)"
                    % self.port
                )
            return (
                "%s sends %d line(s), none of them FSR samples: not the teensy_fsr firmware? "
                "(scripts/flash_tactile.sh)" % (self.port, self.n_lines)
            )
        if self.error:
            return self.error + (
                "  (permissions: is this user in the dialout group?)"
                if "ermission" in self.error
                else ""
            )
        others = sorted(glob.glob("/dev/ttyACM*"))
        return "no Teensy (USB %04x:xxxx) found%s; plug it in (a data cable, not charge-only)" % (
            TEENSY_VID,
            "; other serial devices: " + " ".join(others) if others else "",
        )

    def close(self):
        self._run = False
        self._thread.join(timeout=1.0)
        self.stop_log()


def report(tac, h):
    """Print a health() summary; True when nothing looks wrong."""
    print(
        "%d samples in the last %.1f s: %.0f Hz (firmware: %.0f Hz); %d bad line(s) since start"
        % (h["n"], h["n"] / max(h["rate"], 1e-9), h["rate"], tac.cfg.sample_rate_hz, tac.n_bad)
    )
    for i, ch in enumerate(tac.channels):
        print(
            "  %-4s counts mean %6.1f  std %5.1f  min %4.0f  max %4.0f  %s"
            % (ch, h["mean"][i], h["std"][i], h["min"][i], h["max"][i], h["problems"].get(ch, ""))
        )
    ok = h["rate"] > 0.8 * tac.cfg.sample_rate_hz and not any(
        "saturated" in p for p in h["problems"].values()
    )
    if any(p.startswith("flat") for p in h["problems"].values()):
        print(
            "  flat = no change at all: fine for an unloaded pad, but press it -- if it never moves, "
            "the pad or its wiring is dead"
        )
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", help="serial device (default: find the Teensy by its USB id)")
    ap.add_argument(
        "--check", type=float, metavar="S", help="read S seconds, report, exit 1 if unhealthy"
    )
    ap.add_argument(
        "--wait", type=float, metavar="S", help="give up if no data within S s (default: wait)"
    )
    ap.add_argument("--log", help="also write every sample to this CSV")
    args = ap.parse_args()

    tac = TactileStream(args.port)
    try:
        print("waiting for tactile data (Teensy, USB %04x:0483)..." % TEENSY_VID, flush=True)
        t, last_hint = time.monotonic(), None
        while not tac.wait(1.0):
            hint = tac.diagnose()
            if hint != last_hint and (time.monotonic() - t > 3 or not tac.connected):
                print("  " + hint, flush=True)
                last_hint = hint
            if args.wait is not None and time.monotonic() - t > args.wait:
                print("no tactile data after %.0f s." % args.wait)
                return 1
            if args.check and tac.connected and time.monotonic() - t > 10:
                if not tac.wait(10.0):  # plugged in, 20 s, still no FSR samples
                    print("no FSR samples from %s: %s" % (tac.port, tac.diagnose()))
                    return 1
        print(tac.describe(), flush=True)
        if args.log:
            tac.start_log(args.log, time.monotonic())
        if args.check:
            time.sleep(args.check)
            h = tac.health(args.check)
            if h is None:
                print("data stopped: %s" % tac.diagnose())
                return 1
            return 0 if report(tac, h) else 1
        names = tac.channels
        while True:
            counts, newtons, ts = tac.latest()
            if counts is None or time.monotonic() - ts > 0.5:
                sys.stdout.write("\r  no fresh data: %s\033[K" % tac.diagnose())
            else:
                cells = [
                    "%s %4.0f%s %-10s"
                    % (
                        n,
                        c,
                        " (%.1f N)" % f if newtons is not None else "",
                        "#" * int(10 * c / tac.cfg.adc_max_counts + 0.5),
                    )
                    for n, c, f in zip(names, counts, newtons if newtons is not None else counts)
                ]
                sys.stdout.write("\r  " + " | ".join(cells) + "\033[K")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        tac.close()


if __name__ == "__main__":
    raise SystemExit(main())
