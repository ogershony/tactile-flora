"""Capturing one Baxter episode directory (videos, parquet, meta.json).

Owns: ``Recorder`` (subscriptions, the pre-episode gate, streaming MP4 writers, parquet and
meta.json) and ``run_episode`` (start, wait for the policy to finish, stop).
Must never: start an episode while any channel is flat or saturated, the gripper is not open or
health is stale; resample force (to_lerobot does); import rospy.

Episode directory ``<out_root>/episode_<idx>/``: ``scene.mp4``, ``wrist_left.mp4``,
``wrist_right.mp4``, ``state.parquet`` (t, L_position, L_command, R_position, R_command),
``force.parquet`` (t, L_a, L_b, R_a, R_b newtons, native 200 Hz, recorder-clock t) and
``meta.json`` (the fields of meta/episodes.jsonl plus ``baseline`` and ``n_frames``).

Policy status messages (``cfg.topics.policy_status``) are merged key by key; recognised keys:
object, firmness, grasp_height, lateral_offset, gripper_force, occlusion, task, and
``state == "done"`` to end the episode. Gripper messages carry ``position`` (closure, 0 open ..
1 closed) and ``command``. Videos are written constant-frame-rate at ``cfg.camera_fps`` in arrival
order; ``video_t0_s`` is the stamp of the first scene frame. Unverified: Baxter's native gripper
units (the ROS bridge normalizes them), wrist-camera start skew (ignored).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

from tactile_flora.baxter.ros.transport import Msg, Transport
from tactile_flora.config import BaxterConfig

ROLES = ("scene", "wrist_left", "wrist_right")
CODEC = "mpeg4"
POLICY_KEYS = ("grasp_height", "lateral_offset", "gripper_force")


class _VideoWriter:
    """Streaming constant-frame-rate MP4 writer (PyAV); opened on the first frame."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path, self.fps, self.n = path, fps, 0
        self._container: Any = None
        self._stream: Any = None

    def write(self, image: np.ndarray) -> None:
        if self._container is None:
            h, w = image.shape[:2]
            self._container = av.open(str(self.path), "w")
            self._stream = self._container.add_stream(CODEC, rate=int(round(self.fps)))
            self._stream.width, self._stream.height = w - w % 2, h - h % 2
            self._stream.pix_fmt = "yuv420p"
        img = np.ascontiguousarray(image[: self._stream.height, : self._stream.width, :3])
        for pkt in self._stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            self._container.mux(pkt)
        self.n += 1

    def close(self) -> None:
        if self._container is not None:
            for pkt in self._stream.encode():
                self._container.mux(pkt)
            self._container.close()
            self._container = None


class Recorder:
    """Records one episode at a time from the transport into ``out_root``."""

    def __init__(
        self,
        transport: Transport,
        cfg: BaxterConfig,
        out_root: Path,
        calibration_version: str = "unknown",
        today: Callable[[], str] = lambda: date_cls.today().isoformat(),
    ) -> None:
        self.transport, self.cfg = transport, cfg
        self.out_root = Path(out_root)
        self.calibration_version = calibration_version
        self._today = today
        self.recording = False
        self.policy_done = False
        self._health: tuple[Msg, float] | None = None
        self._closure: dict[str, float] = {}
        self._status: dict[str, Any] = {}
        self._writers: dict[str, _VideoWriter] = {}
        self._video_t0: float | None = None
        self._state_rows: list[list[float]] = []
        self._force_rows: list[list[float]] = []
        self._cur: dict[str, tuple[float, float]] = {"L": (0.0, 0.0), "R": (0.0, 0.0)}
        self._dir: Path | None = None
        self._baseline: list[float] = []
        self._notes = ""
        t = cfg.topics
        for role in ROLES:
            transport.subscribe(t.cameras[role], self._camera_cb(role))
        for arm in ("L", "R"):
            transport.subscribe(t.gripper_state[arm], self._gripper_cb(arm))
        transport.subscribe(t.force, self._on_force)
        transport.subscribe(t.health, self._on_health)
        transport.subscribe(t.policy_status, self._on_status)

    # --- subscriptions -------------------------------------------------------------------
    def _camera_cb(self, role: str) -> Callable[[Msg, float], None]:
        def _cb(msg: Msg, stamp: float) -> None:
            w = self._writers.get(role)
            if not self.recording or w is None:
                return
            if role == "scene" and self._video_t0 is None:
                self._video_t0 = stamp
            w.write(msg["image"])

        return _cb

    def _gripper_cb(self, arm: str) -> Callable[[Msg, float], None]:
        def _cb(msg: Msg, stamp: float) -> None:
            self._closure[arm] = float(msg["position"])
            self._cur[arm] = (float(msg["position"]), float(msg.get("command", 0.0)))
            if self.recording:
                self._state_rows.append([stamp, *self._cur["L"], *self._cur["R"]])

        return _cb

    def _on_force(self, msg: Msg, stamp: float) -> None:
        if self.recording:
            self._force_rows.append([stamp, *[float(v) for v in msg["data"]]])

    def _on_health(self, msg: Msg, stamp: float) -> None:
        self._health = (msg, self.transport.now())

    def _on_status(self, msg: Msg, stamp: float) -> None:
        self._status.update(msg)
        if msg.get("state") == "done":
            self.policy_done = True

    # --- gate ----------------------------------------------------------------------------
    def check_ready(self) -> None:
        """Raise RuntimeError unless health is fresh and clean and both grippers are open."""
        if self._health is None:
            raise RuntimeError("no /tactile/health received; is the FSR node running?")
        msg, received = self._health
        if self.transport.now() - received > self.cfg.health_max_age_s:
            raise RuntimeError("/tactile/health is stale")
        names = self.cfg.channels
        flat = [
            n for n, v in zip(names, msg["variance"], strict=True) if v < self.cfg.flat_variance
        ]
        sat = [
            n
            for n, v in zip(names, msg["saturation"], strict=True)
            if v > self.cfg.saturation_frac_max
        ]
        if flat:
            raise RuntimeError(f"refusing to record: flat channel(s) {flat}")
        if sat:
            raise RuntimeError(f"refusing to record: saturated channel(s) {sat}")
        for arm in ("L", "R"):
            if arm not in self._closure:
                raise RuntimeError(f"no gripper state for arm {arm}")
            if self._closure[arm] > self.cfg.gripper_open_max:
                raise RuntimeError(f"refusing to record: {arm} gripper is not open")

    # --- episode lifecycle -----------------------------------------------------------------
    def _next_dir(self) -> Path:
        self.out_root.mkdir(parents=True, exist_ok=True)
        idx = 0
        while (self.out_root / f"episode_{idx:06d}").exists():
            idx += 1
        return self.out_root / f"episode_{idx:06d}"

    def start_episode(self, operator_notes: str = "") -> Path:
        """Gate, zero the sensors (``Z`` via the node), then start recording."""
        if self.recording:
            raise RuntimeError("an episode is already recording")
        self.check_ready()
        reply = self.transport.call(self.cfg.topics.zero, {})
        self._baseline = [float(v) for v in reply.get("baseline", [])]
        self._dir = self._next_dir()
        self._dir.mkdir(parents=True)
        self._writers = {
            r: _VideoWriter(self._dir / f"{r}.mp4", self.cfg.camera_fps) for r in ROLES
        }
        self._video_t0 = None
        self._state_rows, self._force_rows = [], []
        self._notes = operator_notes
        self.policy_done = False
        self.recording = True
        return self._dir

    def stop_episode(self) -> Path:
        """Stop, flush videos, write parquet and meta.json; returns the episode directory."""
        if not self.recording or self._dir is None:
            raise RuntimeError("no episode is recording")
        self.recording = False
        for w in self._writers.values():
            w.close()
        d = self._dir
        ch = self.cfg.channels
        pd.DataFrame(
            self._state_rows, columns=["t", "L_position", "L_command", "R_position", "R_command"]
        ).to_parquet(d / "state.parquet")
        pd.DataFrame(self._force_rows, columns=["t", *ch]).to_parquet(d / "force.parquet")
        s = self._status
        obj = str(s.get("object", ""))
        meta: dict[str, Any] = {
            "episode_index": int(d.name.removeprefix("episode_")),
            "object": obj,
            "firmness": str(s.get("firmness", "")),
            "policy": {k: s.get(k) for k in POLICY_KEYS},
            "occlusion": bool(s.get("occlusion", False)),
            "operator_notes": self._notes,
            "recording_day": self._today(),
            "calibration_version": self.calibration_version,
            "video_t0_s": self._video_t0 if self._video_t0 is not None else 0.0,
            "n_frames": {r: w.n for r, w in self._writers.items()},
            "baseline": self._baseline,
            "sensor_confirmed": self.cfg.sensor_confirmed,
        }
        if s.get("task"):
            meta["task"] = str(s["task"])
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        self._status = {}
        self._dir = None
        return d


def run_episode(
    recorder: Recorder, operator_notes: str = "", timeout_s: float = 120.0, poll_s: float = 0.05
) -> Path:
    """Start an episode, wait until the policy reports done (or timeout), stop and return it."""
    recorder.start_episode(operator_notes)
    t_end = recorder.transport.now() + timeout_s
    while not recorder.policy_done and recorder.transport.now() < t_end:
        recorder.transport.sleep(poll_s)
    return recorder.stop_episode()
