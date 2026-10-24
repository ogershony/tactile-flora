#!/usr/bin/env python3
"""Record policy rollouts (il_control --live) with their tactile data, one folder each, for
reward evaluation / shaping; bridge/publish_rollouts.py turns a folder of them into a LeRobot
dataset.

    rollouts/<YYYYMMDD>/epNNNN/
        data.parquet          one row per frame at 15 fps (the demos' rate), see FRAME below
        head.mp4, wrist.mp4   H.264, 640x360 RGB: the frames as the policy is fed them
        tactile_200hz.csv     every FSR sample (bridge/tactile.py), t on the same clock as data
        meta.json             checkpoint, settings, stop reason, auto + operator outcome

    rec = RolloutRecorder(); path = rec.start(tac, meta)
    rec.add(t, frame, head_rgb, wrist_rgb)        # whenever rec.due(t)
    rec.finish(stop_reason=..., auto_success=..., label=...)
"""

import json
import os
import queue
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 15
# per-frame columns: observation.state / action / policy.target as the demos' 8 (left s0..w2 rad,
# gripper 0 closed..1 open); action = what was SENT (speed-capped joints, gripper 0/1 command),
# policy.target = the plan's row at that moment (NaN before the first chunk); tactile counts /
# newtons (NaN when stale or uncalibrated); fingertip = measured, m in Baxter base frame
FRAME = (
    "observation.state",
    "action",
    "policy.target",
    "observation.tactile",
    "observation.tactile_newtons",
    "tactile_age",
    "fingertip",
    "chunk",
    "gripper_closed",
)


class _Video(object):
    """H.264 writer fed from a queue on its own thread, so encoding never stalls the control loop."""

    def __init__(self, path, size=(640, 360)):
        import av

        self._c = av.open(path, "w", options={"movflags": "faststart"})
        self._s = self._c.add_stream("libx264", rate=FPS)
        self._s.width, self._s.height, self._s.pix_fmt = size[0], size[1], "yuv420p"
        self._s.options = {"crf": "23", "preset": "veryfast", "bf": "0", "g": str(2 * FPS)}
        self._q, self.n = queue.Queue(), 0
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        import av

        while True:
            img = self._q.get()
            if img is None:
                break
            for p in self._s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                self._c.mux(p)
        for p in self._s.encode():
            self._c.mux(p)
        self._c.close()

    def add(self, img_rgb):
        self._q.put(img_rgb)
        self.n += 1

    def close(self):
        self._q.put(None)
        self._t.join()


class RolloutRecorder(object):
    def __init__(self, root=None):
        self.root = root or os.path.join(ROOT, "rollouts", time.strftime("%Y%m%d"))
        self.path = None

    def start(self, tac, meta, t0):
        """New episode folder; frames and the tactile log are timed from monotonic t0."""
        os.makedirs(self.root, exist_ok=True)
        n = len([d for d in os.listdir(self.root) if d.startswith("ep")])
        self.path = os.path.join(self.root, "ep%04d" % (n + 1))
        os.makedirs(self.path)
        self.meta = dict(
            meta,
            started=time.strftime("%Y-%m-%dT%H:%M:%S"),
            fps=FPS,
            tactile=None
            if tac is None
            else {
                "port": tac.port,
                "channels": list(tac.channels),
                "calibrated": tac.calibration is not None,
            },
        )
        self.t0, self.rows, self._next = t0, [], t0
        self.videos = {
            "head": _Video(os.path.join(self.path, "head.mp4")),
            "wrist": _Video(os.path.join(self.path, "wrist.mp4")),
        }
        self.tac = tac
        if tac is not None:
            tac.start_log(os.path.join(self.path, "tactile_200hz.csv"), t0)
        return self.path

    def due(self, t):
        return self.path is not None and t >= self._next

    def add(self, t, frame, head_rgb, wrist_rgb):
        """One 15 fps frame: `frame` maps FRAME names to values, images are 640x360 RGB."""
        self._next = max(self._next + 1.0 / FPS, t)  # skip, never burst, after a stall
        self.rows.append(dict(frame, t=t - self.t0))
        self.videos["head"].add(head_rgb)
        self.videos["wrist"].add(wrist_rgb)

    def finish(self, **outcome):
        """Close the videos, write data + meta. outcome: stop_reason, auto_success, label, ..."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.path is None:
            return None
        if self.tac is not None:
            self.tac.stop_log()
        for v in self.videos.values():
            v.close()
        n = len(self.rows)
        cols = {
            "timestamp": np.arange(n, dtype=np.float32) / FPS,
            "frame_index": np.arange(n),
            "t": np.array([r["t"] for r in self.rows], np.float64),
        }
        for k in FRAME:
            v = np.array([np.asarray(r[k], np.float32) for r in self.rows], np.float32)
            cols[k] = pa.array(list(v)) if v.ndim > 1 else v
        pq.write_table(pa.table(cols), os.path.join(self.path, "data.parquet"))
        self.meta.update(outcome, frames=n, duration_s=round(self.rows[-1]["t"], 3) if n else 0.0)
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        path, self.path = self.path, None
        return path
