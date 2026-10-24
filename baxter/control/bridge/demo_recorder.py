#!/usr/bin/env python3
"""Record Baxter demonstrations as LeRobot v2.1 sessions for a Dream fine-tune.

One episode per session folder (dream's upload takes a folder of session folders):

    <root>/<session>/meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}
    <root>/<session>/data/chunk-000/episode_000000.parquet
    <root>/<session>/videos/chunk-000/observation.images.<role>/episode_000000.mp4   (H.264)

Vectors follow EMBODIMENT: robots.adapter.zero_shot_embodiment(7), the layout dream's
setup wizard registers for an arm of your own, plus head_camera. Per side: the 7 joints
(rad), the tool pose (eef_xyz m, eef_rot6) and grip (0 closed .. 1 open). State is
measured; action is what was commanded (BaxterBridge.last_cmd / last_grip_cmd). Tool
poses come from dream_ik's Baxter, table-shifted as dream_control runs it, through the
client's own FK + mat_to_rot6d, so training poses are the poses the live client sends.
Frames are resized as dream_control sends them (dream_control.IMAGE_SIZE).

    rec = EpisodeRecorder(bx, sources, table_z=0.003)
    rec.start("pick up the red cube"); ...motion...; rec.stop()
    rec.write("demos/ep0001")
    dream.DreamClient().validate_dataset("demos", EMBODIMENT)
"""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baxter_bridge import JOINTS  # noqa: E402

FPS = 15  # one of dream's usable rates (30, 15, 10, 5); 15 for ACT-style policies
GRIP_LEAD_S = 0.5  # hand-guided: the gripper action is its state this far ahead
SIDES = ("left", "right")
CAMERAS = ("scene", "wrist_l", "wrist_r")


def embodiment():
    from robots.adapter import zero_shot_embodiment

    return dict(zero_shot_embodiment(7), head_camera="scene")


EMBODIMENT = None  # filled on first use (imports dream's robots package)


def _names(spans):
    """Per-dimension names, in vector order, for a {part: [start, end)} layout."""
    out = []
    for part, (a, b) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        out += [part] if b - a == 1 else ["%s_%d" % (part, i) for i in range(b - a)]
    return out


def _assemble(spans, parts):
    width = max(b for _, b in spans.values())
    v = np.zeros(width, np.float32)
    for part, (a, b) in spans.items():
        v[a:b] = parts[part]
    return v


def _stats(values):
    values = np.asarray(values, np.float64).reshape(len(values), -1)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


class EpisodeRecorder(object):
    def __init__(self, bx, sources, table_z=0.003, image_size=(640, 360), action_from="command"):
        """action_from: "command" (what the bridge last commanded: scripted demos) or
        "next_state" (hand-guided demos, nothing is commanded: the next frame's joints,
        and the gripper GRIP_LEAD_S ahead so a close the person starts is the action)."""
        self.action_from = action_from
        global EMBODIMENT
        if EMBODIMENT is None:
            EMBODIMENT = embodiment()
        self.bx, self.sources, self.image_size = bx, sources, image_size
        import dream_control

        dream_control.enable_baxter_ik(table_z)  # the frames the live client uses
        from dream_ik import decode
        from dream_ik.solver import ChunkIK
        from dream_ik.specs import SERVE

        self._fk = ChunkIK("baxter", SERVE.tool, FPS).fk
        self._rot6 = decode.mat_to_rot6d
        self._thread, self._run = None, False

    # --- sampling ---------------------------------------------------------
    def _sample(self):
        q, cmd, grip, grip_cmd = {}, {}, {}, {}
        for side in SIDES:
            m = self.bx.joint_angles(side)
            q[side] = np.array([m[n] for n in JOINTS[side]])
            c = self.bx.last_cmd.get(side) or m
            cmd[side] = np.array([c.get(n, m[n]) for n in JOINTS[side]])
            pos = self.bx.gripper_state(side).get("position")
            grip[side] = 1.0 if pos is None else float(np.clip(pos / 100.0, 0, 1))
            g = self.bx.last_grip_cmd.get(side)
            grip_cmd[side] = grip[side] if g is None else float(np.clip(g / 100.0, 0, 1))
        frames = {}
        for role in CAMERAS:
            img, _ = self.sources[role].latest()
            if img is None:
                raise RuntimeError("no frame from camera '%s'" % role)
            if (img.shape[1], img.shape[0]) != self.image_size:
                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
            frames[role] = img
        return q, cmd, grip, grip_cmd, frames

    def _loop(self):
        period, t0 = 1.0 / FPS, time.monotonic()
        k = 0
        while self._run:
            try:
                self.ticks.append(self._sample())
            except Exception as e:  # recorded as the episode's error; motion goes on
                self.error = "%s: %s" % (type(e).__name__, e)
                self._run = False
                return
            k += 1
            time.sleep(max(0.0, t0 + k * period - time.monotonic()))

    def start(self, task):
        self.task, self.ticks, self.error = task, [], None
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return len(self.ticks)

    # --- writing ----------------------------------------------------------
    def _tool(self, q, side):
        T = np.asarray(self._fk(q, side))
        return T[:3, 3], np.asarray(self._rot6(T[:3, :3]))

    def write(self, folder):
        """Write the recorded episode as one LeRobot v2.1 session at `folder`."""
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.error:
            raise RuntimeError("recording failed: " + self.error)
        n = len(self.ticks)
        st_spans, ac_spans = EMBODIMENT["state"], EMBODIMENT["action"]
        ticks = self.ticks
        if self.action_from == "next_state":
            lead = int(round(GRIP_LEAD_S * FPS))
            ticks = [
                (t[0], ticks[min(i + 1, n - 1)][0], t[2], ticks[min(i + lead, n - 1)][2], t[4])
                for i, t in enumerate(ticks)
            ]
        states, actions = [], []
        for q, cmd, grip, grip_cmd, _ in ticks:
            sp, ap = {}, {}
            for side in SIDES:
                xyz, r6 = self._tool(q[side], side)
                sp.update(
                    {
                        "%s_arm" % side: q[side],
                        "%s_eef_xyz" % side: xyz,
                        "%s_eef_rot6" % side: r6,
                        "%s_grip" % side: grip[side],
                    }
                )
                xyz, r6 = self._tool(cmd[side], side)
                ap.update(
                    {
                        "%s_arm" % side: cmd[side],
                        "%s_eef_xyz" % side: xyz,
                        "%s_eef_rot6" % side: r6,
                        "%s_grip" % side: grip_cmd[side],
                    }
                )
            states.append(_assemble(st_spans, sp))
            actions.append(_assemble(ac_spans, ap))
        states, actions = np.array(states), np.array(actions)
        index = np.arange(n)
        columns = {
            "observation.state": states,
            "action": actions,
            "timestamp": (index / FPS).astype(np.float32),
            "frame_index": index,
            "episode_index": np.zeros(n, np.int64),
            "index": index,
            "task_index": np.zeros(n, np.int64),
        }
        data = os.path.join(folder, "data", "chunk-000")
        os.makedirs(data, exist_ok=True)
        pq.write_table(
            pa.table({k: pa.array(list(v) if v.ndim > 1 else v) for k, v in columns.items()}),
            os.path.join(data, "episode_000000.parquet"),
        )

        features, stats = {}, {}
        for k, v in columns.items():
            names = (
                _names(st_spans)
                if k == "observation.state"
                else _names(ac_spans)
                if k == "action"
                else None
            )
            features[k] = {
                "dtype": str(v.dtype),
                "shape": [v.shape[1] if v.ndim > 1 else 1],
                "names": names,
            }
            stats[k] = _stats(v)
        w, h = self.image_size
        for role in CAMERAS:
            key = EMBODIMENT["video"][role]  # observation.images.<role>
            vdir = os.path.join(folder, "videos", "chunk-000", key)
            os.makedirs(vdir, exist_ok=True)
            frames = [t[4][role] for t in self.ticks]
            with av.open(
                os.path.join(vdir, "episode_000000.mp4"), "w", options={"movflags": "faststart"}
            ) as out:
                stream = out.add_stream("libx264", rate=FPS)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = {"crf": "23", "preset": "veryfast", "bf": "0", "g": str(2 * FPS)}
                for img in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            features[key] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                    "video.fps": FPS,
                },
            }
            px = np.concatenate(
                [f[::8, ::8, ::-1].reshape(-1, 3) / 255.0 for f in frames[:: max(1, n // 50)]]
            )
            stats[key] = {
                k: v if k == "count" else [[[x]] for x in v] for k, v in _stats(px).items()
            }

        meta = os.path.join(folder, "meta")
        os.makedirs(meta, exist_ok=True)
        chunk = "chunk-{episode_chunk:03d}"
        with open(os.path.join(meta, "info.json"), "w") as f:
            json.dump(
                {
                    "codebase_version": "v2.1",
                    "robot_type": "baxter",
                    "total_episodes": 1,
                    "total_frames": n,
                    "total_tasks": 1,
                    "total_videos": len(CAMERAS),
                    "total_chunks": 1,
                    "chunks_size": 1000,
                    "fps": FPS,
                    "splits": {"train": "0:1"},
                    "data_path": "data/%s/episode_{episode_index:06d}.parquet" % chunk,
                    "video_path": "videos/%s/{video_key}/episode_{episode_index:06d}.mp4" % chunk,
                    "features": features,
                },
                f,
                indent=2,
            )
        for name, record in (
            ("tasks", {"task_index": 0, "task": self.task}),
            ("episodes", {"episode_index": 0, "tasks": [self.task], "length": n}),
            ("episodes_stats", {"episode_index": 0, "stats": stats}),
        ):
            with open(os.path.join(meta, name + ".jsonl"), "w") as f:
                f.write(json.dumps(record) + "\n")
        return n
