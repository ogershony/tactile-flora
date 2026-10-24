#!/usr/bin/env python3
"""BaxterBridge -- Python-3 control layer for Baxter over rosbridge/roslibpy.

The foundation of the demo stack. Everything smart (skills, executor, LLM
planner, Dream-1 client) runs in Python 3 and talks to Baxter only through
this object -- never importing the Python-2 SDK. It wraps a roslibpy websocket
connection (rosbridge on :9090) into clean calls:

    bx = BaxterBridge(); bx.connect()
    bx.joint_angles('left')                      # dict name->rad (cached, ~100Hz)
    bx.move_to('left', target, max_speed=0.3)    # interpolated, speed-capped
    bx.gripper('left', 'open' | 'close')
    bx.gripper_state('left')                     # {position, gripping, force}
    bx.ik('left', pos=(x,y,z), quat=(x,y,z,w))   # -> joint dict or None
    bx.endpoint_pose('left')                     # current cartesian pose
    bx.camera_frame('left_hand_camera')          # -> HxWx3 uint8 (bgr) ndarray
    bx.set_face(img_bgr) / bx.set_face_text("hi")

Motion safety: move_to interpolates in joint space and caps joint speed
(default 0.3 rad/s), streaming POSITION_MODE setpoints at 50 Hz. Keep a hand
on the e-stop the first times you run real moves.
"""

import base64
import json
import math
import time

import numpy as np
import roslibpy
from PIL import Image, ImageDraw, ImageFont

JOINTS = {
    "left": ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"],
    "right": ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"],
}
POSITION_MODE = 1  # baxter_core_msgs/JointCommand.POSITION_MODE


class BaxterBridge(object):
    def __init__(self, host="localhost", port=9090):
        self.ros = roslibpy.Ros(host=host, port=port)
        self._joint_state = {}  # name -> position
        self._ee_state = {}  # side -> dict(position,gripping,force,id)
        self._endpoint = {}  # side -> dict(position,orientation)
        self._pubs = {}  # cached advertised Topics
        self._digital = {}  # digital_io name -> bool
        self._digital_subs = {}  # name -> Topic
        self.last_cmd = {}  # side -> {joint: rad} last joint command sent
        self.last_grip_cmd = {}  # side -> 0..100 last gripper command sent

    # --- lifecycle --------------------------------------------------------
    def connect(self, timeout=10):
        self.ros.run()
        t0 = time.time()
        while not self.ros.is_connected and time.time() - t0 < timeout:
            time.sleep(0.05)
        if not self.ros.is_connected:
            raise RuntimeError("rosbridge not reachable (is start-rosbridge.sh up?)")
        self._sub_joint_states()
        for side in ("left", "right"):
            self._sub_ee_state(side)
            self._sub_endpoint(side)
        # let the caches fill
        t0 = time.time()
        while not self._joint_state and time.time() - t0 < 5:
            time.sleep(0.05)
        return self

    def close(self):
        self.ros.terminate()

    # --- subscriptions (cache latest) -------------------------------------
    def _sub_joint_states(self):
        t = roslibpy.Topic(
            self.ros,
            "/robot/joint_states",
            "sensor_msgs/JointState",
            throttle_rate=10,
            queue_length=1,
        )

        def cb(msg):
            for n, p in zip(msg["name"], msg["position"]):
                self._joint_state[n] = p

        t.subscribe(cb)

    def _sub_ee_state(self, side):
        t = roslibpy.Topic(
            self.ros,
            "/robot/end_effector/%s_gripper/state" % side,
            "baxter_core_msgs/EndEffectorState",
            throttle_rate=50,
            queue_length=1,
        )

        def cb(msg):
            self._ee_state[side] = {
                "position": msg.get("position", 0.0),
                "gripping": bool(msg.get("gripping", 0)),
                "force": msg.get("force", 0.0),
                "id": msg.get("id", 0),
            }

        t.subscribe(cb)

    def _sub_endpoint(self, side):
        t = roslibpy.Topic(
            self.ros,
            "/robot/limb/%s/endpoint_state" % side,
            "baxter_core_msgs/EndpointState",
            throttle_rate=50,
            queue_length=1,
        )

        def cb(msg):
            self._endpoint[side] = msg["pose"]

        t.subscribe(cb)

    # --- reads ------------------------------------------------------------
    def joint_angles(self, side):
        return {n: self._joint_state[n] for n in JOINTS[side] if n in self._joint_state}

    def gripper_state(self, side):
        return dict(self._ee_state.get(side, {}))

    def endpoint_pose(self, side):
        return dict(self._endpoint.get(side, {}))

    def digital(self, name):
        """Latest bool state of a DigitalIO (e.g. 'left_upper_button',
        'left_lower_cuff'); subscribes lazily on first use."""
        if name not in self._digital_subs:
            t = roslibpy.Topic(
                self.ros,
                "/robot/digital_io/%s/state" % name,
                "baxter_core_msgs/DigitalIOState",
                throttle_rate=20,
                queue_length=1,
            )
            t.subscribe(lambda m, n=name: self._digital.__setitem__(n, bool(m.get("state", 0))))
            self._digital_subs[name] = t
            time.sleep(0.3)
        return self._digital.get(name, False)

    # --- motion -----------------------------------------------------------
    def _joint_cmd_pub(self, side):
        name = "/robot/limb/%s/joint_command" % side
        if name not in self._pubs:
            p = roslibpy.Topic(self.ros, name, "baxter_core_msgs/JointCommand")
            p.advertise()
            self._pubs[name] = p
            time.sleep(0.2)
        return self._pubs[name]

    def set_speed_ratio(self, side, ratio):
        """POSITION_MODE's own joint speed cap, 0..1 of max (Baxter default 0.3)."""
        name = "/robot/limb/%s/set_speed_ratio" % side
        if name not in self._pubs:
            p = roslibpy.Topic(self.ros, name, "std_msgs/Float64")
            p.advertise()
            self._pubs[name] = p
            time.sleep(0.2)
        # a new topic's first messages are dropped while ROS connects it to Baxter (a
        # 3-message burst never took effect, 2026-10-22): keep sending for ~1.5 s
        for _ in range(15):
            self._pubs[name].publish(roslibpy.Message({"data": float(ratio)}))
            time.sleep(0.1)

    def set_joint_positions(self, side, angles):
        """One raw POSITION_MODE setpoint (for streaming control loops)."""
        names = list(angles.keys())
        self.last_cmd[side] = dict(angles)
        self._joint_cmd_pub(side).publish(
            roslibpy.Message(
                {
                    "mode": POSITION_MODE,
                    "command": [angles[n] for n in names],
                    "names": names,
                }
            )
        )

    def move_to(
        self, side, target, max_speed=0.3, rate=50.0, tolerance=0.02, settle=0.4, timeout=15.0
    ):
        """Interpolate current->target in joint space, speed-capped (rad/s)."""
        names = JOINTS[side]
        start = self.joint_angles(side)
        target = {n: target[n] for n in names if n in target}
        max_delta = max(abs(target[n] - start[n]) for n in target) if target else 0.0
        duration = max(max_delta / float(max_speed), 0.3)
        steps = max(int(duration * rate), 1)
        pub = self._joint_cmd_pub(side)
        dt = 1.0 / rate
        t0 = time.time()
        for i in range(1, steps + 1):
            a = i / float(steps)
            cmd = {n: start[n] + a * (target[n] - start[n]) for n in target}
            self.last_cmd[side] = cmd
            pub.publish(
                roslibpy.Message(
                    {
                        "mode": POSITION_MODE,
                        "command": [cmd[n] for n in cmd],
                        "names": list(cmd.keys()),
                    }
                )
            )
            time.sleep(dt)
            if time.time() - t0 > timeout:
                break
        # hold target to settle
        self.last_cmd[side] = dict(target)
        t1 = time.time()
        while time.time() - t1 < settle:
            pub.publish(
                roslibpy.Message(
                    {
                        "mode": POSITION_MODE,
                        "command": [target[n] for n in target],
                        "names": list(target.keys()),
                    }
                )
            )
            time.sleep(dt)
        now = self.joint_angles(side)
        return max(abs(now[n] - target[n]) for n in target) if target else 0.0

    # --- gripper ----------------------------------------------------------
    def gripper(self, side, action, block=True):
        """action: 'open' | 'close' | float position 0..100."""
        pos = {"open": 100.0, "close": 0.0}.get(action, action)
        self.last_grip_cmd[side] = float(pos)
        gid = self._ee_state.get(side, {}).get("id", 65538)
        name = "/robot/end_effector/%s_gripper/command" % side
        if name not in self._pubs:
            p = roslibpy.Topic(self.ros, name, "baxter_core_msgs/EndEffectorCommand")
            p.advertise()
            self._pubs[name] = p
            time.sleep(0.2)
        self._pubs[name].publish(
            roslibpy.Message(
                {
                    "id": gid,
                    "command": "go",
                    "args": json.dumps({"position": float(pos)}),
                    "sender": "baxter_bridge",
                    "sequence": 0,
                }
            )
        )
        if block:
            time.sleep(1.0)
        return self.gripper_state(side)

    # --- IK ---------------------------------------------------------------
    def ik(self, side, pos, quat):
        srv = roslibpy.Service(
            self.ros,
            "/ExternalTools/%s/PositionKinematicsNode/IKService" % side,
            "baxter_core_msgs/SolvePositionIK",
        )
        req = roslibpy.ServiceRequest(
            {
                "pose_stamp": [
                    {
                        "header": {"frame_id": "base", "stamp": {"secs": 0, "nsecs": 0}},
                        "pose": {
                            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                            "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                        },
                    }
                ],
                "seed_angles": [],
            }
        )
        res = srv.call(req, timeout=5)
        if not res.get("isValid") or not res["isValid"][0]:
            return None
        js = res["joints"][0]
        return dict(zip(js["name"], js["position"]))

    # --- cameras ----------------------------------------------------------
    def camera_frame(self, cam, timeout=5.0):
        """Return the latest frame as an HxWx3 BGR uint8 ndarray.

        Baxter wrist cams publish bgra8 (4 channels); we drop alpha and keep
        BGR, which is what cv2 / color-threshold perception expect.
        """
        box = {"img": None}
        t = roslibpy.Topic(self.ros, "/cameras/%s/image" % cam, "sensor_msgs/Image", queue_length=1)

        def cb(msg):
            if box["img"] is None:
                raw = base64.b64decode(msg["data"])
                h, w = msg["height"], msg["width"]
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, -1)
                box["img"] = arr[:, :, :3]  # bgra8 -> bgr (drop alpha)

        t.subscribe(cb)
        t0 = time.time()
        while box["img"] is None and time.time() - t0 < timeout:
            time.sleep(0.02)
        t.unsubscribe()
        return box["img"]

    def stream_camera(self, cam, throttle_ms=200):
        """Keep subscribed to a camera; returns a CameraStream whose latest()
        is (bgr ndarray, monotonic arrival time) or (None, None). For control
        loops that need a frame every tick (camera_frame resubscribes per call)."""
        return CameraStream(self.ros, cam, throttle_ms)

    # --- face -------------------------------------------------------------
    def set_face(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self._publish_xdisplay(img_bgr.tobytes(), w, h)

    def set_face_text(self, text, bg=(20, 20, 30), fg=(80, 220, 120)):
        w, h = 1024, 600
        img = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except Exception:
            font = ImageFont.load_default()
        bb = d.textbbox((0, 0), text, font=font)
        d.text(
            ((w - (bb[2] - bb[0])) / 2, (h - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=fg, font=font
        )
        b, g, r = img.split()[2], img.split()[1], img.split()[0]
        bgr = Image.merge("RGB", (b, g, r))
        self._publish_xdisplay(bgr.tobytes(), w, h)

    def _publish_xdisplay(self, data_bytes, w, h):
        name = "/robot/xdisplay"
        if name not in self._pubs:
            p = roslibpy.Topic(self.ros, name, "sensor_msgs/Image")
            p.advertise()
            self._pubs[name] = p
            time.sleep(0.3)
        self._pubs[name].publish(
            roslibpy.Message(
                {
                    "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
                    "height": h,
                    "width": w,
                    "encoding": "bgr8",
                    "is_bigendian": 0,
                    "step": w * 3,
                    "data": base64.b64encode(data_bytes).decode("ascii"),
                }
            )
        )


class CameraStream(object):
    """Persistent /cameras/<cam>/image subscription caching the newest frame."""

    def __init__(self, ros, cam, throttle_ms=200):
        self._img, self._t = None, None
        self._topic = roslibpy.Topic(
            ros,
            "/cameras/%s/image" % cam,
            "sensor_msgs/Image",
            throttle_rate=throttle_ms,
            queue_length=1,
        )
        self._topic.subscribe(self._cb)

    def _cb(self, msg):
        raw = base64.b64decode(msg["data"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg["height"], msg["width"], -1)
        self._img, self._t = arr[:, :, :3], time.monotonic()  # bgra8 -> bgr

    def latest(self):
        return self._img, self._t

    def close(self):
        self._topic.unsubscribe()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="BaxterBridge smoke test")
    ap.add_argument("--move", action="store_true", help="also run a small LEFT w2 move (MOTION!)")
    args = ap.parse_args()

    bx = BaxterBridge().connect()
    print("connected.")
    print("left angles:", {k: round(v, 3) for k, v in bx.joint_angles("left").items()})
    print("left gripper:", bx.gripper_state("left"))
    ep = bx.endpoint_pose("left")
    if ep:
        p = ep["position"]
        print("left endpoint xyz:", round(p["x"], 3), round(p["y"], 3), round(p["z"], 3))
        sol = bx.ik(
            "left",
            (p["x"], p["y"], p["z"] + 0.05),
            (
                ep["orientation"]["x"],
                ep["orientation"]["y"],
                ep["orientation"]["z"],
                ep["orientation"]["w"],
            ),
        )
        print("ik (+5cm):", "valid" if sol else "FAILED")
    img = bx.camera_frame("left_hand_camera")
    print("left cam frame:", None if img is None else img.shape)
    bx.set_face_text("BRIDGE")
    print("face set.")

    if args.move:
        print("MOTION: nudging left_w2 +0.2 and back ...")
        s = bx.joint_angles("left")
        t = dict(s)
        t["left_w2"] += 0.2
        bx.move_to("left", t, max_speed=0.2)
        err_there = abs(bx.joint_angles("left")["left_w2"] - t["left_w2"])
        bx.move_to("left", s, max_speed=0.2)
        err_back = abs(bx.joint_angles("left")["left_w2"] - s["left_w2"])
        print("move err there=%.4f back=%.4f" % (err_there, err_back))
    bx.close()
    print("done.")
