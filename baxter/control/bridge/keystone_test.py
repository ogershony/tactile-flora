#!/usr/bin/env python3
"""Keystone test: prove the Python-3 brain can talk to Baxter over rosbridge.

This is the architecture bet for the whole project. Everything "smart" (LLM
planner, Dream-1 client, perception) runs in Python 3 and never imports
baxter_interface (Python 2). It reaches Baxter as JSON over a local rosbridge
websocket via roslibpy, which encodes even Baxter's custom messages as plain
dicts -- no compiling baxter_core_msgs for py3.

Two proofs here, both NO MOTION:
  1. subscribe to /robot/joint_states and measure the rate we can pull through
     the bridge  -> tells us whether a 20 Hz control loop is viable
  2. publish an image to /robot/xdisplay (the face) -> proves the publish path,
     including a Baxter-relevant message, end to end

Run on the host:  .venv/bin/python bridge/keystone_test.py
"""

import argparse
import base64
import struct
import time

import roslibpy
from PIL import Image, ImageDraw, ImageFont


def measure_joint_states(ros, seconds):
    """Subscribe to joint_states, count messages, return (count, hz, sample)."""
    got = {"n": 0, "last": None}
    topic = roslibpy.Topic(
        ros,
        "/robot/joint_states",
        "sensor_msgs/JointState",
        queue_length=1,
        throttle_rate=0,
    )

    def cb(msg):
        got["n"] += 1
        got["last"] = msg

    topic.subscribe(cb)
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.05)
    topic.unsubscribe()
    dt = time.time() - t0
    return got["n"], got["n"] / dt, got["last"]


def face_image_bytes(text):
    """1024x600 bgr8 raw bytes for /robot/xdisplay, with centered text."""
    w, h = 1024, 600
    img = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (h - th) / 2 - bbox[1]), text, fill=(80, 220, 120), font=font)
    # Baxter's xdisplay takes bgr8; PIL gives rgb, so swap channels.
    r, g, b = img.split()
    bgr = Image.merge("RGB", (b, g, r))
    return bgr.tobytes(), w, h


def publish_face(ros, text):
    data, w, h = face_image_bytes(text)
    pub = roslibpy.Topic(ros, "/robot/xdisplay", "sensor_msgs/Image")
    pub.advertise()
    time.sleep(0.5)
    msg = {
        "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
        "height": h,
        "width": w,
        "encoding": "bgr8",
        "is_bigendian": 0,
        "step": w * 3,
        # rosbridge encodes/decodes uint8[] as base64 by default.
        "data": base64.b64encode(data).decode("ascii"),
    }
    pub.publish(roslibpy.Message(msg))
    time.sleep(1.0)
    pub.unadvertise()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--text", default="BRIDGE OK")
    ap.add_argument(
        "--no-face", action="store_true", help="skip the xdisplay publish (subscribe proof only)"
    )
    args = ap.parse_args()

    ros = roslibpy.Ros(host=args.host, port=args.port)
    ros.run()
    t0 = time.time()
    while not ros.is_connected and time.time() - t0 < 10:
        time.sleep(0.1)
    if not ros.is_connected:
        print("FAIL: could not connect to rosbridge at %s:%d" % (args.host, args.port))
        return 1
    print("connected to rosbridge at %s:%d" % (args.host, args.port))

    print("\n[1/2] subscribe /robot/joint_states for %.0fs ..." % args.seconds)
    n, hz, sample = measure_joint_states(ros, args.seconds)
    print("  received %d msgs -> %.1f Hz through the bridge" % (n, hz))
    if sample:
        names = sample.get("name", [])
        pos = sample.get("position", [])
        if "left_w2" in names:
            i = names.index("left_w2")
            print("  sample: %d joints, left_w2=%.4f" % (len(names), pos[i]))
    verdict = "OK (>=20 Hz loop viable)" if hz >= 20 else "LOW (<20 Hz)"
    print("  rate verdict: %s" % verdict)

    if not args.no_face:
        print("\n[2/2] publish '%s' to /robot/xdisplay (face) ..." % args.text)
        publish_face(ros, args.text)
        print("  published. Check Baxter's face screen.")
    else:
        print("\n[2/2] skipped (--no-face)")

    ros.terminate()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
