#!/usr/bin/env python3
"""Teach a Baxter skill by hand (zero-G), capturing object-relative keyframes.

Run in YOUR terminal (live feedback matters):

    .venv/bin/python bridge/teach.py pick --arm left --object 0.72 0.15

How to teach:
  1. SQUEEZE the cuff and move the arm to a pose (zero-G, free to move).
  2. Use the OVAL cuff button to open/close the gripper (native grasp button).
  3. Press ENTER in this terminal to CAPTURE a keyframe (pose + gripper state).
  4. Repeat for each waypoint (e.g. approach-above, lower+grip, lift).
  5. Type q + Enter (or Ctrl-C) to save -> skills/<name>.json

--object X Y records the taught object's table coords so the skill is
object-relative (replays offset to wherever the object is). Omit for an
absolute skill.
"""

import argparse
import select
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from baxter_bridge import BaxterBridge
from skills import Skill, Keyframe


def classify_gripper(gs):
    if gs.get("gripping") or gs.get("position", 100) < 30:
        return "close"
    return "open"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    ap.add_argument(
        "--object",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="taught object table (x,y) for object-relative",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or "skills/%s.json" % args.name

    bx = BaxterBridge().connect()
    bx.set_face_text("TEACH: " + args.name)
    print("Teaching '%s' on %s arm. Object ref: %s" % (args.name, args.arm, args.object))
    print(
        "Squeeze cuff + move. OVAL button = gripper (native). "
        "ENTER = capture keyframe. q+Enter = save.\n"
    )

    keyframes = []
    try:
        while True:
            # non-blocking: capture on Enter, quit on 'q'
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                break
            pose = bx.endpoint_pose(args.arm)
            if not pose:
                print("  (no endpoint pose yet, hold on)")
                continue
            grip = classify_gripper(bx.gripper_state(args.arm))
            kf = Keyframe("kf%d" % (len(keyframes) + 1), pose, grip)
            keyframes.append(kf)
            p = pose["position"]
            print(
                "  captured kf%d: xyz=(%.3f,%.3f,%.3f) gripper=%s"
                % (len(keyframes), p["x"], p["y"], p["z"], grip)
            )
            bx.set_face_text("kf %d" % len(keyframes))
    except KeyboardInterrupt:
        pass

    if not keyframes:
        print("\nNo keyframes captured; nothing saved.")
        bx.close()
        return
    # reference_xy = object position at teach time. The most reliable source is
    # the grasp keyframe (where the gripper closed = where the object was);
    # a separate --object detection can go stale if the object moved.
    grasp_kf = next((k for k in keyframes if k.gripper == "close"), None)
    if grasp_kf is not None:
        ref = (grasp_kf.pose["position"]["x"], grasp_kf.pose["position"]["y"])
        print("reference_xy from grasp keyframe: (%.3f, %.3f)" % ref)
    elif args.object:
        ref = tuple(args.object)
    else:
        ref = None
    skill = Skill(args.name, args.arm, keyframes, reference_xy=ref)
    skill.save(out)
    bx.set_face_text("saved")
    print("\nSaved %d keyframes -> %s" % (len(keyframes), out))
    bx.close()


if __name__ == "__main__":
    main()
