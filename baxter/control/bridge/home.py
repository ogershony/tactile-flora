#!/usr/bin/env python3
"""Send both arms home (config/rest_pose.json) without dragging over the table.

A hand low over the table (e.g. after a Dream run) is first lifted by raising
its shoulder (s1), then both arms move to the rest pose, speed-capped. Low is
judged by Baxter's own endpoint state, not commanded joints: once commands stop
the arms settle 3-4 cm under gravity (2026-10-22).

    .venv/bin/python bridge/home.py
    from home import home; home(bx)            # from a control loop's cleanup
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baxter_bridge import BaxterBridge  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOW_Z = 0.07  # hand below this (m, Baxter base frame; table ~0.0; rest pose 0.084-0.095) gets lifted first
LIFT_S1 = 0.3  # rad; s1 more negative raises the arm
SPEED = 0.3  # rad/s


def home(bx, sides=("left", "right"), speed=SPEED, log=print):
    rest = json.load(open(os.path.join(ROOT, "config", "rest_pose.json")))
    for side in sides:
        z = bx.endpoint_pose(side).get("position", {}).get("z")
        if z is None or z < LOW_Z:
            q = bx.joint_angles(side)
            q["%s_s1" % side] -= LIFT_S1
            log("%s: hand at z=%s, lifting first" % (side, "?" if z is None else "%.3f" % z))
            bx.move_to(side, q, max_speed=speed)
    for side in sides:
        err = bx.move_to(side, rest[side], max_speed=speed)
        log("%s: home (err %.3f rad)" % (side, err))


def main():
    bx = BaxterBridge().connect()
    time.sleep(0.5)  # let endpoint states arrive
    try:
        home(bx)
    finally:
        bx.close()


if __name__ == "__main__":
    main()
