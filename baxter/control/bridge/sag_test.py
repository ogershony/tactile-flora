#!/usr/bin/env python3
"""Measure how far Baxter's fingertips sit below where they're commanded.

Per arm, a few fixed fingertip poses above the table (Baxter's IK service,
home orientation). At each: hold the command (streamed), then report the
fingertip z commanded vs measured (dream FK of the joints) vs Baxter's own
endpoint state. At the 0.20 m poses also stop commanding for 1.5 s and report
the drop (Baxter's hold mode after a command timeout). Every pose stays >= 0.12 m
above the table. Runs in .venv-dream (needs dream_ik for FK).

    .venv-dream/bin/python bridge/sag_test.py [--sides left right]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baxter_bridge import JOINTS, POSITION_MODE, BaxterBridge  # noqa: E402
from home import home  # noqa: E402
import roslibpy  # noqa: E402

ARM_BASE_DZ = 0.164  # dream's per-side Baxter base sits 0.164 m below Baxter `base`
POSES = [(0.60, 0.20), (0.60, 0.12), (0.75, 0.20), (0.75, 0.12)]  # (x, z) m, Baxter base frame
HOLD_S = 2.5
DROP_S = 1.5


def hold(bx, side, q, seconds, rate=50.0):
    """Stream one joint command for `seconds` (Baxter times out without it)."""
    pub = bx._joint_cmd_pub(side)
    names = JOINTS[side]
    t0 = time.time()
    while time.time() - t0 < seconds:
        pub.publish(
            roslibpy.Message(
                {"mode": POSITION_MODE, "command": [q[n] for n in names], "names": names}
            )
        )
        time.sleep(1.0 / rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", nargs="+", default=["left", "right"], choices=["left", "right"])
    args = ap.parse_args()

    from dream_ik.solver import ChunkIK
    from dream_ik.specs import SERVE

    fk = ChunkIK("baxter", SERVE.tool, 20.0).fk

    def tip_z(side, q):
        return (
            float(np.asarray(fk(np.array([q[n] for n in JOINTS[side]]), side))[2, 3]) - ARM_BASE_DZ
        )

    bx = BaxterBridge().connect()
    time.sleep(0.5)
    try:
        home(bx)
        print(
            "\n%-5s %5s %5s | %8s %8s %8s | %6s %6s"
            % ("side", "x", "z_cmd", "fk(cmd)", "fk(meas)", "endpoint", "sag", "drop")
        )
        for side in args.sides:
            y = bx.endpoint_pose(side)["position"]["y"]
            o = bx.endpoint_pose(side)["orientation"]
            quat = (o["x"], o["y"], o["z"], o["w"])
            for x, z in POSES:
                q = bx.ik(side, (x, y, z), quat)
                if q is None:
                    print("%-5s %5.2f %5.2f | IK has no solution, skipped" % (side, x, z))
                    continue
                bx.move_to(side, q, max_speed=0.3)
                hold(bx, side, q, HOLD_S)
                meas = bx.joint_angles(side)
                ep = bx.endpoint_pose(side)["position"]["z"]
                zc, zm = tip_z(side, q), tip_z(side, meas)
                drop = ""
                if z >= 0.2:
                    time.sleep(DROP_S)  # no commands: Baxter's own hold
                    drop = "%6.3f" % (ep - bx.endpoint_pose(side)["position"]["z"])
                print(
                    "%-5s %5.2f %5.2f | %8.3f %8.3f %8.3f | %6.3f %6s"
                    % (side, x, z, zc, zm, ep, zc - ep, drop),
                    flush=True,
                )
            home(bx, sides=(side,))
        print(
            "\nsag = fk(cmd) - endpoint (m, + = fingertips lower than commanded); "
            "drop = endpoint fall in %.1f s after commands stop" % DROP_S
        )
    finally:
        bx.close()


if __name__ == "__main__":
    main()
