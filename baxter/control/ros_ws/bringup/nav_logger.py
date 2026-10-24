#!/usr/bin/env python
"""Navigator + cuff button mapper (Python 2 / Baxter SDK, runs in container).

Watches the left (or right) arm's Navigator buttons/wheel and the cuff
DigitalIOs, printing every state change so we can learn which name maps to
which physical control -- needed for the teaching UI ("squeeze cuff to
record"). No arm motion; this only reads inputs.

    python /root/ros_ws/bringup/nav_logger.py --arm left --seconds 45
"""

from __future__ import print_function
import argparse
import time

import rospy
import baxter_interface as b
from baxter_interface.digital_io import DigitalIO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    ap.add_argument("--seconds", type=float, default=45.0)
    args = ap.parse_args()

    rospy.init_node("navmap", anonymous=True)
    nav = b.Navigator(args.arm)

    dios = {}
    for n in (
        "%s_lower_cuff" % args.arm,
        "%s_upper_button" % args.arm,
        "%s_lower_button" % args.arm,
    ):
        try:
            dios[n] = DigitalIO(n)
        except Exception as e:
            print("could not open DigitalIO", n, ":", e)

    time.sleep(1.0)

    def snap():
        s = {
            "nav.button0": nav.button0,
            "nav.button1": nav.button1,
            "nav.button2": nav.button2,
            "nav.wheel": nav.wheel,
        }
        for n, d in dios.items():
            s["dio." + n] = d.state
        return s

    state = snap()
    print("baseline:", state)
    print("--- watching %.0fs on %s arm; press things now ---" % (args.seconds, args.arm))
    t0 = time.time()
    seen = set()
    while time.time() - t0 < args.seconds and not rospy.is_shutdown():
        cur = snap()
        for k in cur:
            if cur[k] != state.get(k):
                print("%6.1fs  %-22s %s -> %s" % (time.time() - t0, k, state.get(k), cur[k]))
                seen.add(k)
        state = cur
        time.sleep(0.03)
    print("--- done. controls that changed: %s ---" % (", ".join(sorted(seen)) if seen else "NONE"))


if __name__ == "__main__":
    main()
