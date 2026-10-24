#!/usr/bin/env python3
"""Autonomous perception-driven pick: see the block, grasp it.

Detects the colored block in the overhead webcam, maps its pixel to a table
(x,y) via the calibrated homography, and runs the object-relative 'pick' skill
offset to that position. The whole point of perception + object-relative
skills: the block can be anywhere in the work zone.

    .venv/bin/python bridge/grasp_block.py --color red            # MOTION
    .venv/bin/python bridge/grasp_block.py --color red --dry-run  # detect only

Returns the executor result: ok | ik_fail | grasp_fail.
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from webcam import Webcam
from perception import TableHomography, find_objects
from baxter_bridge import BaxterBridge
from skills import Skill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="red")
    ap.add_argument("--skill", default="skills/pick.json")
    ap.add_argument("--homography", default="config/table_homography.json")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="detect + print the target, NO motion")
    args = ap.parse_args()

    H = TableHomography.load(args.homography)
    cam = Webcam(args.index)
    img = cam.frame()
    cam.close()
    objs = [o for o in find_objects(img, H) if o["color"] == args.color]
    if not objs:
        print("no %s object detected" % args.color)
        return 1
    obj = objs[0]
    print(
        "%s block: pixel (%.0f,%.0f) -> table (%.3f, %.3f)"
        % (args.color, obj["pixel"][0], obj["pixel"][1], obj["table_xy"][0], obj["table_xy"][1])
    )
    if args.dry_run:
        print("dry-run: no motion.")
        return 0

    bx = BaxterBridge().connect()
    skill = Skill.load(args.skill)
    result = skill.execute(bx, object_xy=obj["table_xy"])
    print("RESULT:", result, "| gripper:", bx.gripper_state(skill.arm))
    bx.close()
    return 0 if result == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
