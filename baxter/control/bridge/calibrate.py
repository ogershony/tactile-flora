#!/usr/bin/env python3
"""Calibrate the overhead USB webcam -> table (Baxter base-frame) mapping.

The webcam is FIXED, so there's no camera pose to manage -- calibrate once.
At several REACHABLE, spread-out spots: detect the block in the webcam, then
touch it with the gripper (zero-G) to read its base-frame (x,y). A homography
is fit from the pixel<->xy pairs and saved to config/table_homography.json.

  grab        : capture a webcam frame + detect, save images (check the view)
  collect     : interactive pixel<->xy pairing -> saves the homography
  homography  : build from 4 manual correspondences (--pixels / --table)

Only the touch uses the arm (you move it in zero-G); the camera is independent.
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cv2
from baxter_bridge import BaxterBridge
from webcam import Webcam
from perception import TableHomography, detect_blocks, annotate

HOMO_PATH = "config/table_homography.json"


def grab(cam):
    img = cam.frame()
    objs = detect_blocks(img)
    cv2.imwrite("usb_cam.jpg", img)
    cv2.imwrite("usb_cam_annotated.jpg", annotate(img, objs))
    print("frame %s -> %d blob(s) (saved usb_cam.jpg):" % (img.shape, len(objs)))
    for o in objs:
        print(
            "  %-6s pixel=(%.0f,%.0f) area=%.0f"
            % (o["color"], o["pixel"][0], o["pixel"][1], o["area"])
        )


def collect(bx, cam, arm, color, n):
    pairs = []
    print("\nPair the %s block's webcam pixel with its touched (x,y), >=4 times." % color)
    print("Place it at SPREAD, clearly REACHABLE spots (don't let the arm shove the robot).\n")
    while len(pairs) < n:
        i = len(pairs) + 1
        input("[%d/%d] block placed -> Enter to detect in webcam..." % (i, n))
        objs = [o for o in detect_blocks(cam.frame()) if o["color"] == color]
        if not objs:
            print("   no %s blob seen; reposition and retry." % color)
            continue
        px = objs[0]["pixel"]
        h, w = 480, 640
        m = 50
        if px[0] < m or px[0] > w - m or px[1] < m or px[1] > h - m:
            print(
                "   !! block near frame edge (%.0f,%.0f) -> unreliable pixel; "
                "move it toward the middle and redo." % (px[0], px[1])
            )
            continue
        print("   detected %s at pixel (%.0f,%.0f)" % (color, px[0], px[1]))
        input("   now zero-G the gripper to GRASP the block -> Enter to read xy...")
        ep = bx.endpoint_pose(arm)["position"]
        xy = (ep["x"], ep["y"])
        print("   endpoint table xy = (%.3f, %.3f)" % xy)
        if any((p[1][0] - xy[0]) ** 2 + (p[1][1] - xy[1]) ** 2 < 0.04**2 for p in pairs):
            print(
                "   !! within 4cm of an earlier point -> likely a reach-limit/"
                "garbage reading. NOT recorded; move within reach and redo."
            )
            continue
        pairs.append((px, xy))
    H = TableHomography.from_points([p[0] for p in pairs], [p[1] for p in pairs])
    H.save(HOMO_PATH)
    print("\nsaved homography -> %s" % HOMO_PATH)
    err = []
    for px, tb in pairs:
        got = H.pixel_to_table(px)
        e = ((got[0] - tb[0]) ** 2 + (got[1] - tb[1]) ** 2) ** 0.5
        err.append(e)
        print(
            "  check: pixel (%.0f,%.0f) -> (%.3f,%.3f) vs measured (%.3f,%.3f)  %.1fmm"
            % (px[0], px[1], got[0], got[1], tb[0], tb[1], e * 1000)
        )
    print("mean residual: %.1f mm" % (1000 * sum(err) / len(err)))


def test(bx, cam, arm, color):
    """Held-out accuracy check: detect -> predict xy -> touch -> report error."""
    H = TableHomography.load(HOMO_PATH)
    objs = [o for o in detect_blocks(cam.frame()) if o["color"] == color]
    if not objs:
        print("no %s block seen." % color)
        return
    px = objs[0]["pixel"]
    pred = H.pixel_to_table(px)
    print(
        "detected pixel (%.0f,%.0f) -> PREDICTED table (%.3f, %.3f)"
        % (px[0], px[1], pred[0], pred[1])
    )
    input("zero-G the gripper to GRASP the block -> Enter to read ACTUAL xy...")
    ep = bx.endpoint_pose(arm)["position"]
    e = ((pred[0] - ep["x"]) ** 2 + (pred[1] - ep["y"]) ** 2) ** 0.5
    print(
        "actual (%.3f,%.3f)  predicted (%.3f,%.3f)  ERROR %.1f mm"
        % (ep["x"], ep["y"], pred[0], pred[1], e * 1000)
    )


def build_homography(pixels, table):
    px = [(pixels[i], pixels[i + 1]) for i in range(0, 8, 2)]
    tb = [(table[i], table[i + 1]) for i in range(0, 8, 2)]
    TableHomography.from_points(px, tb).save(HOMO_PATH)
    print("saved homography -> %s" % HOMO_PATH)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grab")
    g.add_argument("--index", type=int, default=0)
    c = sub.add_parser("collect")
    c.add_argument("--arm", default="left")
    c.add_argument("--color", default="red")
    c.add_argument("--n", type=int, default=5)
    c.add_argument("--index", type=int, default=0)
    h = sub.add_parser("homography")
    h.add_argument("--pixels", nargs=8, type=float, required=True)
    h.add_argument("--table", nargs=8, type=float, required=True)
    t = sub.add_parser("test")
    t.add_argument("--arm", default="left")
    t.add_argument("--color", default="red")
    t.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "homography":
        build_homography(args.pixels, args.table)
        return
    cam = Webcam(args.index)
    if args.cmd == "grab":
        grab(cam)
    elif args.cmd == "collect":
        bx = BaxterBridge().connect()
        collect(bx, cam, args.arm, args.color, args.n)
        bx.close()
    elif args.cmd == "test":
        bx = BaxterBridge().connect()
        test(bx, cam, args.arm, args.color)
        bx.close()
    cam.close()


if __name__ == "__main__":
    main()
