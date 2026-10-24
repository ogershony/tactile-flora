#!/usr/bin/env python3
"""Perception for the Baxter demo: wrist-cam pixels -> table (x,y) + block finding.

Two pieces:
  1. TableHomography -- a 3x3 projective map from image pixels to table
     coordinates (metres, Baxter base frame xy), computed from 4 corner
     correspondences captured once from a fixed look-down camera pose
     (see calibrate.py). Save/load as JSON.
  2. detect_blocks / find_objects -- HSV color thresholding to locate colored
     blocks/cups and (with a homography) report their table (x, y).

The output feeds the object-relative skills: Skill.execute(bx, object_xy=...).

    from perception import TableHomography, find_objects
    H = TableHomography.load('config/table_homography.json')
    objs = find_objects(bx.camera_frame('left_hand_camera'), H)
    # -> [{'color':'red','table_xy':(0.72,0.15),'pixel':(..),'area':..}, ...]
"""

import json
import os

import cv2
import numpy as np

# HSV ranges (OpenCV H in 0..179). Red wraps the hue seam so it needs two bands.
# On the bare wood table (warm, S~110) colored blocks are separated by high
# saturation: measured red block S~195. So S floor ~150 excludes wood.
# "yellow" is dropped -- indistinguishable from tan wood. If a neutral/dark
# mat is added later, these can be loosened. Tune with perception.py --mask.
COLOR_HSV = {
    "red": [((0, 150, 60), (12, 255, 255)), ((165, 150, 60), (179, 255, 255))],
    "green": [((38, 120, 50), (90, 255, 255))],
    "blue": [((90, 120, 50), (135, 255, 255))],
}


class TableHomography(object):
    def __init__(self, H):
        self.H = np.asarray(H, dtype=np.float64)

    @classmethod
    def from_points(cls, pixel_pts, table_pts):
        """pixel_pts, table_pts: 4 matching (u,v) and (x,y), same order."""
        src = np.asarray(pixel_pts, dtype=np.float32)
        dst = np.asarray(table_pts, dtype=np.float32)
        if len(src) == 4:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            # RANSAC rejects outlier correspondences (threshold in table metres).
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.02)
            if mask is not None:
                print("  homography inliers: %d/%d" % (int(mask.sum()), len(src)))
        return cls(H)

    def pixel_to_table(self, uv):
        p = np.array([uv[0], uv[1], 1.0])
        q = self.H.dot(p)
        return (float(q[0] / q[2]), float(q[1] / q[2]))

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"H": self.H.tolist()}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))["H"])


def detect_blocks(img_bgr, colors=None, min_area=100):
    """Return [{'color','pixel':(u,v),'area'}] for colored blobs in the image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    out = []
    for name, ranges in (colors or COLOR_HSV).items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            out.append(
                {"color": name, "pixel": (M["m10"] / M["m00"], M["m01"] / M["m00"]), "area": a}
            )
    return sorted(out, key=lambda o: -o["area"])


def find_objects(img_bgr, homography, colors=None, min_area=150):
    objs = detect_blocks(img_bgr, colors, min_area)
    for o in objs:
        o["table_xy"] = homography.pixel_to_table(o["pixel"])
    return objs


def annotate(img_bgr, objs):
    """Draw detections for debugging; returns a copy."""
    im = img_bgr.copy()
    for o in objs:
        u, v = int(o["pixel"][0]), int(o["pixel"][1])
        cv2.circle(im, (u, v), 8, (255, 255, 255), 2)
        txt = o["color"]
        if "table_xy" in o:
            txt += " (%.2f,%.2f)" % o["table_xy"]
        cv2.putText(im, txt, (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from baxter_bridge import BaxterBridge

    ap = argparse.ArgumentParser(description="detect color blobs in a wrist cam frame")
    ap.add_argument("--cam", default="left_hand_camera")
    ap.add_argument("--save", default="cam_detect.jpg")
    ap.add_argument("--mask", default=None, help="color name to dump its threshold mask (e.g. red)")
    args = ap.parse_args()

    bx = BaxterBridge().connect()
    img = bx.camera_frame(args.cam)
    bx.close()
    if img is None:
        print("no frame")
        sys.exit(1)
    if args.mask:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = None
        for lo, hi in COLOR_HSV[args.mask]:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = mm if m is None else cv2.bitwise_or(m, mm)
        cv2.imwrite("mask_%s.jpg" % args.mask, m)
        print("mask -> mask_%s.jpg (white = passes %s threshold)" % (args.mask, args.mask))
    objs = detect_blocks(img)
    print("frame %s, %d blob(s):" % (img.shape, len(objs)))
    for o in objs:
        print(
            "  %-7s pixel=(%.0f,%.0f) area=%.0f"
            % (o["color"], o["pixel"][0], o["pixel"][1], o["area"])
        )
    cv2.imwrite(args.save, annotate(img, objs))
    print("annotated frame -> %s" % args.save)
