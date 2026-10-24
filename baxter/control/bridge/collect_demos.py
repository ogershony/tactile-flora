#!/usr/bin/env python3
"""Collect scripted, self-resetting pick demonstrations for a Dream fine-tune.

The overhead webcam (mounted on Baxter's head) finds the red cube; the arm on its
side picks it from home while demo_recorder records (cameras, joints, tool poses);
then, unrecorded, the cube goes down at a new random spot and the arms clear the
view for the next detection. Episodes whose grasp fails are not kept.

    --present        left gripper opens at a hand-over spot: put the cube on the table below it
    --calibrate      then: the arms place the cube at a grid of known spots and fit
                     config/table_homography_head.json
    --episodes N     collect N kept episodes into demos/<date>/epNNNN (MOTION)
    --detect         print the cube's pixel + table xy with arms in the view pose
    --hand N         hand-guided episodes, in YOUR terminal: Enter = record / save, d = discard,
                     q = quit; squeeze the cuff to guide, OVAL cuff button = gripper

Grasps use one fixed orientation for both arms (pointing down, fingers closing along
Baxter's y), so the cube stays square to the table and either arm can take it.
Heights are Baxter endpoint z (fingertip point, table ~0.003; cube 40 mm).

    .venv-dream/bin/python bridge/collect_demos.py --calibrate
    .venv-dream/bin/python bridge/collect_demos.py --episodes 50
"""

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baxter_bridge import JOINTS, BaxterBridge  # noqa: E402
from home import home  # noqa: E402
from perception import COLOR_HSV, TableHomography  # noqa: E402
from demo_recorder import FPS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOMOGRAPHY = os.path.join(ROOT, "config", "table_homography_head.json")
TASK = "pick up the red cube"
DOWN = (0.0, 1.0, 0.0, 0.0)  # endpoint quaternion (x,y,z,w): z down, fingers close along base y
Z_UP, Z_GRASP, Z_PLACE = 0.15, 0.030, 0.035
X_RANGE = (0.60, 0.80)
Y_RANGE = {"left": (-0.02, 0.26), "right": (-0.26, 0.02)}  # IK: no (0.82, +-0.04) across the centre
SPEED = 0.3
VIEW_OFFSET = {
    "left": {"left_s0": 0.5, "left_s1": -0.4},
    "right": {"right_s0": -0.5, "right_s1": -0.4},
}
CUBE_AREA = (60, 4000)  # px, cube blob in the 640x480 webcam frame
EDGE, TOP_BAND = 12, 50  # px: blobs touching the border (the arms) / above the table


def find_cube(img):
    """The red cube's pixel (u, v), or None. Baxter's red arm covers touch the frame
    border, and the background sits in the top band, so both are left out."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in COLOR_HSV["red"])
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not CUBE_AREA[0] <= area <= CUBE_AREA[1]:
            continue
        if (
            x <= EDGE
            or y <= EDGE
            or x + bw >= w - EDGE
            or y + bh >= h - EDGE
            or cents[i][1] < TOP_BAND
        ):
            continue
        if not 0.4 <= bw / float(bh) <= 2.5:
            continue
        if best is None or area > best[1]:
            best = (tuple(cents[i]), area)
    return None if best is None else best[0]


class Rig(object):
    def __init__(self):
        import dream_control
        from webcam import WebcamStream

        self.bx = BaxterBridge().connect()
        time.sleep(0.5)
        self.rest = json.load(open(os.path.join(ROOT, "config", "rest_pose.json")))
        self.cam = WebcamStream(dream_control.WEBCAM)
        self.wrists = {
            "wrist_l": self.bx.stream_camera("left_hand_camera", 1000 // FPS),
            "wrist_r": self.bx.stream_camera("right_hand_camera", 1000 // FPS),
        }

    def close(self):
        for s in self.wrists.values():
            s.close()
        self.cam.close()
        self.bx.close()

    # --- motion -----------------------------------------------------------
    def move(self, side, xyz, quat=DOWN, speed=SPEED):
        q = self.bx.ik(side, xyz, quat)
        if q is None:
            raise RuntimeError(
                "no IK for %s at (%.3f, %.3f, %.3f)" % (side, xyz[0], xyz[1], xyz[2])
            )
        self.bx.move_to(side, q, max_speed=speed)

    def view_pose(self):
        """Both arms raised and swung outward, out of the overhead camera's way."""
        for side in ("left", "right"):
            q = dict(self.rest[side])
            for n, d in VIEW_OFFSET[side].items():
                q[n] += d
            z = self.bx.endpoint_pose(side).get("position", {}).get("z", 1.0)
            if z < Z_UP - 0.02:  # never sweep sideways low over the table
                p = self.bx.endpoint_pose(side)["position"]
                self.move(side, (p["x"], p["y"], Z_UP))
            self.bx.move_to(side, q, max_speed=SPEED)

    def home(self):
        for side in ("left", "right"):
            p = self.bx.endpoint_pose(side).get("position")
            if p and p["z"] < Z_UP - 0.02:
                self.move(side, (p["x"], p["y"], Z_UP))
        home(self.bx, speed=SPEED, log=lambda *_: None)

    def grasp(self, side, xy):
        """Above -> down -> close -> up. True when the gripper reports holding."""
        self.bx.gripper(side, "open")
        self.move(side, (xy[0], xy[1], Z_UP))
        self.move(side, (xy[0], xy[1], Z_GRASP), speed=0.2)
        self.bx.gripper(side, "close")
        self.move(side, (xy[0], xy[1], Z_UP), speed=0.2)
        time.sleep(0.3)
        return bool(self.bx.gripper_state(side).get("gripping"))

    def place(self, side, xy):
        self.move(side, (xy[0], xy[1], Z_UP))
        self.move(side, (xy[0], xy[1], Z_PLACE), speed=0.2)
        self.bx.gripper(side, "open")
        self.move(side, (xy[0], xy[1], Z_UP), speed=0.2)

    def detect(self, tries=5):
        for _ in range(tries):
            time.sleep(0.3)
            img, _ = self.cam.latest()
            px = None if img is None else find_cube(img)
            if px is not None:
                return px, img
        return None, img


def side_for(y):
    return "left" if y >= 0 else "right"


HANDOVER = (0.68, 0.15, Z_UP)


def present(rig):
    """Left gripper open at the hand-over spot, for the cube (then --calibrate)."""
    rig.home()
    rig.move("left", HANDOVER)
    rig.bx.gripper("left", "open")
    print(
        "put the cube on the table right below the open LEFT fingers, square to the edge; then --calibrate"
    )


def calibrate(rig):
    bx = rig.bx  # after --present: cube on the table right below
    if not rig.grasp("left", HANDOVER[:2]):  # the open left fingers, square to the table edge
        raise RuntimeError("left gripper did not pick up the cube below the hand-over spot")
    xs = (0.62, 0.71, 0.80)
    plan = [("left", (x, y)) for y in (0.24, 0.12, 0.0) for x in xs] + [
        ("right", (x, y)) for y in (0.0, -0.12, -0.24) for x in xs
    ]
    pixels, table, holding, last = [], [], "left", None
    for side, xy in plan:
        if side != holding:  # hand over at the centre line via the table
            rig.place(holding, last)
            rig.home()
            if not rig.grasp(side, last):
                raise RuntimeError("%s arm failed to take the cube at %s" % (side, last))
            holding = side
        rig.place(side, xy)
        rig.view_pose()
        px, img = rig.detect()
        if px is None:
            cv2.imwrite(os.path.join(ROOT, "calib_miss.jpg"), img)
            print("  (%.2f, %+.2f): cube not seen (saved calib_miss.jpg), skipped" % xy, flush=True)
        else:
            pixels.append(px)
            table.append(xy)
            print(
                "  (%.2f, %+.2f) -> pixel (%.0f, %.0f)" % (xy[0], xy[1], px[0], px[1]), flush=True
            )
        rig.home()
        if not rig.grasp(side, xy):
            raise RuntimeError("%s arm failed to re-grasp at %s" % (side, xy))
        last = xy
    if len(pixels) < 6:
        raise RuntimeError("only %d calibration points seen" % len(pixels))
    H = TableHomography.from_points(pixels, table)
    err = [np.hypot(*(np.array(H.pixel_to_table(p)) - np.array(t))) for p, t in zip(pixels, table)]
    H.save(HOMOGRAPHY)
    print(
        "homography from %d points: residual mean %.1f mm, max %.1f mm -> %s"
        % (len(pixels), 1000 * np.mean(err), 1000 * np.max(err), HOMOGRAPHY)
    )
    rig.place(holding, last)  # leave the cube on the table for collection
    rig.home()


def collect(rig, episodes, out_root):
    from demo_recorder import EpisodeRecorder

    H = TableHomography.load(HOMOGRAPHY)
    sources = {"scene": rig.cam, **rig.wrists}
    rec = EpisodeRecorder(rig.bx, sources)
    os.makedirs(out_root, exist_ok=True)
    kept = len([d for d in os.listdir(out_root) if d.startswith("ep")])
    misses = 0
    while kept < episodes:
        rig.view_pose()
        px, img = rig.detect()
        if px is None:
            cv2.imwrite(os.path.join(ROOT, "collect_miss.jpg"), img)
            raise RuntimeError(
                "cube not found (saved collect_miss.jpg): put it back in the work area"
            )
        xy = H.pixel_to_table(px)
        side = side_for(xy[1])
        if not (X_RANGE[0] - 0.03 <= xy[0] <= X_RANGE[1] + 0.03 and abs(xy[1]) <= 0.30):
            raise RuntimeError("cube at (%.2f, %+.2f) is outside the work area" % tuple(xy))
        rig.home()
        print(
            "episode %d: cube at (%.3f, %+.3f), %s arm" % (kept + 1, xy[0], xy[1], side), flush=True
        )
        rec.start(TASK)
        time.sleep(0.5)  # a still start, as a run begins
        ok = rig.grasp(side, xy)
        time.sleep(0.5)
        n = rec.stop()
        if ok:
            kept += 1
            rec.write(os.path.join(out_root, "ep%04d" % kept))
            print("  kept: %d frames" % n, flush=True)
            misses = 0
        else:
            misses += 1
            print("  grasp failed, not kept (%d in a row)" % misses, flush=True)
            if misses >= 3:
                raise RuntimeError("3 failed grasps in a row: check the calibration / cube")
            rig.bx.gripper(side, "open")
            rig.view_pose()
            continue
        # unrecorded reset: put the cube down somewhere new on this arm's side
        new = (random.uniform(*X_RANGE), random.uniform(*Y_RANGE[side]))
        rig.place(side, new)
    rig.home()
    print("done: %d episodes in %s" % (kept, out_root))


def hand(rig, episodes, out_root):
    """Hand-guided demos, run from YOUR terminal: Enter starts an episode, Enter saves
    it (d + Enter discards), q + Enter ends. Squeeze the cuff to move the arm (zero-G);
    the OVAL cuff button works the gripper. Both arms go home before each episode."""
    from demo_recorder import EpisodeRecorder

    print("loading (IK model, ~10 s)...", flush=True)
    rec = EpisodeRecorder(rig.bx, {"scene": rig.cam, **rig.wrists}, action_from="next_state")
    os.makedirs(out_root, exist_ok=True)
    kept = len([d for d in os.listdir(out_root) if d.startswith("ep")])
    while kept < episodes:
        print("\nhoming both arms - hands off the cuffs...", flush=True)
        rig.home()
        if (
            input(
                "[%d/%d kept] place the cube, then Enter to RECORD (q = quit): " % (kept, episodes)
            ).strip()
            == "q"
        ):
            break
        rec.start(TASK)
        ans = input(
            "  RECORDING - squeeze cuff, guide, UPPER cuff button = close, lift.  Enter = save, d = discard: "
        )
        n = rec.stop()
        if ans.strip() == "d" or n < 20:
            print("  discarded (%d frames)" % n, flush=True)
            continue
        if min(min(t[2].values()) for t in rec.ticks) > 0.5:
            # 2026-10-23: 50 episodes recorded with the cuff-control node down -- the
            # gripper never closed and the cube stayed on the table in all of them
            print(
                "  NOT saved: the gripper never closed. Is cuff control running? Upper cuff "
                "button = close, lower = open.",
                flush=True,
            )
            continue
        kept += 1
        rec.write(os.path.join(out_root, "ep%04d" % kept))
        print("  saved ep%04d: %d frames (%.1f s)" % (kept, n, n / float(FPS)), flush=True)
    rig.home()
    print("done: %d episodes in %s" % (kept, out_root))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--present", action="store_true", help="open the left gripper at the hand-over spot"
    )
    g.add_argument("--calibrate", action="store_true", help="after --present + cube in the fingers")
    g.add_argument("--episodes", type=int)
    g.add_argument("--detect", action="store_true")
    g.add_argument(
        "--hand",
        type=int,
        metavar="N",
        help="record up to N hand-guided episodes (run in your terminal)",
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "demos", time.strftime("%Y%m%d")))
    args = ap.parse_args()
    rig = Rig()
    try:
        if args.present:
            present(rig)
            rig.close()
            return
        if args.calibrate:
            calibrate(rig)
        elif args.detect:
            rig.view_pose()
            px, img = rig.detect()
            print("cube pixel:", px)
            if px is not None and os.path.exists(HOMOGRAPHY):
                print(
                    "table xy: (%.3f, %+.3f)"
                    % tuple(TableHomography.load(HOMOGRAPHY).pixel_to_table(px))
                )
            cv2.imwrite(os.path.join(ROOT, "detect.jpg"), img)
            rig.home()
        elif args.hand:
            hand(rig, args.hand, args.out)
        else:
            collect(rig, args.episodes, args.out)
    except BaseException:
        print("stopping: arms home", flush=True)
        try:
            for side in ("left", "right"):
                p = rig.bx.endpoint_pose(side).get("position")
                if p and p["z"] < Z_UP:
                    rig.move(side, (p["x"], p["y"], Z_UP))
            home(rig.bx, log=lambda *_: None)
        finally:
            rig.close()
        raise
    rig.close()


if __name__ == "__main__":
    main()
