#!/usr/bin/env python3
"""Skill model + executor for the Baxter demo.

A Skill is a named, ordered list of keyframes taught by hand (see teach.py).
Each keyframe stores the gripper's Cartesian endpoint pose + the gripper
action at that point. Skills are object-relative: at teach time we note the
object's table (x, y); at run time we shift every keyframe's x/y by
(object_now - object_taught), keep z and orientation, solve IK, and move.
That's what lets one taught 'pick' work wherever the block ends up.

    from baxter_bridge import BaxterBridge
    from skills import Skill
    bx = BaxterBridge().connect()
    Skill.load('skills/pick.json').execute(bx, object_xy=(0.72, 0.15))

Executor returns 'ok' | 'ik_fail' | 'grasp_fail' so the planner can re-plan.
"""

import json
import os

GRASP_POS_THRESHOLD = 15.0  # gripper position above this with gripping==True = holding


class Keyframe(object):
    def __init__(self, label, pose, gripper, max_speed=0.3, dwell=0.3):
        self.label = label
        self.pose = pose  # {"position":{x,y,z}, "orientation":{x,y,z,w}}
        self.gripper = gripper  # "open" | "close" | None
        self.max_speed = max_speed
        self.dwell = dwell

    def to_dict(self):
        return {
            "label": self.label,
            "pose": self.pose,
            "gripper": self.gripper,
            "max_speed": self.max_speed,
            "dwell": self.dwell,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["label"], d["pose"], d.get("gripper"), d.get("max_speed", 0.3), d.get("dwell", 0.3)
        )


class Skill(object):
    def __init__(self, name, arm, keyframes, reference_xy=None):
        self.name = name
        self.arm = arm
        self.keyframes = keyframes
        self.reference_xy = reference_xy  # object (x,y) at teach time, or None

    # --- persistence ------------------------------------------------------
    def to_dict(self):
        return {
            "name": self.name,
            "arm": self.arm,
            "reference_xy": self.reference_xy,
            "keyframes": [k.to_dict() for k in self.keyframes],
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(
            d["name"],
            d["arm"],
            [Keyframe.from_dict(k) for k in d["keyframes"]],
            d.get("reference_xy"),
        )

    # --- execution --------------------------------------------------------
    def execute(self, bx, object_xy=None, verbose=True):
        """Replay keyframes, offset to object_xy if the skill is object-relative.
        Returns 'ok' | 'ik_fail' | 'grasp_fail'."""
        dx = dy = 0.0
        if self.reference_xy is not None and object_xy is not None:
            dx = object_xy[0] - self.reference_xy[0]
            dy = object_xy[1] - self.reference_xy[1]
        for kf in self.keyframes:
            p, o = kf.pose["position"], kf.pose["orientation"]
            sol = bx.ik(
                self.arm, (p["x"] + dx, p["y"] + dy, p["z"]), (o["x"], o["y"], o["z"], o["w"])
            )
            if sol is None:
                if verbose:
                    print("  [%s] IK failed" % kf.label)
                return "ik_fail"
            if verbose:
                print("  [%s] move (gripper=%s)" % (kf.label, kf.gripper))
            bx.move_to(self.arm, sol, max_speed=kf.max_speed)
            if kf.gripper in ("open", "close"):
                bx.gripper(self.arm, kf.gripper)
            if kf.dwell:
                import time

                time.sleep(kf.dwell)
            # grasp check right after a close
            if kf.gripper == "close":
                gs = bx.gripper_state(self.arm)
                if not (gs.get("gripping") and gs.get("position", 0) > GRASP_POS_THRESHOLD):
                    if verbose:
                        print("  grasp check FAILED: %s" % gs)
                    return "grasp_fail"
        return "ok"
