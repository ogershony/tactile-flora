#!/usr/bin/env python3
"""Forward kinematics for Baxter's arms from config/baxter.urdf, numpy only.

For loops that run outside .venv-dream (no dream_ik), e.g. il_control in .venv-lerobot.
The tool point is the URDF's <side>_gripper link (<side>_gripper_base + 0.1327 m), Baxter's
own endpoint, so z matches bx.endpoint_pose() (table ~0.003, Baxter base frame).

    from baxter_fk import fingertip
    fingertip("left", q7)      # -> (x, y, z), m in Baxter base frame
"""

import functools
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "config", "baxter.urdf")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _rot(axis, a):
    """Rotation by angle a about a unit axis (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


@functools.lru_cache(maxsize=None)
def _chain(side, tip):
    """[(fixed 4x4 origin, axis or None)] from `base` to `tip`, in order."""
    joints = {j.find("child").get("link"): j for j in ET.parse(URDF).getroot().findall("joint")}
    chain, link = [], tip
    while link in joints:
        j = joints[link]
        o = j.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = _rpy(*map(float, o.get("rpy", "0 0 0").split()))
            T[:3, 3] = list(map(float, o.get("xyz", "0 0 0").split()))
        axis = None
        if j.get("type") == "revolute":
            if not j.get("name").startswith(side + "_"):
                raise RuntimeError("unexpected joint %s on the %s chain" % (j.get("name"), side))
            axis = np.array(list(map(float, j.find("axis").get("xyz").split())))
        chain.append((T, axis))
        link = j.find("parent").get("link")
    if link != "base":
        raise RuntimeError("%s does not chain back to base" % tip)
    return tuple(reversed(chain))


def fk(side, q, tip=None):
    """4x4 pose of `tip` (default <side>_gripper) for the 7 joints q (s0..w2, rad)."""
    T, k = np.eye(4), 0
    for origin, axis in _chain(side, tip or side + "_gripper"):
        T = T @ origin
        if axis is not None:
            R = np.eye(4)
            R[:3, :3] = _rot(axis, float(q[k]))
            T, k = T @ R, k + 1
    if k != 7:
        raise RuntimeError("expected 7 revolute joints, found %d" % k)
    return T


def fingertip(side, q):
    return fk(side, q)[:3, 3]
