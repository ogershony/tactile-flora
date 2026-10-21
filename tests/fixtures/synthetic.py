"""Synthetic episodes with a known grasp shape (free space, closure, hold, release)."""

from __future__ import annotations

import numpy as np

from tactile_flora.schema import (
    Arm,
    Episode,
    ForceStream,
    GripperStream,
    PadArmForce,
    VideoRef,
    WristArmForce,
)

RATE = 100.0
BASELINE = 1.0
HOLD = 5.0
NOISE = 0.02


def _profile(t: np.ndarray, dur: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (gripper 0..1, force fraction 0..1): open to 0.6 s, close 0.3 s, hold, release."""
    t_close, t_closed = 0.6, 0.9
    t_rel, t_open = dur - 0.5, dur - 0.1
    g = np.interp(t, [0, t_close, t_closed, t_rel, t_open, dur + 1], [0, 0, 1, 1, 0, 0])
    frac = np.interp(t, [0, t_closed, t_closed + 0.1, t_rel, t_open, dur + 1], [0, 0, 1, 1, 0, 0])
    return g, frac


def _episode(
    kind: str, seed: int, n_frames: int, arms: tuple[Arm, ...], with_gripper: bool
) -> Episode:
    rng = np.random.default_rng(seed)
    dur = n_frames * 0.25 + 0.25
    t = np.arange(0, dur, 1 / RATE)
    g, frac = _profile(t, dur)
    force: dict[Arm, PadArmForce | WristArmForce] = {}
    for k, arm in enumerate(arms):
        total = BASELINE + (HOLD - BASELINE) * frac * (1 + 0.1 * k)
        total = total + rng.normal(0, NOISE, t.size)
        if kind == "pad":
            asym = 0.2 * frac
            force[arm] = PadArmForce(f_a=list((total + asym) / 2), f_b=list((total - asym) / 2))
        else:
            zeros = [0.0] * t.size
            force[arm] = WristArmForce(
                fx=list(total * 0.6), fy=list(total * 0.0), fz=list(total * 0.8),
                tx=zeros, ty=zeros, tz=zeros,
            )  # fmt: skip
    stream = ForceStream(channel=kind, arms=force, rate_hz=RATE, t=list(t))  # type: ignore[arg-type]
    gripper = (
        GripperStream(arms={a: list(g) for a in arms}, rate_hz=RATE, t=list(t))
        if with_gripper
        else None
    )
    return Episode(
        source="synthetic",
        episode_id=f"{kind}-{seed}",
        session=f"s{seed}",
        task="grasp the cup",
        duration_s=float(dur),
        cameras={"scene": VideoRef(path="scene.mp4", fps=4.0, n_frames=n_frames)},
        force=stream,
        gripper=gripper,
        license="CC0",
    )


def make_pad_episode(
    seed: int,
    n_frames: int = 10,
    arms: tuple[Arm, ...] = ("L", "R"),
    with_gripper: bool = True,
) -> Episode:
    """Tiny pad-channel episode at 100 Hz: baseline 1.0, hold about 5.0 (times 1.1 for R)."""
    return _episode("pad", seed, n_frames, arms, with_gripper)


def make_wrist_episode(
    seed: int,
    n_frames: int = 10,
    arms: tuple[Arm, ...] = ("L",),
    with_gripper: bool = True,
) -> Episode:
    """Tiny wrist-channel episode; force magnitude equals the pad total of the same shape."""
    return _episode("wrist", seed, n_frames, arms, with_gripper)
