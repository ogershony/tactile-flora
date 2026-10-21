"""Resample a ForceStream onto frame times (median F, D, dF) plus raw slices per 0.5 s group.

Owns: the per-frame raw-unit features and the raw 100 Hz slice per 0.5 s group. Contact flags are
not computed here because tau comes from normalization (see normalize.apply).
Must never: normalize, render text, or read datasets.

A frame whose window holds no samples gets NaN F, Delta and dF; downstream code renders those
as 0.00 with c=0. dF is (F_t - F_{t-P}) / P with P = cfg.frame_period_s (the spec constant, never
the actual spacing of the frame list) and F_{t-P} the raw median in [t-P-hw, t-P+hw]; if no raw
sample lies there dF=0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tactile_flora.config import ResampleConfig
from tactile_flora.schema import (
    Arm,
    ArmFeatures,
    ForceStream,
    GripperStream,
    PadArmForce,
    ResampledForce,
    WristArmForce,
)


def _scalar_and_delta(data: PadArmForce | WristArmForce) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(data, PadArmForce):
        a, b = data.f_a_array, data.f_b_array
        return a + b, a - b
    return np.linalg.norm(data.wrench[:, :3], axis=1), None


def _median(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def _nearest(t: np.ndarray, values: np.ndarray, query: float) -> float:
    i = int(np.searchsorted(t, query))
    if i > 0 and (i == len(t) or abs(t[i - 1] - query) <= abs(t[i] - query)):
        i -= 1
    return float(values[i])


def resample(
    stream: ForceStream,
    gripper: GripperStream | None,
    frame_times: Sequence[float],
    cfg: ResampleConfig,
) -> ResampledForce:
    """Return raw-unit per-frame features per arm for ``frame_times``.

    Group k (0-based) covers 0-based frame positions 2k and 2k+1 (those that exist), so every
    frame is in exactly one group and 64 frames give 32 groups; its raw slice spans
    [min t - half_window, max t + half_window] of those frames.
    """
    t_raw = stream.t_array
    hw = cfg.half_window_s
    times = [float(t) for t in frame_times]
    n = len(times)
    arms: dict[Arm, ArmFeatures] = {}
    for arm, data in stream.arms.items():
        scalar, diff = _scalar_and_delta(data)
        lo = np.searchsorted(t_raw, np.asarray(times) - hw, side="left")
        hi = np.searchsorted(t_raw, np.asarray(times) + hw, side="right")
        F = [_median(scalar[a:b]) for a, b in zip(lo, hi, strict=True)]
        delta = None if diff is None else [_median(diff[a:b]) for a, b in zip(lo, hi, strict=True)]
        period = cfg.frame_period_s
        lo_p = np.searchsorted(t_raw, np.asarray(times) - period - hw, side="left")
        hi_p = np.searchsorted(t_raw, np.asarray(times) - period + hw, side="right")
        dF: list[float] = []
        for i in range(n):
            if hi_p[i] <= lo_p[i]:
                dF.append(0.0)
            else:
                dF.append((F[i] - _median(scalar[lo_p[i] : hi_p[i]])) / period)
        g: list[float] | None = None
        if gripper is not None and arm in gripper.arms and gripper.t:
            gt = np.asarray(gripper.t, dtype=np.float64)
            gv = np.asarray(gripper.arms[arm], dtype=np.float64)
            g = [_nearest(gt, gv, t) for t in times]
        groups: list[list[float]] = []
        for k in range((n + 1) // 2):
            pos = [p for p in (2 * k, 2 * k + 1) if p < n]
            a = int(np.searchsorted(t_raw, times[pos[0]] - hw, side="left"))
            b = int(np.searchsorted(t_raw, times[pos[-1]] + hw, side="right"))
            groups.append([float(v) for v in scalar[a:b]])
        arms[arm] = ArmFeatures(F=F, delta=delta, dF=dF, g=g, raw_groups=groups)
    return ResampledForce(channel=stream.channel, frame_times=times, arms=arms)
