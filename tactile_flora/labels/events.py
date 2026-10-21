"""Programmatic force-event detectors (grasp_acquired, slip, contact_loss, regrasp, crush, off_axis).

Owns: pure span detectors over NORMALIZED features (a ResampledForce after normalize.apply) at the
4 Hz frame clock, and ``detect_all`` which packages them as programmatic event Labels.
Must never: read raw datasets, run on raw-unit features, or hold thresholds (they come from
EventsConfig).

Time-to-frame mapping: the frame period dt is the median difference of ``frame_times`` (0.25 s at
4 Hz). A duration D "at least" maps to ceil(D / dt) frames; "over" D maps to floor(D / dt) + 1
frames, so at 4 Hz "at least 0.1 s" is 1 frame and "over 0.3 s" is 2 frames. A frame i covers the
span [t_i, t_i + dt). A "held" state starts at each grasp_acquired frame and ends at the first frame
where g <= grasp_g or the contact flag c is 0 (``held_mask``). Detectors that need a gripper
signal return no spans when ``g`` is None; off_axis needs ``delta`` (pad channel only). NaN frames
are unknown and never trigger a detector.
"""

from __future__ import annotations

import math

import numpy as np

from tactile_flora.config import EventsConfig
from tactile_flora.schema import Arm, ArmFeatures, Label, ResampledForce, Span

_ARM_WORD: dict[Arm, str] = {"L": "left", "R": "right"}
_EPS = 1e-9


def _dt(features: ResampledForce) -> float:
    t = np.asarray(features.frame_times, dtype=np.float64)
    return float(np.median(np.diff(t))) if t.size > 1 else 0.25


def _frames_at_least(duration_s: float, dt: float) -> int:
    return max(1, math.ceil(duration_s / dt - _EPS))


def _frames_over(duration_s: float, dt: float) -> int:
    return math.floor(duration_s / dt + _EPS) + 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as (first, last) inclusive frame indices."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _span(features: ResampledForce, dt: float, first: int, last: int, q: int, why: str) -> Span:
    t = features.frame_times
    return Span(t0=t[first], t1=t[last] + dt, quality=q, rationale=why)  # type: ignore[arg-type]


def _cols(af: ArmFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return (
        np.asarray(af.F, dtype=np.float64),
        np.asarray(af.dF, dtype=np.float64),
        None if af.g is None else np.asarray(af.g, dtype=np.float64),
    )


def _grasp_frames(F: np.ndarray, g: np.ndarray, dt: float, cfg: EventsConfig) -> list[int]:
    w = math.floor(cfg.grasp_window_s / dt + _EPS)
    out: list[int] = []
    for i in range(1, len(g)):
        if not (g[i] > cfg.grasp_g and g[i - 1] <= cfg.grasp_g):
            continue
        for j in range(i, min(i + w, len(g) - 1) + 1):
            if g[j] > cfg.grasp_g and F[j] > cfg.grasp_f:
                out.append(j)
                break
    return out


def detect_grasp_acquired(features: ResampledForce, arm: Arm, cfg: EventsConfig) -> list[float]:
    """Times (s) at which a grasp is acquired. Not a label.

    Physical rationale: a real grasp is the gripper closing onto an object, so g crosses above
    grasp_g and the fingertip load F' climbs above grasp_f within grasp_window_s. Closing on air
    never loads the pads, and loading without closure is a bump, so both must occur. The event
    time is the first frame with both conditions.
    """
    af = features.arms[arm]
    F, _, g = _cols(af)
    if g is None:
        return []
    return [features.frame_times[j] for j in _grasp_frames(F, g, _dt(features), cfg)]


def held_mask(features: ResampledForce, arm: Arm, cfg: EventsConfig) -> np.ndarray:
    """Boolean per frame: True from each grasp_acquired frame until g <= grasp_g or c == 0."""
    af = features.arms[arm]
    F, _, g = _cols(af)
    n = len(F)
    held = np.zeros(n, dtype=bool)
    if g is None:
        return held
    starts = set(_grasp_frames(F, g, _dt(features), cfg))
    c = af.c if af.c is not None else [1] * n
    on = False
    for i in range(n):
        if i in starts:
            on = True
        elif on and (not g[i] > cfg.grasp_g or c[i] == 0):
            on = False
        held[i] = on
    return held


def detect_slip(
    features: ResampledForce, arm: Arm, cfg: EventsConfig, *, task_complete: bool = False
) -> list[Span]:
    """Slip: the load falls fast while the gripper has not moved, so the object slides out.

    dF' below slip_df for at least slip_min_s while g is constant within slip_g_tol (over the
    frame before, at and after) and the object was held on the previous frame. Quality 2.
    """
    af = features.arms[arm]
    F, dF, g = _cols(af)
    if g is None:
        return []
    dt = _dt(features)
    held = held_mask(features, arm, cfg)
    n = len(F)
    mask = np.zeros(n, dtype=bool)
    for i in range(1, n):
        nxt = g[i + 1] if i + 1 < n else g[i]
        steady = abs(g[i] - g[i - 1]) <= cfg.slip_g_tol and abs(nxt - g[i]) <= cfg.slip_g_tol
        mask[i] = bool(dF[i] < cfg.slip_df and steady and held[i - 1])
    need = _frames_at_least(cfg.slip_min_s, dt)
    why = f"the object slips in the {_ARM_WORD[arm]} gripper"
    return [_span(features, dt, a, b, 2, why) for a, b in _runs(mask) if b - a + 1 >= need]


def detect_contact_loss(
    features: ResampledForce, arm: Arm, cfg: EventsConfig, *, task_complete: bool = False
) -> list[Span]:
    """Contact loss: the pads unload (c == 0, F' below tau') while the gripper stays closed.

    A closed gripper with no load means the object left it. Needs F' below tau' (c == 0) for over
    contact_loss_s, g > grasp_g throughout, and the object held just before the run. Quality 1 when
    the task was not complete (a failure), 2 when it was (a drop that did not end the task).
    """
    af = features.arms[arm]
    F, _, g = _cols(af)
    if g is None or af.c is None:
        return []
    dt = _dt(features)
    held = held_mask(features, arm, cfg)
    c = np.asarray(af.c)
    mask = (c == 0) & (g > cfg.grasp_g) & np.isfinite(F)
    need = _frames_over(cfg.contact_loss_s, dt)
    q = 2 if task_complete else 1
    why = f"the object drops from the {_ARM_WORD[arm]} gripper"
    return [
        _span(features, dt, a, b, q, why)
        for a, b in _runs(mask)
        if b - a + 1 >= need and a > 0 and held[a - 1]
    ]


def detect_regrasp(
    features: ResampledForce, arm: Arm, cfg: EventsConfig, *, task_complete: bool = False
) -> list[Span]:
    """Regrasp: the object was lost and picked up again, a recoverable but inefficient motion.

    Two grasp_acquired events within regrasp_window_s with a contact_loss starting between them.
    The span runs from the contact loss to the second grasp. Quality 2.
    """
    dt = _dt(features)
    grasps = detect_grasp_acquired(features, arm, cfg)
    losses = detect_contact_loss(features, arm, cfg, task_complete=task_complete)
    out: list[Span] = []
    for a, b in zip(grasps, grasps[1:], strict=False):
        if b - a > cfg.regrasp_window_s + _EPS:
            continue
        for loss in losses:
            if a <= loss.t0 < b:
                out.append(Span(t0=loss.t0, t1=b + dt, quality=2, rationale="re-grasps the object"))
                break
    return out


def detect_crush(
    features: ResampledForce, arm: Arm, cfg: EventsConfig, *, task_complete: bool = False
) -> list[Span]:
    """Crush: load far above the median grasp while the gripper is still squeezing harder.

    F' above crush_f while g increases frame over frame (the controller keeps closing on a
    compliant object). Quality 1.
    """
    af = features.arms[arm]
    F, _, g = _cols(af)
    if g is None:
        return []
    dt = _dt(features)
    mask = np.zeros(len(F), dtype=bool)
    mask[1:] = (F[1:] > cfg.crush_f) & (np.diff(g) > 0)
    return [_span(features, dt, a, b, 1, "excessive force on the object") for a, b in _runs(mask)]


def detect_off_axis(
    features: ResampledForce, arm: Arm, cfg: EventsConfig, *, task_complete: bool = False
) -> list[Span]:
    """Off-axis: the two pads carry very different loads, so the object is pinched off-center.

    |Delta'| above off_axis_delta while held. Pad channel only (wrist has no Delta). Quality 2.
    """
    af = features.arms[arm]
    if features.channel != "pad" or af.delta is None or af.g is None:
        return []
    d = np.abs(np.asarray(af.delta, dtype=np.float64))
    mask = (d > cfg.off_axis_delta) & held_mask(features, arm, cfg)
    return [
        _span(features, _dt(features), a, b, 2, "off-center grasp") for a, b in _runs(mask)
    ]


def detect_all(
    features: ResampledForce, cfg: EventsConfig, *, task_complete: bool = False
) -> list[Label]:
    """One programmatic event Label per arm holding every detector's spans (sorted by t0)."""
    labels: list[Label] = []
    for arm in sorted(features.arms):
        spans: list[Span] = []
        for det in (detect_slip, detect_contact_loss, detect_regrasp, detect_crush, detect_off_axis):
            spans.extend(det(features, arm, cfg, task_complete=task_complete))
        spans.sort(key=lambda s: (s.t0, s.t1))
        labels.append(Label(kind="event", origin="programmatic", spans=spans))
    return labels
