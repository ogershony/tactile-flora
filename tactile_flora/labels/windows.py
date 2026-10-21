"""FLORA window geometry, dense label stitching and target-string construction.

Owns: window anchors, frame times, group labels, stitching to 1 Hz, outcome frame indices and
the exact target strings (grammar constants live in prompts/parse.py).
Must never: rendering force or building prompts.
"""

from __future__ import annotations

import math

from tactile_flora.prompts.parse import GROUP_S, N_GROUPS, WINDOW_PRE_S
from tactile_flora.schema import Episode, Label, Span, Window

N_POSITIONS = 64
FPS = 4.0
STRIDE_S = 10.0
FINAL_OFFSET_S = 9.75  # final window anchor = t_last - 9.75
KEPT_GROUPS = 24  # groups 1-24 cover [alpha - 2, alpha + 10) == the 10 s stride


def make_windows(episode: Episode) -> list[Window]:
    """Public entry point of this module (needs force rendering; wired in a later milestone)."""
    raise NotImplementedError("milestone 4")


def dense_anchors(t_last: float) -> list[float]:
    """Anchors 10k while 10k < t_last - 9.75, then one final window at t_last - 9.75.

    When t_last - 9.75 <= 0 (episode shorter than one window's tail) a single window at
    anchor 0 is returned; the padded tail takes the boundary frame's label.
    """
    bound = t_last - FINAL_OFFSET_S
    if bound <= 0:
        return [0.0]
    anchors: list[float] = []
    k = 0
    while STRIDE_S * k < bound:
        anchors.append(STRIDE_S * k)
        k += 1
    anchors.append(bound)
    return anchors


def dense_frame_times(anchor: float) -> list[float]:
    """64 times alpha + (i - 9)/4 for i = 1..64 (alpha - 2.0 .. alpha + 13.75)."""
    return [anchor + (i - 9) / FPS for i in range(1, N_POSITIONS + 1)]


def group_of_position(i: int) -> int:
    """Group (1..32) of 1-based position i: group g covers positions 2g-1 and 2g."""
    if not 1 <= i <= N_POSITIONS:
        raise ValueError("position must be in 1..64")
    return (i + 1) // 2


def _frame_label(spans: list[Span], t: float, duration: float) -> int:
    t = min(max(t, 0.0), duration)
    worst = 3
    for s in spans:
        inside = s.t0 <= t < s.t1 or (t >= duration and s.t0 <= t <= s.t1)
        if inside:
            worst = min(worst, s.quality)
    return worst


def dense_group_labels(spans: list[Span], anchor: float, episode_duration: float) -> list[int]:
    """32 group digits: worst quality over covering spans per frame, worst of 2 frames per group.

    Frame times are clamped to [0, duration] so padded positions copy the boundary label.
    """
    times = dense_frame_times(anchor)
    frame = [_frame_label(spans, t, episode_duration) for t in times]
    return [min(frame[2 * g], frame[2 * g + 1]) for g in range(N_GROUPS)]


def stitch_dense(windows: list[tuple[float, list[int]]]) -> list[tuple[float, int]]:
    """Concatenate windows into (group_start_time_s, digit) pairs, one per 0.5 s group.

    Windows are ordered by anchor. Every non-final window keeps groups 1-24 (its 10 s stride).
    The final window's anchor is generally not on the grid, so it overlaps its predecessor; it
    keeps only groups whose centre is at or after the previous window's kept end
    (prev anchor + 10). A single window keeps all 32 groups.
    """
    out: list[tuple[float, int]] = []
    ordered = sorted(windows, key=lambda w: w[0])
    for n, (anchor, digits) in enumerate(ordered):
        if len(digits) != N_GROUPS:
            raise ValueError("each window needs 32 digits")
        final = n == len(ordered) - 1
        prev_end = ordered[n - 1][0] + STRIDE_S if n > 0 else -math.inf
        last_g = N_GROUPS if final else KEPT_GROUPS
        for g in range(last_g):
            start = anchor - WINDOW_PRE_S + GROUP_S * g
            if start + GROUP_S / 2 >= prev_end - 1e-9:
                out.append((start, digits[g]))
    return out


def to_1hz(stitched: list[tuple[float, int]]) -> list[tuple[int, int]]:
    """Pool stitched groups to 1 Hz: bin = floor(group centre), digit = most severe in the bin.

    Bins before 0 s (window padding) are dropped. Metrics are computed on this output.
    """
    bins: dict[int, int] = {}
    for start, d in stitched:
        b = math.floor(start + GROUP_S / 2 + 1e-9)
        if b < 0:
            continue
        bins[b] = min(bins.get(b, 3), d)
    return sorted(bins.items())


def outcome_frame_indices(n_frames: int, fps: float) -> list[int]:
    """M = min(64, max(2, floor(10 N/fps))) rounded down to even; index round(i (N-1)/(m-1)).

    Rounding is half-up. First and last frames are always included.
    """
    if n_frames < 1 or fps <= 0:
        raise ValueError("need n_frames >= 1 and fps > 0")
    m = min(N_POSITIONS, max(2, math.floor(10 * n_frames / fps + 1e-9)))
    m -= m % 2
    return [math.floor(i * (n_frames - 1) / (m - 1) + 0.5) for i in range(m)]


def _runs(digits: list[int]) -> list[tuple[int, int, int]]:
    """Maximal runs of equal non-normal digits as (first, last, digit), 1-based groups."""
    out: list[tuple[int, int, int]] = []
    g = 0
    while g < len(digits):
        h = g
        while h + 1 < len(digits) and digits[h + 1] == digits[g]:
            h += 1
        if digits[g] != 3:
            out.append((g + 1, h + 1, digits[g]))
        g = h + 1
    return out


def _run_rationale(
    spans: list[Span], first: int, last: int, digit: int, anchor: float, duration: float
) -> str:
    """Rationale of the span with the run's quality that overlaps the run most (tie: earliest t0).

    Deterministic stand-in for source priority, which Span does not carry. Empty if none.
    """
    ts = min(max(anchor - WINDOW_PRE_S + GROUP_S * (first - 1), 0.0), duration)
    te = min(max(anchor - WINDOW_PRE_S + GROUP_S * last, 0.0), duration)
    best: tuple[float, float, str] | None = None
    for s in spans:
        if s.quality != digit:
            continue
        overlap = min(s.t1, te) - max(s.t0, ts)
        if overlap <= 0 and not (te == ts and s.t0 <= ts <= s.t1):
            continue
        key = (-overlap, s.t0)
        if best is None or key < best[:2]:
            best = (key[0], key[1], s.rationale)
    return " ".join(best[2].split()) if best else ""


def to_flora_dense_target(spans: list[Span], anchor: float, duration: float) -> str:
    """Exact dense answer string; the grammar is defined in prompts/parse.py.

    One rationale line per maximal run of equal non-normal digits.
    """
    digits = dense_group_labels(spans, anchor, duration)
    lines = ["Rationales:"]
    for first, last, d in _runs(digits):
        text = _run_rationale(spans, first, last, d, anchor, duration)
        lines.append(f"{first}-{last} {text}" if text else f"{first}-{last}")
    return f"Worst: {min(digits)}\nQuality: {''.join(map(str, digits))}\n" + "\n".join(lines)


def to_flora_outcome_target(label: Label, with_rationale: bool) -> str:
    r"""Outcome answer: ``Rationale: <text>\nScore: <1-5>`` or ``Score: <d>`` (score-only)."""
    if label.outcome is None:
        raise ValueError("label has no outcome score")
    if not with_rationale:
        return f"Score: {label.outcome}"
    text = ""
    if label.spans:
        text = " ".join(label.spans[0].rationale.split())
    return f"Rationale: {text}\nScore: {label.outcome}"
