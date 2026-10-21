"""FLORA window geometry, dense label stitching and target-string construction.

Owns: window anchors, frame times, group labels, stitching to 1 Hz, outcome frame indices and
the exact target strings (grammar constants live in prompts/parse.py).
Must never: rendering force or building prompts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from tactile_flora.prompts.parse import (
    GROUP_S,
    MAX_DENSE_RATIONALE,
    MAX_OUTCOME_RATIONALE,
    N_GROUPS,
    WINDOW_PRE_S,
)
from tactile_flora.schema import Episode, Label, Span, Split, Window

N_POSITIONS = 64
FPS = 4.0
STRIDE_S = 10.0
FINAL_OFFSET_S = 9.75  # final window anchor = t_last - 9.75
KEPT_GROUPS = 24  # groups 1-24 cover [alpha - 2, alpha + 10) == the 10 s stride


def make_windows(
    episode: Episode,
    force_lines_per_anchor: Mapping[float, Sequence[str]] | None,
    merged: list[Label],
    split: Split,
    *,
    outcome_force_lines: Sequence[str] = (),
) -> list[Window]:
    """Dense windows for every anchor plus one outcome window, from merged labels.

    ``merged`` is the output of labels.merge.merge_labels. Dense windows are emitted only when
    ``merged`` holds a dense/event label (an empty one means "all normal"); an outcome window only
    when it holds an outcome label (frame times come from the scene camera). Force lines are
    rendered elsewhere: dense lines are looked up by anchor (default empty, the control
    condition), outcome lines come from ``outcome_force_lines``.
    """
    ref = f"{episode.source}/{episode.episode_id}"
    lines = force_lines_per_anchor or {}
    out: list[Window] = []
    dense = [lab for lab in merged if lab.kind in ("dense", "event")]
    if dense:
        spans = [s for lab in dense for s in lab.spans]
        for anchor in dense_anchors(episode.duration_s):
            out.append(
                Window(
                    episode_ref=ref,
                    prompt_kind="dense",
                    anchor_s=anchor,
                    frame_times=tuple(dense_frame_times(anchor)),
                    force_lines=tuple(lines.get(anchor, ())),
                    target=to_flora_dense_target(spans, anchor, episode.duration_s),
                    split=split,
                    label_origin=dense_label_origin(dense, anchor),
                )
            )
    outcome = next((lab for lab in merged if lab.kind == "outcome" and lab.outcome), None)
    if outcome is not None:
        cam = episode.cameras.get("scene")
        if cam is None:
            raise ValueError(f"{ref}: outcome label but no scene camera")
        idx = outcome_frame_indices(cam.n_frames, cam.fps)
        out.append(
            Window(
                episode_ref=ref,
                prompt_kind="outcome",
                anchor_s=0.0,
                frame_times=tuple(i / cam.fps for i in idx),
                force_lines=tuple(outcome_force_lines),
                target=to_flora_outcome_target(outcome, with_rationale=True),
                split=split,
                label_origin=outcome.origin,
            )
        )
    return out


_ORIGIN_ORDER = ("human", "source", "programmatic", "pseudo")


def dense_label_origin(dense: Sequence[Label], anchor: float) -> str:
    """Origin that produced a dense window's target.

    The origin with the most non-normal span time inside [anchor - 2, anchor + 14) (ties: higher
    precedence); with no non-normal span there, the top origin of the dense labels present.
    """
    lo, hi = anchor - WINDOW_PRE_S, anchor + 16.0 - WINDOW_PRE_S
    time: dict[str, float] = {}
    for lab in dense:
        for s in lab.spans:
            if s.quality == 3:
                continue
            ov = min(s.t1, hi) - max(s.t0, lo)
            if ov > 0:
                time[s.origin or lab.origin] = time.get(s.origin or lab.origin, 0.0) + ov
    if time:
        return min(
            time, key=lambda o: (-time[o], _ORIGIN_ORDER.index(o) if o in _ORIGIN_ORDER else 9)
        )
    present = {lab.origin for lab in dense}
    return next((o for o in _ORIGIN_ORDER if o in present), "none")


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


DENSE_PLACEHOLDER_RATIONALE = "non-normal action"
OUTCOME_PLACEHOLDER_RATIONALE = "no rationale"


def _clip_text(text: str, limit: int, placeholder: str) -> str:
    """Whitespace-collapse, truncate to ``limit`` chars; ``placeholder`` when empty."""
    out = " ".join(text.split())[:limit].rstrip()
    return out or placeholder


def to_flora_dense_target(spans: list[Span], anchor: float, duration: float) -> str:
    r"""Exact dense answer string; the grammar is defined in prompts/parse.py.

    Layout: ``Worst: w\nQuality:\n<32 digits>\nRationales:`` then one ``first-last text``
    line per maximal run of equal non-normal digits. Rationale text is whitespace-collapsed and
    truncated to 240 chars; a run with no rationale gets the deterministic placeholder
    "non-normal action" so the line still satisfies the grammar (text must be 1..240 chars).
    """
    digits = dense_group_labels(spans, anchor, duration)
    lines = ["Rationales:"]
    for first, last, d in _runs(digits):
        text = _run_rationale(spans, first, last, d, anchor, duration)
        text = _clip_text(text, MAX_DENSE_RATIONALE, DENSE_PLACEHOLDER_RATIONALE)
        lines.append(f"{first}-{last} {text}")
    return f"Worst: {min(digits)}\nQuality:\n{''.join(map(str, digits))}\n" + "\n".join(lines)


def to_flora_outcome_target(label: Label, with_rationale: bool) -> str:
    r"""Outcome answer: ``Rationale: <text>\n\nScore: <1-5>`` or ``Score: <d>`` (score-only).

    Rationale text is whitespace-collapsed, truncated to 512 chars, and "no rationale" when empty.
    """
    if label.outcome is None:
        raise ValueError("label has no outcome score")
    if not with_rationale:
        return f"Score: {label.outcome}"
    text = label.spans[0].rationale if label.spans else ""
    text = _clip_text(text, MAX_OUTCOME_RATIONALE, OUTCOME_PLACEHOLDER_RATIONALE)
    return f"Rationale: {text}\n\nScore: {label.outcome}"
