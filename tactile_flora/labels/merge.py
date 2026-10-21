"""Merge labels from several origins into one outcome label and one set of dense spans.

Owns: the origin precedence human > source > programmatic > pseudo, the within-frame severity rule
(quality 1 over 2 over 3) and ``resolve_frame_quality``.
Must never: detect events, read datasets, or build target strings.

Dense and event labels are merged into a single Label(kind="dense") of non-overlapping Spans, each
tagged with ``Span.origin``. At every time the spans of the highest-precedence origin covering it
are used (a quality-3 span of that origin overrides lower origins), then the most severe of those.
Adjacent pieces with identical quality, rationale and origin are coalesced. The dense label is only
emitted when some input is a dense or event label, so "no labels" stays distinguishable from "all
normal". Outcome labels: the highest-precedence origin wins; ties take the lowest score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tactile_flora.schema import Label, Span

PRECEDENCE: tuple[str, ...] = ("human", "source", "programmatic", "pseudo")


def _covers(s: Span, t: float) -> bool:
    return s.t0 <= t < s.t1


def _resolve(spans_by_origin: Mapping[str, Sequence[Span]], t: float) -> Span | None:
    for origin in PRECEDENCE:
        hits = [s for s in spans_by_origin.get(origin, ()) if _covers(s, t)]
        if hits:
            # most severe first, then a span with a rationale, then the earliest start
            return min(hits, key=lambda s: (s.quality, s.rationale == "", s.t0))
    return None


def resolve_frame_quality(
    spans_by_origin: Mapping[str, Sequence[Span]], t: float
) -> tuple[int, str]:
    """Return (quality, rationale) at time ``t``; (3, "") when no span covers it."""
    s = _resolve(spans_by_origin, t)
    return (3, "") if s is None else (s.quality, s.rationale)


def _merge_spans(labels: list[Label]) -> list[Span]:
    by_origin: dict[str, list[Span]] = {}
    for lab in labels:
        by_origin.setdefault(lab.origin, []).extend(s for s in lab.spans if s.t1 > s.t0)
    cuts = sorted({x for ss in by_origin.values() for s in ss for x in (s.t0, s.t1)})
    out: list[Span] = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        best = _resolve(by_origin, a)
        if best is None:
            continue
        origin = next(o for o in PRECEDENCE if best in by_origin.get(o, ()))
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.t1 == a
            and (prev.quality, prev.rationale, prev.origin)
            == (best.quality, best.rationale, origin)
        ):
            out[-1] = prev.model_copy(update={"t1": b})
        else:
            out.append(
                Span(t0=a, t1=b, quality=best.quality, rationale=best.rationale, origin=origin)
            )
    return out


def merge_labels(labels: list[Label]) -> list[Label]:
    """Return [outcome label?, dense label?] after applying origin precedence and severity."""
    result: list[Label] = []
    outcomes = [lab for lab in labels if lab.kind == "outcome" and lab.outcome is not None]
    if outcomes:
        for origin in PRECEDENCE:
            group = [lab for lab in outcomes if lab.origin == origin]
            if group:
                result.append(min(group, key=lambda lab: lab.outcome or 5))
                break
    dense = [lab for lab in labels if lab.kind in ("dense", "event")]
    if dense:
        top = next(o for o in PRECEDENCE if any(lab.origin == o for lab in dense))
        result.append(Label(kind="dense", origin=top, spans=_merge_spans(dense)))  # type: ignore[arg-type]
    return result
