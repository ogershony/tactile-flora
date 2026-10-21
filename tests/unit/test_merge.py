from tactile_flora.labels.merge import merge_labels, resolve_frame_quality
from tactile_flora.schema import Label, Span


def dense(origin: str, *spans: Span, kind: str = "dense") -> Label:
    return Label(kind=kind, origin=origin, spans=list(spans))  # type: ignore[arg-type]


def sp(t0: float, t1: float, q: int, why: str = "") -> Span:
    return Span(t0=t0, t1=t1, quality=q, rationale=why)  # type: ignore[arg-type]


def test_precedence_across_origins() -> None:
    labels = [
        dense("pseudo", sp(0, 10, 1, "pseudo")),
        dense("programmatic", sp(2, 6, 2, "prog"), kind="event"),
        dense("source", sp(4, 8, 2, "src")),
        dense("human", sp(5, 6, 3)),
    ]
    (m,) = merge_labels(labels)
    got = [(s.t0, s.t1, s.quality, s.origin) for s in m.spans]
    assert got == [
        (0, 2, 1, "pseudo"),
        (2, 4, 2, "programmatic"),
        (4, 5, 2, "source"),
        (5, 6, 3, "human"),
        (6, 8, 2, "source"),
        (8, 10, 1, "pseudo"),
    ]
    assert m.origin == "human"


def test_severity_within_origin() -> None:
    labels = [
        dense("programmatic", sp(0, 4, 2, "slip"), sp(2, 3, 1, "crush"), kind="event"),
        dense("programmatic", sp(3, 5, 2, "slip"), kind="event"),
    ]
    (m,) = merge_labels(labels)
    got = [(s.t0, s.t1, s.quality, s.rationale) for s in m.spans]
    assert got == [(0, 2, 2, "slip"), (2, 3, 1, "crush"), (3, 5, 2, "slip")]


def test_resolve_frame_quality() -> None:
    by = {"programmatic": [sp(0, 2, 2, "a")], "source": [sp(1, 3, 1, "b")]}
    assert resolve_frame_quality(by, 0.5) == (2, "a")
    assert resolve_frame_quality(by, 1.5) == (1, "b")
    assert resolve_frame_quality(by, 3.0) == (3, "")


def test_outcome_precedence_and_empty_dense() -> None:
    labels = [
        Label(kind="outcome", origin="pseudo", outcome=1),
        Label(kind="outcome", origin="source", outcome=4),
        Label(kind="outcome", origin="programmatic", outcome=2),
    ]
    assert [(m.kind, m.outcome) for m in merge_labels(labels)] == [("outcome", 4)]
    assert merge_labels([]) == []
    assert merge_labels([dense("programmatic", kind="event")])[0].spans == []
