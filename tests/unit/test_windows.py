import pytest

from tactile_flora.labels.windows import (
    dense_anchors,
    dense_frame_times,
    dense_group_labels,
    group_of_position,
    outcome_frame_indices,
    stitch_dense,
    to_1hz,
    to_flora_outcome_target,
)
from tactile_flora.schema import Label, Span


def test_anchor_schedule() -> None:
    assert dense_anchors(35.0) == [0, 10, 20, 25.25]
    assert dense_anchors(20.0) == [0, 10, 10.25]
    assert dense_anchors(9.0) == [0.0]
    assert dense_anchors(9.75) == [0.0]
    assert dense_anchors(9.75 + 10) == [0, 10.0]


def test_frame_times_and_groups() -> None:
    t = dense_frame_times(10.0)
    assert len(t) == 64 and t[0] == 8.0 and t[-1] == 23.75
    assert [group_of_position(i) for i in (1, 2, 3, 4, 63, 64)] == [1, 1, 2, 2, 32, 32]
    with pytest.raises(ValueError):
        group_of_position(0)


def test_group_labels_severity_and_padding() -> None:
    # anchor 0: frames at -2.0..13.75; clamped to 0 before start
    spans = [Span(t0=0.0, t1=1.0, quality=2), Span(t0=0.5, t1=0.75, quality=1)]
    d = dense_group_labels(spans, 0.0, 100.0)
    # groups 1-4 cover t<0 -> clamp to 0 -> boundary label 2; group 5 (t 0..0.25) is 2;
    # group 6 (0.5, 0.75) : frame 0.5 is 1
    assert d[:4] == [2, 2, 2, 2]
    assert d[4] == 2 and d[5] == 1
    assert d[6] == 3 or d[6] == 2  # frames 1.0, 1.25: outside span -> 3
    assert d[6] == 3
    assert len(d) == 32


def test_end_padding_takes_boundary() -> None:
    spans = [Span(t0=5.0, t1=8.0, quality=1)]
    d = dense_group_labels(spans, 0.0, 8.0)
    assert d[-1] == 1  # padded past the end copies the last frame


def test_stitching_drops_tail() -> None:
    w0 = (0.0, [1] * 32)
    w1 = (10.0, [2] * 32)
    s = stitch_dense([w0, w1])
    assert len(s) == 24 + 28  # final window keeps groups with centre >= 10 s
    assert [d for _, d in s[:24]] == [1] * 24
    assert s[24][0] == 10.0 and s[24][1] == 2


def test_stitch_final_offgrid() -> None:
    w = [(0.0, [3] * 32), (10.0, [3] * 32), (15.25, [1] * 32)]
    s = stitch_dense(w)
    times = [t for t, _ in s]
    assert times == sorted(times)
    one = to_1hz(s)
    assert [b for b, _ in one] == list(range(0, 30))
    assert one[-1][1] == 1


def test_1hz_pools_severe() -> None:
    assert to_1hz([(-0.5, 1), (0.0, 3), (0.5, 2), (1.0, 3), (1.5, 3)]) == [(0, 2), (1, 3)]


def test_outcome_indices() -> None:
    idx = outcome_frame_indices(120, 30.0)
    assert len(idx) == 40 and idx[0] == 0 and idx[-1] == 119
    assert outcome_frame_indices(10, 30.0) == [0, 9]
    assert len(outcome_frame_indices(10000, 30.0)) == 64
    assert len(outcome_frame_indices(1, 30.0)) == 2


def test_outcome_target() -> None:
    lab = Label(kind="outcome", origin="human", outcome=4)
    assert to_flora_outcome_target(lab, False) == "Score: 4"
    assert to_flora_outcome_target(lab, True) == "Rationale: no rationale\n\nScore: 4"
