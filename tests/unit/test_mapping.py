import pytest

from tactile_flora.labels.mapping import (
    map_native_labels,
    phantom_success_p99,
    phantom_success_stats,
    rh20t_outcome,
)
from tactile_flora.schema import ArmFeatures, Episode, ResampledForce
from tests.fixtures.synthetic import make_pad_episode


def ep(source: str, **meta: object) -> Episode:
    e = make_pad_episode(0, n_frames=40)  # duration 10.25 s
    return e.model_copy(update={"source": source, "meta": dict(meta)})


@pytest.mark.parametrize(
    ("rating", "moved", "expected"),
    [(0, False, 1), (0, True, 1), (1, False, 2), (1, True, 3)]
    + [(r, m, 5) for r in range(2, 10) for m in (False, True)],
)
def test_rh20t_rating_rule(rating: int, moved: bool, expected: int) -> None:
    assert rh20t_outcome(rating, moved) == expected


def test_rh20t_rating_out_of_range() -> None:
    with pytest.raises(ValueError):
        rh20t_outcome(10, False)


def test_rh20t_unrated_gives_no_label() -> None:
    assert rh20t_outcome(-1, False) is None
    assert map_native_labels("rh20t", ep("rh20t", rating=-1)) == []


def test_eval_only_sources_are_noops() -> None:
    for name in ("baxter", "hrdexdb", "softvtbench"):
        assert map_native_labels(name, ep(name, success=False)) == []


def test_phantom_p99_only_for_failure_demos_and_computed_per_object() -> None:
    good = ep("phantom", success=True, object="cup")
    bad = ep("phantom", success=False, object="cup")
    f = _features([0.5, 1.0, 1.5, 2.6, 2.8, 1.0])
    p99 = phantom_success_p99([(good, _features([0.0, 1.0, 2.0] * 10)), (bad, f)])
    assert set(p99) == {"cup"} and 1.9 < p99["cup"] <= 2.0
    # a successful demo never gets the failure span, even above the p99
    assert [
        x.kind for x in map_native_labels("phantom", good, features=f, phantom_p99=p99["cup"])
    ] == ["outcome"]
    kinds = [x.kind for x in map_native_labels("phantom", bad, features=f, phantom_p99=p99["cup"])]
    assert kinds == ["outcome", "dense"]


def test_rh20t_mapping() -> None:
    labels = map_native_labels("rh20t", ep("rh20t", rating=1, object_moved_toward_goal=True))
    assert [(lab.kind, lab.origin, lab.outcome) for lab in labels] == [("outcome", "source", 3)]


def _features(F: list[float]) -> ResampledForce:
    n = len(F)
    return ResampledForce(
        channel="pad",
        frame_times=[i * 0.25 for i in range(n)],
        arms={"L": ArmFeatures(F=F, dF=[0.0] * n)},
    )


def test_phantom_success_and_fail() -> None:
    ok = map_native_labels("phantom", ep("phantom", success=True, task_folder="pour"))
    assert [lab.outcome for lab in ok] == [5]
    bad = map_native_labels("phantom", ep("phantom", success=True, task_folder="pour_fail"))
    assert [lab.outcome for lab in bad] == [1]
    bad2 = map_native_labels("phantom", ep("phantom", success=False, task_folder="pour"))
    assert [lab.outcome for lab in bad2] == [1]


def test_phantom_dense_failure_span() -> None:
    e = ep("phantom", success=False, task_folder="x_fail", object="sponge")
    f = _features([0.5, 1.0, 1.5, 2.6, 2.8, 1.0])
    labels = map_native_labels("phantom", e, features=f, phantom_p99=2.5)
    dense = next(lab for lab in labels if lab.kind == "dense")
    assert dense.origin == "source"
    (s,) = dense.spans
    assert (s.t0, s.t1, s.quality) == (0.75, e.duration_s, 1)
    assert s.rationale == "excessive force on the sponge"
    # p99 from meta, nothing exceeds it: no dense label
    e2 = ep("phantom", success=True, object="cup", peak_force_p99=3.0)
    assert [lab.kind for lab in map_native_labels("phantom", e2, features=f)] == ["outcome"]


def test_reassemble() -> None:
    segs = [
        {"t0": 0.0, "t1": 1.0, "text": "reach", "success": True},
        {"t0": 1.0, "t1": 2.0, "text": "gripper misses the peg", "success": False},
    ]
    e = ep("reassemble", action_success=[True, False, True], segments=segs)
    out, dense = map_native_labels("reassemble", e)
    assert out.outcome == 1
    assert (dense.kind, dense.origin) == ("dense", "source")
    assert (dense.spans[0].quality, dense.spans[0].rationale) == (1, "gripper misses the peg")
    e2 = ep("reassemble", action_success=[True, True])
    assert [lab.outcome for lab in map_native_labels("reassemble", e2)] == [5]


def test_airoa() -> None:
    segs = [{"t0": 3.0, "t1": 4.0, "cause": "object slipped"}]
    e = ep("airoa", success=False, failure_cause="dropped the bottle", failed_segments=segs)
    out, dense = map_native_labels("airoa", e)
    assert out.outcome == 1 and out.spans[0].rationale == "dropped the bottle"
    assert dense.spans[0].rationale == "object slipped" and dense.spans[0].quality == 1
    ok = map_native_labels("airoa", ep("airoa", success=True, failed_segments=[]))
    assert ok[0].outcome == 5 and ok[1].spans == []


def test_sharpadex_none_and_unknown() -> None:
    assert map_native_labels("sharpadex", ep("sharpadex")) == []
    with pytest.raises(ValueError):
        map_native_labels("nope", ep("nope"))


def _contact_features(F: list[float], tau: float = 0.2) -> ResampledForce:
    f = _features(F)
    for af in f.arms.values():
        af.c = [int(v > tau) for v in af.F]
    return f


def test_phantom_success_stats_peak_p5() -> None:
    good = [ep("phantom", success=True, object="cup") for _ in range(3)]
    feats = [_features([0.0, 0.5, p, 0.1]) for p in (0.9, 1.0, 1.2)]
    bad = ep("phantom", success=False, object="cup")
    p99, peak_p5 = phantom_success_stats(
        [*zip(good, feats, strict=True), (bad, _features([5.0] * 4))]
    )
    assert set(p99) == {"cup"} and set(peak_p5) == {"cup"}
    assert 0.9 <= peak_p5["cup"] < 1.0  # 5th percentile of the success peaks {0.9, 1.0, 1.2}
    assert phantom_success_p99([*zip(good, feats, strict=True)]) == p99


def test_phantom_under_grasp_span_from_first_contact() -> None:
    e = ep("phantom", success=False, task_folder="whiteboard_fail", object="whiteboard")
    weak = _contact_features([0.0, 0.1, 0.3, 0.4, 0.3, 0.0])
    labels = map_native_labels("phantom", e, features=weak, phantom_p99=2.5, phantom_peak_p5=0.8)
    out = next(lab for lab in labels if lab.kind == "outcome")
    dense = next(lab for lab in labels if lab.kind == "dense")
    (s,) = dense.spans
    assert (s.t0, s.t1, s.quality) == (0.5, e.duration_s, 1)  # first c == 1 is the third frame
    assert s.rationale == "insufficient grip force on the whiteboard"
    assert out.outcome == 1 and out.spans[0].rationale == s.rationale
    # a weak failure that never made contact spans the whole episode
    none = _features([0.0, 0.1, 0.1, 0.0])
    labels = map_native_labels("phantom", e, features=none, phantom_p99=2.5, phantom_peak_p5=0.8)
    assert next(lab for lab in labels if lab.kind == "dense").spans[0].t0 == 0.0
    # a successful demo never gets the span, however weak
    good = ep("phantom", success=True, object="whiteboard")
    labels = map_native_labels("phantom", good, features=weak, phantom_p99=2.5, phantom_peak_p5=0.8)
    assert [lab.kind for lab in labels] == ["outcome"] and labels[0].spans == []


def test_phantom_crush_rule_wins_and_middle_peaks_get_no_span() -> None:
    e = ep("phantom", success=False, task_folder="x_fail", object="sponge")
    crush = _contact_features([0.5, 1.0, 1.5, 2.6, 2.8, 1.0])
    labels = map_native_labels("phantom", e, features=crush, phantom_p99=2.5, phantom_peak_p5=0.8)
    out, dense = labels
    assert dense.spans[0].t0 == 0.75 and dense.spans[0].rationale == "excessive force on the sponge"
    assert out.spans[0].rationale == "excessive force on the sponge"
    # peak between the under-grasp and crush thresholds: outcome 1 without a span
    mid = _contact_features([0.5, 1.0, 1.5, 1.2, 0.0])
    labels = map_native_labels("phantom", e, features=mid, phantom_p99=2.5, phantom_peak_p5=0.8)
    assert [lab.kind for lab in labels] == ["outcome"] and labels[0].outcome == 1
    # without the success-set peak statistic the under-grasp rule is off
    weak = _contact_features([0.0, 0.1, 0.3, 0.0])
    labels = map_native_labels("phantom", e, features=weak, phantom_p99=2.5)
    assert [lab.kind for lab in labels] == ["outcome"]
