from pathlib import Path

import pytest

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.schema import ArmFeatures, ResampledForce
from tests.fixtures.synthetic import make_pad_episode

CFG = load_config("configs/force.yaml", ForceConfig)
TIMES = [i * 0.25 for i in range(10)]


def _feats(seed: int) -> ResampledForce:
    ep = make_pad_episode(seed)
    return resample(ep.force, ep.gripper, TIMES, CFG.resample)


def test_fit_apply() -> None:
    raw = [_feats(s) for s in range(3)]
    consts = normalize.fit(raw, CFG.resample, source="synthetic")
    n = consts.arms["L"]
    assert abs(n.baseline - 1.0) < 0.1 and abs(n.grasp - 5.0) < 0.3
    assert n.tau == pytest.approx(n.baseline + 3 * n.sigma)
    out = normalize.apply(raw[0], consts, CFG.render)
    L = out.arms["L"]
    assert L.c is not None and L.c[0] == 0 and L.c[-4] == 1
    assert 0.9 < max(L.F) <= 3.0
    assert out.norm_version == consts.version()


def test_roundtrip(tmp_path: Path) -> None:
    consts = normalize.fit([_feats(1)], CFG.resample, source="synthetic")
    p = tmp_path / "v" / "norm.json"
    normalize.save(consts, p)
    back = normalize.load(p)
    assert back == consts and back.version() == consts.version()
    p.write_text(p.read_text().replace('"source": "synthetic"', '"source": "x"'))
    with pytest.raises(normalize.NormError):
        normalize.load(p)


def test_degenerate_grasp_raises() -> None:
    flat = ResampledForce(
        channel="pad",
        frame_times=[0.0, 0.25, 1.0],
        arms={"L": ArmFeatures(F=[1.0, 1.0, 1.0], delta=[0, 0, 0], dF=[0, 0, 0], g=[0, 0, 1.0])},
    )
    with pytest.raises(normalize.NormError):
        normalize.fit([flat], CFG.resample, source="s")


def test_nan_frame_contact_zero() -> None:
    consts = normalize.fit([_feats(0)], CFG.resample, source="synthetic")
    f = ResampledForce(
        channel="pad",
        frame_times=[0.0],
        arms={"L": ArmFeatures(F=[float("nan")], delta=[float("nan")], dF=[0.0], g=[0.0])},
    )
    assert normalize.apply(f, consts, CFG.render).arms["L"].c == [0]


def test_sigma_from_raw_samples_median_over_episodes() -> None:
    raw = [_feats(s) for s in range(3)]
    n = normalize.fit(raw, CFG.resample, source="synthetic").arms["L"]
    # fixture noise on the pad sum is 0.02 at 100 Hz; frame-median std would be about 0.002
    assert 0.015 < n.sigma < 0.03
    # an outlier episode with a shifted baseline and huge noise does not move the pooled values
    odd = _feats(9)
    odd.arms["L"].raw_groups[0] = [
        float(v) * 3 + (i % 2) * 5 for i, v in enumerate(odd.arms["L"].raw_groups[0])
    ]
    m = normalize.fit([*raw, odd], CFG.resample, source="synthetic").arms["L"]
    assert abs(m.sigma - n.sigma) < 0.01 and abs(m.baseline - n.baseline) < 0.1


def test_free_space_requires_open_gripper_and_clip_option() -> None:
    f = _feats(0)
    closed = f.model_copy(deep=True)
    for af in closed.arms.values():
        assert af.g is not None
        af.g[0] = 1.0
    consts = normalize.fit([f], CFG.resample, source="s")
    with pytest.raises(normalize.NormError):
        normalize.fit([closed], CFG.resample, source="s")
    raw = normalize.apply(f, consts, CFG.render, clip=False).arms["L"]
    clipped = normalize.apply(f, consts, CFG.render).arms["L"]
    assert max(clipped.F) <= 3.0 and min(clipped.F) >= 0.0
    assert any(a != b for a, b in zip(raw.F, clipped.F, strict=True))


def test_contact_floor_raises_tau_and_refits_grasp() -> None:
    raw = [_feats(s) for s in range(3)]
    plain = normalize.fit(raw, CFG.resample, source="synthetic").arms["L"]
    # A floor below the sigma rule changes nothing.
    low = CFG.resample.model_copy(update={"contact_min_frac": 0.001})
    assert normalize.fit(raw, low, source="synthetic").arms["L"] == plain
    high = CFG.resample.model_copy(update={"contact_min_frac": 0.5})
    floored = normalize.fit(raw, high, source="synthetic").arms["L"]
    assert floored.baseline == plain.baseline
    assert floored.tau > plain.tau
    assert floored.tau == pytest.approx(floored.baseline + 0.5 * (plain.grasp - plain.baseline))
    assert floored.tau == pytest.approx(
        floored.baseline + CFG.resample.contact_sigma * floored.sigma
    )
    assert floored.grasp >= plain.grasp
    assert floored.scale == pytest.approx(floored.grasp - floored.baseline)


def _scaled(seed: int, force_gain: float = 1.0, g_gain: float = 1.0) -> ResampledForce:
    """Fixture features with the force excursion above baseline and the closure scaled."""
    f = _feats(seed).model_copy(deep=True)
    for af in f.arms.values():
        af.F = [1.0 + (v - 1.0) * force_gain for v in af.F]
        af.raw_groups = [[1.0 + (v - 1.0) * force_gain for v in grp] for grp in af.raw_groups]
        if af.g is not None:
            af.g = [v * g_gain for v in af.g]
    return f


def test_single_group_fit_is_group_all_and_applies_to_any_group() -> None:
    raw = [_feats(s) for s in range(3)]
    bare = normalize.fit(raw, CFG.resample, source="synthetic")
    paired = normalize.fit([("all", r) for r in raw], CFG.resample, source="synthetic")
    assert bare == paired and bare.version() == paired.version()
    assert set(bare.groups) == {"all"} and bare.arms is bare.groups["all"]
    n = bare.arms["L"]
    assert n.grasp_rule == "threshold" and n.grasp_g == 0.6
    # a single-group fit applies to every episode of the source, whatever its group
    assert normalize.apply(raw[0], bare, CFG.render, group="Carton") == normalize.apply(
        raw[0], bare, CFG.render
    )


def test_grouped_fit_apply_and_version(tmp_path: Path) -> None:
    light = [("light", _scaled(s)) for s in range(2)]
    heavy = [("heavy", _scaled(s, force_gain=2.0)) for s in range(2, 4)]
    consts = normalize.fit([*light, *heavy], CFG.resample, source="phantom")
    assert set(consts.groups) == {"light", "heavy"}
    assert consts.groups["heavy"]["L"].grasp == pytest.approx(
        1.0 + 2 * (consts.groups["light"]["L"].grasp - 1.0), rel=0.05
    )
    a = normalize.apply(heavy[0][1], consts, CFG.render, group="heavy").arms["L"]
    b = normalize.apply(heavy[0][1], consts, CFG.render, group="light").arms["L"]
    assert max(a.F) == pytest.approx(1.0, abs=0.15) and max(b.F) == pytest.approx(2.0, abs=0.3)
    with pytest.raises(normalize.NormError):
        normalize.apply(heavy[0][1], consts, CFG.render, group="unknown")
    with pytest.raises(normalize.NormError):
        _ = consts.arms  # no "all" group in a grouped fit
    # the version hashes the whole mapping: dropping a group changes it
    one = consts.model_copy(update={"groups": {"light": consts.groups["light"]}})
    assert one.version() != consts.version()
    p = tmp_path / "norm.json"
    normalize.save(consts, p)
    assert normalize.load(p) == consts
    with pytest.raises(normalize.NormError):
        normalize.fit([("a", _feats(0)), ("b", (_feats(1), _feats(2)))], CFG.resample, source="s")  # type: ignore[list-item]


def test_grasp_threshold_is_configurable_and_falls_back_to_p75() -> None:
    # closure peaks at 0.5: no frame passes g > 0.6, so the fit takes the most-closed quarter
    raw = [_scaled(s, g_gain=0.5) for s in range(3)]
    consts = normalize.fit(raw, CFG.resample, source="s")
    n = consts.arms["L"]
    assert n.grasp_rule == "p75" and n.grasp_g == pytest.approx(0.5)
    assert abs(n.grasp - 5.0) < 0.3 and abs(n.baseline - 1.0) < 0.1
    assert normalize.grasp_mask([0.49, 0.5, 0.6], n).tolist() == [False, True, True]
    # a lower configured cut keeps the threshold rule
    low = normalize.fit(raw, CFG.resample, source="s", grasp_g=0.4).arms["L"]
    assert low.grasp_rule == "threshold" and low.grasp_g == 0.4
    assert abs(low.grasp - n.grasp) < 0.3
    assert normalize.grasp_mask([0.4, 0.41], low).tolist() == [False, True]
    # the same episodes fitted at the default cut and at the fallback give the same F'
    ref = normalize.fit([_feats(s) for s in range(3)], CFG.resample, source="s").arms["L"]
    assert n.grasp == pytest.approx(ref.grasp, abs=0.1)
    # the fallback applies per group: a group that reaches the cut keeps the threshold rule
    mixed = normalize.fit(
        [("weak", raw[0]), ("weak", raw[1]), ("firm", _feats(0)), ("firm", _feats(1))],
        CFG.resample,
        source="phantom",
    )
    assert mixed.groups["weak"]["L"].grasp_rule == "p75"
    assert mixed.groups["firm"]["L"].grasp_rule == "threshold"


def test_fallback_does_not_hide_missing_contact() -> None:
    # closure present but the force never leaves the baseline: still an error (no contact)
    flat = _scaled(0, force_gain=0.0, g_gain=0.5)
    with pytest.raises(normalize.NormError, match="no grasp frames"):
        normalize.fit([flat], CFG.resample, source="s")


def test_norm_group() -> None:
    ep = make_pad_episode(0)
    assert normalize.norm_group(ep) == "all"
    ph = ep.model_copy(update={"source": "phantom"})
    assert (
        normalize.norm_group(ph.model_copy(update={"meta": {"task_folder": "Carton"}})) == "Carton"
    )
    assert (
        normalize.norm_group(ph.model_copy(update={"meta": {"task_folder": "Carton_fail"}}))
        == "Carton"
    )
    assert normalize.norm_group(ph.model_copy(update={"meta": {"object": "egg"}})) == "egg"
    assert normalize.norm_group(ph.model_copy(update={"meta": {}})) == "all"
    # other sources never group, whatever their meta says
    assert normalize.norm_group(ep.model_copy(update={"meta": {"task_folder": "x_fail"}})) == "all"


def test_fallback_when_too_few_episodes_reach_the_cut() -> None:
    # five episodes, one of them closing fully: 1/5 < 25% contribute, so the p75 rule applies
    weak = [_scaled(s, g_gain=0.5) for s in range(4)]
    firm = _feats(4)
    one_of_five = normalize.fit([*weak, firm], CFG.resample, source="s").arms["L"]
    assert one_of_five.grasp_rule == "p75"
    # 2/5 = 40% is enough at the default 25%, but not at min_group_frac 0.5
    two_of_five = normalize.fit([*weak[:3], firm, _feats(5)], CFG.resample, source="s").arms["L"]
    assert two_of_five.grasp_rule == "threshold"
    strict = normalize.fit(
        [*weak[:3], firm, _feats(5)], CFG.resample, source="s", min_group_frac=0.5
    ).arms["L"]
    assert strict.grasp_rule == "p75"
    # exactly 25% (1 of 4) is not "fewer than" and keeps the threshold rule
    one_of_four = normalize.fit([*weak[:3], firm], CFG.resample, source="s").arms["L"]
    assert one_of_four.grasp_rule == "threshold"
    # all three fits land on the same grasp force for this fixture (hold 5.0 N)
    for n in (one_of_five, two_of_five, strict, one_of_four):
        assert abs(n.grasp - 5.0) < 0.3


def test_p99_fprime_diagnostic(tmp_path: Path) -> None:
    raw = [_feats(s) for s in range(3)]
    consts = normalize.fit(raw, CFG.resample, source="synthetic")
    n = consts.arms["L"]
    assert n.p99_fprime is not None and 0.9 < n.p99_fprime < 1.3  # hold at 5 N is F' 1.0
    # a 3x squeeze in the same group lifts the tail but not the grasp median
    heavy = normalize.fit(
        [*raw, *(_scaled(s, force_gain=3.0) for s in (7,))], CFG.resample, source="s"
    )
    assert heavy.arms["L"].p99_fprime is not None and heavy.arms["L"].p99_fprime > 2.0
    assert abs(heavy.arms["L"].grasp - n.grasp) < 0.3
    p = tmp_path / "norm.json"
    normalize.save(consts, p)
    assert normalize.load(p).arms["L"].p99_fprime == n.p99_fprime
