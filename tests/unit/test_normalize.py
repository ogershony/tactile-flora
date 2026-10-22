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
