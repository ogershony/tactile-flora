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
