import pytest

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.eval.ablations import PoolEntry, apply_force_mode
from tactile_flora.force import normalize
from tactile_flora.force.render import render_lines
from tactile_flora.force.resample import resample
from tactile_flora.schema import ResampledForce
from tests.fixtures.synthetic import make_pad_episode

CFG = load_config("configs/force.yaml", ForceConfig)
TIMES = [i * 0.25 for i in range(10)]


def _pool() -> list[ResampledForce]:
    raw = []
    for s in range(3):
        ep = make_pad_episode(s)
        raw.append(resample(ep.force, ep.gripper, TIMES, CFG.resample))
    consts = normalize.fit(raw, CFG.resample, source="synthetic")
    return [normalize.apply(r, consts, CFG.render) for r in raw]


def test_modes() -> None:
    pool = _pool()
    f = pool[0]
    assert apply_force_mode(f, "real", seed=0).features == f
    assert apply_force_mode(f, "omitted", seed=0).features is None
    vb = apply_force_mode(f, "video_blank", seed=0)
    assert vb.video_blank and vb.features == f
    assert not apply_force_mode(f, "real", seed=0).video_blank


def test_zero_renders_zeros() -> None:
    z = apply_force_mode(_pool()[0], "zero", seed=0).features
    assert z is not None
    for line in render_lines(z, TIMES, "pad"):
        assert "F=0.00 D=0.00 dF=0.00 g=0.00 c=0" in line
        assert "c=1" not in line


def test_shuffled_draws_from_other_episode() -> None:
    pool = _pool()
    a = apply_force_mode(pool[0], "shuffled", pool=pool, seed=3).features
    b = apply_force_mode(pool[0], "shuffled", pool=pool, seed=3).features
    assert a == b
    assert a is not None and a != pool[0]
    assert any(a.arms["L"] == p.arms["L"] for p in pool[1:])
    assert a.frame_times == pool[0].frame_times


def test_shuffled_needs_donor() -> None:
    f = _pool()[0]
    with pytest.raises(ValueError):
        apply_force_mode(f, "shuffled", pool=[f], seed=0)


def _entries(sources: list[str], episodes: list[str], kinds: list[str]) -> list[PoolEntry]:
    feats = _pool()
    return [
        PoolEntry(features=feats[i % 3], source=s, episode_id=e, prompt_kind=k)
        for i, (s, e, k) in enumerate(zip(sources, episodes, kinds, strict=True))
    ]


def test_shuffled_donor_same_source_other_episode_same_kind() -> None:
    pool = _entries(
        ["a", "a", "a", "b", "a"],
        ["e0", "e0", "e1", "e2", "e3"],
        ["dense", "dense", "outcome", "dense", "dense"],
    )
    me = pool[0]
    donors = {
        id(apply_force_mode(me.features, "shuffled", pool=pool, seed=s, identity=me).features)
        for s in range(30)
    }
    assert donors  # only entry 4 qualifies: same source+kind, different episode
    for s in range(30):
        got = apply_force_mode(me.features, "shuffled", pool=pool, seed=s, identity=me).features
        assert got is not None and got.arms == pool[4].features.arms
    lone = [pool[0], pool[1], pool[2], pool[3]]  # e0 twice, e1 other kind, e2 other source
    with pytest.raises(ValueError, match="different episode of the same source"):
        apply_force_mode(me.features, "shuffled", pool=lone, seed=0, identity=me)
    bare = [p.features for p in pool]
    with pytest.raises(ValueError):  # bare features carry no identity, so none qualifies
        apply_force_mode(me.features, "shuffled", pool=bare, seed=0, identity=me)
