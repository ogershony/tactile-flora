import math

import numpy as np

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.force.resample import resample
from tactile_flora.schema import ForceStream, PadArmForce
from tests.fixtures.synthetic import make_pad_episode, make_wrist_episode

CFG = load_config("configs/force.yaml", ForceConfig)
TIMES = [i * 0.25 for i in range(10)]


def test_pad_features() -> None:
    ep = make_pad_episode(0)
    r = resample(ep.force, ep.gripper, TIMES, CFG.resample)
    L = r.arms["L"]
    assert len(L.F) == 10 and L.delta is not None and L.g is not None
    assert abs(L.F[0] - 1.0) < 0.1
    assert max(L.F) > 4.5
    assert L.dF[0] == 0.0
    assert math.isclose(L.dF[4], (L.F[4] - L.F[3]) / 0.25)
    assert L.g[0] == 0.0 and L.g[-3] > 0.9
    assert len(L.raw_groups) == 5
    assert len(L.raw_groups[1]) == 50  # frames 1,2: [0.125, 0.625] inclusive


def test_wrist_has_no_delta_and_magnitude() -> None:
    ep = make_wrist_episode(0)
    r = resample(ep.force, None, TIMES, CFG.resample)
    assert r.arms["L"].delta is None and r.arms["L"].g is None
    assert abs(r.arms["L"].F[0] - 1.0) < 0.1


def test_empty_window_is_nan() -> None:
    s = ForceStream(
        channel="pad",
        arms={"L": PadArmForce(f_a=[1.0, 1.0], f_b=[0.0, 0.0])},
        rate_hz=100,
        t=[0.0, 0.01],
    )
    r = resample(s, None, [0.0, 5.0], CFG.resample)
    assert r.arms["L"].F[0] == 1.0 and np.isnan(r.arms["L"].F[1])


def test_groups_cover_every_frame_once() -> None:
    ep = make_pad_episode(0, n_frames=64)
    times = [i * 0.25 for i in range(64)]
    L = resample(ep.force, ep.gripper, times, CFG.resample).arms["L"]
    assert len(L.raw_groups) == 32
    t_raw = ep.force.t_array
    hw = CFG.resample.half_window_s
    for k, grp in enumerate(L.raw_groups):
        lo, hi = times[2 * k] - hw, times[2 * k + 1] + hw
        assert len(grp) == int(((t_raw >= lo) & (t_raw <= hi)).sum())
    odd = resample(ep.force, ep.gripper, times[:63], CFG.resample).arms["L"]
    assert len(odd.raw_groups) == 32 and len(odd.raw_groups[-1]) < len(odd.raw_groups[0])


def test_first_frame_df_uses_raw_at_previous_period() -> None:
    t = [i * 0.01 for i in range(300)]
    a = [1.0 if x < 1.3 else 2.0 for x in t]
    s = ForceStream(
        channel="pad", arms={"L": PadArmForce(f_a=a, f_b=[0.0] * 300)}, rate_hz=100, t=t
    )
    r = resample(s, None, [1.5], CFG.resample).arms["L"]
    assert r.dF[0] == 4.0  # (2.0 - 1.0) / 0.25, with no earlier frame in the list
    r = resample(s, None, [0.1, 2.0], CFG.resample).arms["L"]
    assert r.dF[0] == 0.0  # nothing at t - 0.25
    # spacing 0.5 still divides by the 0.25 spec constant
    r = resample(s, None, [1.0, 1.5], CFG.resample).arms["L"]
    assert r.dF[1] == (r.F[1] - 1.0) / 0.25
