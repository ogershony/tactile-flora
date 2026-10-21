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
