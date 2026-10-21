from tactile_flora.config import ForceConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.labels.events import (
    detect_all,
    detect_contact_loss,
    detect_crush,
    detect_grasp_acquired,
    detect_off_axis,
    detect_regrasp,
    detect_slip,
)
from tactile_flora.schema import ArmFeatures, ResampledForce
from tests.fixtures.synthetic import make_pad_episode

FC = load_config("configs/force.yaml", ForceConfig)
EV = FC.events
DT = 0.25
TAU = 0.1


def feats(
    F: list[float],
    g: list[float],
    delta: list[float] | None = None,
    channel: str = "pad",
) -> ResampledForce:
    n = len(F)
    dF = [0.0] + [(F[i] - F[i - 1]) / DT for i in range(1, n)]
    af = ArmFeatures(F=F, delta=delta, dF=dF, g=g, c=[int(f > TAU) for f in F])
    return ResampledForce(
        channel=channel,  # type: ignore[arg-type]
        frame_times=[i * DT for i in range(n)],
        arms={"L": af},
    )


def pad(xs: list[float], n: int = 16) -> list[float]:
    return xs + [xs[-1]] * (n - len(xs))


G_HELD = pad([0, 0] + [1.0] * 14)
F_HELD = [0, 0, 0.8] + [1.0] * 13


def test_clean_grasp_has_no_events() -> None:
    f = feats(F_HELD, G_HELD, delta=[0.05] * 16)
    assert detect_grasp_acquired(f, "L", EV) == [0.5]
    assert detect_all(f, EV)[0].spans == []


def test_grasp_needs_force_and_closure() -> None:
    assert detect_grasp_acquired(feats([0.0] * 16, G_HELD), "L", EV) == []  # closes on air
    assert detect_grasp_acquired(feats(F_HELD, [0.0] * 16), "L", EV) == []  # force, no closure


def test_slip() -> None:
    F = [0, 0, 0.8, 1, 1, 1, 0.4, 0.4, 0.4] + [0.4] * 7
    spans = detect_slip(feats(F, G_HELD), "L", EV)
    assert [(s.t0, s.quality) for s in spans] == [(1.5, 2)]
    assert spans[0].rationale == "the object slips in the left gripper"
    # moving gripper means release or regrasp, not slip
    g = [0, 0, 1, 1, 1, 1, 0.8, 0.6, 0.4] + [0.2] * 7
    assert detect_slip(feats(F, g), "L", EV) == []
    # never held: no slip
    assert detect_slip(feats([1, 1, 0.4] + [0.4] * 13, [0.0] * 16), "L", EV) == []


def test_contact_loss_and_quality() -> None:
    F = [0, 0, 0.8, 1, 1, 0, 0, 0] + [1.0] * 8
    spans = detect_contact_loss(feats(F, G_HELD), "L", EV)
    assert [(s.t0, s.t1, s.quality) for s in spans] == [(1.25, 2.0, 1)]
    assert spans[0].rationale == "the object drops from the left gripper"
    done = detect_contact_loss(feats(F, G_HELD), "L", EV, task_complete=True)
    assert done[0].quality == 2
    # a single-frame dip (0.25 s) is not over 0.3 s
    one = [0, 0, 0.8, 1, 1, 0] + [1.0] * 10
    assert detect_contact_loss(feats(one, G_HELD), "L", EV) == []
    # open gripper: release, not a drop
    assert detect_contact_loss(feats(F, [0.0] * 16), "L", EV) == []


def test_regrasp() -> None:
    F = [0, 0, 0.8, 1, 1, 0, 0, 0, 0, 0, 0, 0.8, 1, 1, 1, 1]
    g = [0, 0] + [1.0] * 7 + [0, 0] + [1.0] * 5
    f = feats(F, g)
    assert detect_grasp_acquired(f, "L", EV) == [0.5, 2.75]
    spans = detect_regrasp(f, "L", EV)
    assert [(s.t0, s.quality, s.rationale) for s in spans] == [(1.25, 2, "re-grasps the object")]
    # grasps farther apart than 3 s: no regrasp
    long_gap = feats(F + [0.0] * 8, g + [0.0] * 8)
    assert detect_regrasp(long_gap, "L", EV) != []
    far = feats(
        [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8],
        [0, 0, 1.0] + [1.0] * 9 + [0] * 3 + [0, 1.0, 1.0],
    )
    assert detect_regrasp(far, "L", EV) == []


def test_crush() -> None:
    F = [0, 0, 0.8, 2.5, 2.5, 1.0] + [1.0] * 10
    g = [0, 0, 0.7, 0.8, 0.9, 1.0] + [1.0] * 10
    spans = detect_crush(feats(F, g), "L", EV)
    assert [(s.t0, s.t1, s.quality) for s in spans] == [(0.75, 1.25, 1)]
    assert spans[0].rationale == "excessive force on the object"
    # high force with a settled gripper is not crush
    assert detect_crush(feats(F, G_HELD), "L", EV) == []


def test_off_axis() -> None:
    delta = [0.9, 0.0, 0.0, 0.0, 0.8, 0.8, 0.1] + [0.0] * 9
    spans = detect_off_axis(feats(F_HELD, G_HELD, delta=delta), "L", EV)
    assert [(s.t0, s.t1, s.quality) for s in spans] == [(1.0, 1.5, 2)]
    assert spans[0].rationale == "off-center grasp"
    assert detect_off_axis(feats(F_HELD, G_HELD, delta=delta, channel="wrist"), "L", EV) == []
    assert detect_off_axis(feats(F_HELD, G_HELD), "L", EV) == []


def test_detect_all_labels_per_arm() -> None:
    f = feats([0, 0, 0.8, 1, 1, 0, 0, 0] + [1.0] * 8, G_HELD)
    labels = detect_all(f, EV)
    assert len(labels) == 1
    assert labels[0].kind == "event" and labels[0].origin == "programmatic"
    assert any("drops" in s.rationale for s in labels[0].spans)


def test_real_pipeline_clean_grasp_has_no_events() -> None:
    times = [i * DT for i in range(20)]
    raw = []
    for seed in range(3):
        ep = make_pad_episode(seed, n_frames=20)
        raw.append(resample(ep.force, ep.gripper, times, FC.resample))
    consts = normalize.fit(raw, FC.resample, source="synthetic")
    norm = normalize.apply(raw[0], consts, FC.render)
    assert len(detect_grasp_acquired(norm, "L", EV)) == 1
    for label in detect_all(norm, EV):
        assert label.spans == []
