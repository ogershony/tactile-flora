import numpy as np
import pytest
from pydantic import ValidationError

from tactile_flora.schema import (
    CheckResult,
    Episode,
    ForceStream,
    Label,
    PadArmForce,
    Span,
    VideoRef,
    Window,
    WristArmForce,
)
from tactile_flora.sources.base import strip_outcome_words


def _pad_stream(n: int = 10) -> ForceStream:
    return ForceStream(
        channel="pad",
        arms={"L": PadArmForce(f_a=[0.0] * n, f_b=[1.0] * n)},
        rate_hz=100.0,
        t=[i / 100 for i in range(n)],
    )


def test_episode_roundtrip_json() -> None:
    ep = Episode(
        source="phantom",
        episode_id="e0",
        session="s0",
        task="pick the cup",
        duration_s=1.0,
        cameras={"scene": VideoRef(path="a.mp4", fps=30, n_frames=10)},
        force=_pad_stream(),
        labels=[Label(kind="outcome", origin="source", outcome=5)],
        license="CC-BY-4.0",
    )
    again = Episode.model_validate_json(ep.model_dump_json())
    assert again == ep
    arm = again.force.arms["L"]
    assert isinstance(arm, PadArmForce)
    assert arm.f_b_array.dtype == np.float64


def test_wrist_arm_and_channel_mismatch() -> None:
    wrist = WristArmForce(fx=[0.0], fy=[0.0], fz=[0.0], tx=[0.0], ty=[0.0], tz=[0.0])
    assert wrist.wrench.shape == (1, 6)
    with pytest.raises(ValidationError):
        ForceStream(channel="pad", arms={"L": wrist}, rate_hz=100.0, t=[0.0])


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        PadArmForce(f_a=[0.0], f_b=[0.0, 1.0])


def test_label_validation() -> None:
    with pytest.raises(ValidationError):
        Label(kind="outcome", origin="human", outcome=6)
    with pytest.raises(ValidationError):
        Span(t0=0, t1=1, quality=4, rationale="x")  # pyright: ignore[reportArgumentType]


def test_window_is_immutable() -> None:
    w = Window(
        episode_ref="phantom:e0",
        prompt_kind="dense",
        anchor_s=0.0,
        frame_times=tuple(i / 4 for i in range(64)),
        force_lines=(),
        target="Worst: 3",
        split="train",
    )
    with pytest.raises(ValidationError):
        w.anchor_s = 1.0
    with pytest.raises(ValidationError):
        Window(**{**w.model_dump(), "frame_times": (0.0,)})


def test_check_result_defaults() -> None:
    assert CheckResult(passed=True).action == "keep"


def test_strip_outcome_words() -> None:
    assert strip_outcome_words("pick_fail the cup") == "pick the cup"
    assert strip_outcome_words("Retry grasp after failure, then recovery") == "grasp after , then"
    assert strip_outcome_words("stack the blocks") == "stack the blocks"
