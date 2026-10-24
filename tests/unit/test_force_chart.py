"""Tests for the force chart frames and the chart force modes."""

from __future__ import annotations

import numpy as np
import pytest

from tactile_flora.eval.ablations import apply_force_mode
from tactile_flora.prompts import templates as T
from tactile_flora.prompts.flora import build_prompt
from tactile_flora.schema import ArmFeatures, ResampledForce, Window
from tactile_flora.video.force_chart import (
    CHART_FRAMES,
    RED,
    append_chart,
    chart_image,
    clip_array,
)


def _features(n: int = 16, contact: bool = True) -> ResampledForce:
    F = [0.0] * 4 + [1.5] * (n - 8) + [0.0] * 4
    return ResampledForce(
        channel="pad",
        frame_times=[i * 0.5 for i in range(n)],
        arms={
            "R": ArmFeatures(
                F=F,
                delta=[0.0] * n,
                dF=[0.0] * n,
                g=[0.0] * 4 + [1.0] * (n - 4),
                c=[int(v > 0.1) if contact else 0 for v in F],
            )
        },
        norm_version="x",
    )


def test_chart_image_shape_determinism_and_marks() -> None:
    a = chart_image(_features(), (96, 96))
    b = chart_image(_features(), (96, 96))
    assert a.shape == (96, 96, 3) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    red = np.all(a == np.array(RED, dtype=np.uint8), axis=-1)
    assert red.any()
    # The red stroke sits in the upper half where F=1.5 (of 3) is drawn, and is absent at the
    # left margin where F=0 only touches the bottom.
    rows = np.flatnonzero(red.any(axis=1))
    assert rows.min() < 48
    gray_band = np.all(a == np.array((215, 215, 215), dtype=np.uint8), axis=-1)
    assert gray_band.any()
    no_contact = chart_image(_features(contact=False), (96, 96))
    assert not np.all(no_contact == np.array((215, 215, 215), dtype=np.uint8), axis=-1).any()


def test_chart_requires_normalized_features() -> None:
    f = _features()
    f.arms["R"].c = None
    with pytest.raises(ValueError, match="normalized"):
        chart_image(f, (96, 96))


def test_append_chart_adds_one_temporal_group() -> None:
    frames = [np.zeros((96, 96, 3), dtype=np.uint8)] * 3
    chart = np.ones((96, 96, 3), dtype=np.uint8)
    out = append_chart(frames, chart)
    assert len(out) == 3 + CHART_FRAMES and all(o is chart for o in out[3:])
    with pytest.raises(ValueError, match="shape"):
        append_chart(frames, np.ones((64, 64, 3), dtype=np.uint8))


def test_chart_modes_and_prompt() -> None:
    f = _features()
    chart = apply_force_mode(f, "chart", seed=0)
    assert chart.chart and not chart.text and chart.features is f
    both = apply_force_mode(f, "chart_text", seed=0)
    assert both.chart and both.text
    plain = apply_force_mode(f, "real", seed=0)
    assert plain.text and not plain.chart
    with pytest.raises(ValueError, match="unknown force mode"):
        apply_force_mode(f, "bogus", seed=0)  # type: ignore[arg-type]
    w = Window(
        episode_ref="src/ep",
        prompt_kind="outcome",
        anchor_s=0.0,
        frame_times=tuple(f.frame_times),
        force_lines=(),
        target="Rationale: no rationale\n\nScore: 5",
        split="train",
    )
    bundle = build_prompt(w, None, force_chart=f, fps=30.0)
    assert len(bundle.video.frame_times) == 16 + CHART_FRAMES
    assert bundle.video.frame_times[-1] == bundle.video.frame_times[-CHART_FRAMES - 1]
    assert len(bundle.video.frame_indices) == len(bundle.video.frame_times)
    assert bundle.video.chart is f and bundle.video.n_video_frames == 16
    assert bundle.system.endswith(T.CHART_SENTENCE)
    assert T.FORCE_SENTENCE not in bundle.system
    text_and_chart = build_prompt(w, "Force:\nline", force_chart=f, fps=30.0)
    assert text_and_chart.system.endswith(f"{T.FORCE_SENTENCE}\n{T.CHART_SENTENCE}")
    no_chart = build_prompt(w, None, fps=30.0)
    assert no_chart.video.chart is None and no_chart.video.n_video_frames == 16
    dense = w.model_copy(update={"prompt_kind": "dense", "frame_times": tuple(range(64))})
    with pytest.raises(ValueError, match="outcome prompts only"):
        build_prompt(dense, None, force_chart=f)


def test_clip_array_appends_chart_after_blanking() -> None:
    from tactile_flora.prompts.flora import VideoSpec
    from tactile_flora.schema import Episode, ForceStream, VideoRef

    class _Reader:
        def read_many(self, episode, role, indices):  # noqa: ANN001, ANN202
            return [np.zeros((40, 40, 3), dtype=np.uint8) for _ in indices]

    ep = Episode(
        source="s",
        episode_id="e",
        session="x",
        task="t",
        duration_s=8.0,
        cameras={"scene": VideoRef(path="none.mp4", fps=30.0, n_frames=240)},
        force=ForceStream(channel="pad", arms={}, rate_hz=1.0, t=[]),
        gripper=None,
        labels=[],
        license="CC-BY-4.0",
        meta={},
    )
    f = _features()
    video = VideoSpec(
        frame_times=[i * 0.5 for i in range(16)] + [7.5] * CHART_FRAMES,
        frame_indices=[int(i * 15) for i in range(16)] + [225] * CHART_FRAMES,
        fps=30.0,
        layout=(1, 1),
        roles=["scene"],
        blank=True,
        chart=f,
    )
    clip = clip_array(ep, video, _Reader())
    assert clip.shape[0] == 16 + CHART_FRAMES
    assert (clip[:16] == 128).all()  # blanked camera frames
    np.testing.assert_array_equal(clip[16], clip[17])
    assert (clip[16] == 255).any() and not (clip[16] == 128).all()
