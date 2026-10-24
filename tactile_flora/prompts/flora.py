"""FLORA dense and outcome prompt builders with the force block inserted in one place.

Owns: PromptBundle, VideoSpec, build_prompt, to_messages, prompt_diff, and the single function
(_assemble) that decides where the force block and force sentence go.
Must never: parse outputs, render force, or load models / tokenizers.

The force block (header + lines, already rendered by force/render.py) goes after the video and
before the trailing template text. ``answer_prefix`` is "" because the target string already
starts at the first grammar token (``Worst:`` / ``Rationale:`` / ``Score:``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from tactile_flora.prompts import templates as T
from tactile_flora.schema import ResampledForce, Window
from tactile_flora.video.force_chart import CHART_FRAMES
from tactile_flora.video.mosaic import MOSAIC_CELL_PX, mosaic_layout

DENSE_FPS = 4.0


class VideoSpec(BaseModel):
    """Everything the video processor needs; frames are decoded elsewhere."""

    frame_times: list[float]
    frame_indices: list[int]
    fps: float
    layout: tuple[int, int]  # (rows, cols)
    roles: list[str]  # row-major cell order, scene first (upper left)
    cell_px: int = MOSAIC_CELL_PX
    blank: bool = False
    chart: ResampledForce | None = None  # force chart appended as one temporal group

    @property
    def n_video_frames(self) -> int:
        """Decoded camera frames (the chart placeholders excluded)."""
        return len(self.frame_times) - (CHART_FRAMES if self.chart is not None else 0)


class PromptBundle(BaseModel):
    """A prompt as pieces; ``to_messages`` turns it into chat messages."""

    system: str
    user_text_before_video: str
    user_text_after_video: str
    video: VideoSpec
    answer_prefix: str = ""


def _assemble(
    system: str, clip_line: str, force_block: str | None, chart: bool = False
) -> tuple[str, str]:
    """Place the force block, force sentence and chart sentence; the ONLY function that does.

    ``clip_line`` is the user-template text after ``<video>``. Returns (system,
    user_text_after_video). No block means neither header nor sentence. With a block, the
    sentence is a separate trailing system line and the block is its own line directly after
    the video item, before the trailing text. With ``chart`` the chart sentence is one more
    trailing system line.
    """
    after = clip_line if force_block is None else f"\n{force_block}{clip_line}"
    if force_block is not None:
        system = f"{system}\n{T.FORCE_SENTENCE}"
    if chart:
        system = f"{system}\n{T.CHART_SENTENCE}"
    return system, after


def build_prompt(
    window: Window,
    force_block: str | None,
    *,
    video_blank: bool = False,
    with_rationale: bool = True,
    task: str = "",
    camera_roles: Sequence[str] = ("scene",),
    fps: float | None = None,
    frame_indices: Sequence[int] | None = None,
    force_chart: ResampledForce | None = None,
) -> PromptBundle:
    """Build the prompt for one window; the single source of truth for training and eval.

    ``fps`` is the source video fps for the timestamp tokens (default 4.0 for dense). If
    ``frame_indices`` is omitted they are round(t * fps), floored at 0. ``force_chart``
    (outcome prompts only) appends one temporal group of chart frames: the frame times and
    indices are extended by repeating their last entry ``CHART_FRAMES`` times.
    """
    layout, roles = mosaic_layout(camera_roles)
    rate = fps if fps is not None else DENSE_FPS
    times = list(window.frame_times)
    idx = (
        list(frame_indices)
        if frame_indices is not None
        else [max(0, int(t * rate + 0.5)) for t in times]
    )
    if len(idx) != len(times):
        raise ValueError("frame_indices must match frame_times")
    if force_chart is not None:
        if window.prompt_kind != "outcome":
            raise ValueError("the force chart is defined for outcome prompts only")
        times = times + [times[-1]] * CHART_FRAMES
        idx = idx + [idx[-1]] * CHART_FRAMES
    task = " ".join(task.split())
    if window.prompt_kind == "dense":
        system = T.DENSE_SYSTEM
        user = T.DENSE_USER.format(task=task, start_s=times[0], end_s=times[-1])
    else:
        system = T.OUTCOME_SYSTEM_RATIONALE if with_rationale else T.OUTCOME_SYSTEM_SCORE_ONLY
        user = T.OUTCOME_USER.format(task=task)
    before, _, clip = user.partition(T.VIDEO_MARKER)
    system, after = _assemble(system, clip, force_block, chart=force_chart is not None)
    video = VideoSpec(
        frame_times=times,
        frame_indices=idx,
        fps=rate,
        layout=layout,
        roles=roles,
        blank=video_blank,
        chart=force_chart,
    )
    return PromptBundle(
        system=system, user_text_before_video=before, user_text_after_video=after, video=video
    )


def to_messages(bundle: PromptBundle) -> list[dict[str, Any]]:
    """Qwen chat-template-ready messages with a ``{"type": "video"}`` placeholder."""
    return [
        {"role": "system", "content": [{"type": "text", "text": bundle.system}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": bundle.user_text_before_video},
                {"type": "video"},
                {"type": "text", "text": bundle.user_text_after_video},
            ],
        },
    ]


def prompt_diff(
    window: Window,
    *,
    header: str = "Force:",
    with_rationale: bool = True,
    task: str = "",
    camera_roles: Sequence[str] = ("scene",),
) -> bool:
    """Assert the force and control prompts differ only by the block and the force sentence.

    The block is ``header`` plus ``window.force_lines``. Returns True; raises AssertionError
    on any other difference.
    """
    block = "\n".join([header, *window.force_lines])
    kw: dict[str, Any] = {
        "with_rationale": with_rationale,
        "task": task,
        "camera_roles": camera_roles,
    }
    force = build_prompt(window, block, **kw)
    control = build_prompt(window, None, **kw)
    assert force.system == f"{control.system}\n{T.FORCE_SENTENCE}", "system differs beyond sentence"
    assert force.user_text_after_video == f"\n{block}{control.user_text_after_video}", (
        "user text differs beyond the force block"
    )
    assert force.user_text_before_video == control.user_text_before_video
    assert force.video == control.video and force.answer_prefix == control.answer_prefix
    return True
