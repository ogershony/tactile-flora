"""Data models that cross stage boundaries (Episode, Window, ManifestRow, ...).

Owns: every pydantic model that one stage writes to disk and another reads. Timestamps are
float seconds on the episode clock (zero at the first video frame).
Must never: hold configuration (see config.py), import from any sibling stage, or do I/O.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Arm = Literal["L", "R"]
Quality = Literal[1, 2, 3]
Split = Literal["train", "val", "test", "benchmark"]
PromptKind = Literal["dense", "outcome"]


class VideoRef(BaseModel):
    """Reference to one camera's video file on disk."""

    path: str
    fps: float = Field(gt=0)
    n_frames: int = Field(ge=0)


class PadArmForce(BaseModel):
    """Two pad readings (raw units) for one arm; stored as float lists for serialization."""

    kind: Literal["pad"] = "pad"
    f_a: list[float]
    f_b: list[float]

    @property
    def f_a_array(self) -> np.ndarray:
        """Pad reading a as a float64 array."""
        return np.asarray(self.f_a, dtype=np.float64)

    @property
    def f_b_array(self) -> np.ndarray:
        """Pad reading b as a float64 array."""
        return np.asarray(self.f_b, dtype=np.float64)

    def n_samples(self) -> int:
        """Return the number of samples in each channel."""
        return len(self.f_a)

    @model_validator(mode="after")
    def _equal_lengths(self) -> PadArmForce:
        if len(self.f_a) != len(self.f_b):
            raise ValueError("f_a and f_b must have equal length")
        return self


class WristArmForce(BaseModel):
    """Six-axis wrist wrench (raw units) for one arm; stored as float lists."""

    kind: Literal["wrist"] = "wrist"
    fx: list[float]
    fy: list[float]
    fz: list[float]
    tx: list[float]
    ty: list[float]
    tz: list[float]

    @property
    def wrench(self) -> np.ndarray:
        """Wrench as an (N, 6) float64 array with columns fx fy fz tx ty tz."""
        return np.stack(
            [
                np.asarray(c, dtype=np.float64)
                for c in (self.fx, self.fy, self.fz, self.tx, self.ty, self.tz)
            ],
            axis=1,
        )

    def n_samples(self) -> int:
        """Return the number of samples in each channel."""
        return len(self.fx)

    @model_validator(mode="after")
    def _equal_lengths(self) -> WristArmForce:
        n = len(self.fx)
        if any(len(c) != n for c in (self.fy, self.fz, self.tx, self.ty, self.tz)):
            raise ValueError("all wrench components must have equal length")
        return self


ArmForce = Annotated[PadArmForce | WristArmForce, Field(discriminator="kind")]


class ForceStream(BaseModel):
    """Raw force for every arm on the episode clock."""

    channel: Literal["pad", "wrist"]
    arms: dict[Arm, ArmForce]
    rate_hz: float = Field(gt=0)
    t: list[float]

    @property
    def t_array(self) -> np.ndarray:
        """Timestamps as a float64 array."""
        return np.asarray(self.t, dtype=np.float64)

    @model_validator(mode="after")
    def _consistent(self) -> ForceStream:
        for arm, data in self.arms.items():
            if data.kind != self.channel:
                raise ValueError(f"arm {arm} kind {data.kind} != stream channel {self.channel}")
            if data.n_samples() != len(self.t):
                raise ValueError(f"arm {arm} length {data.n_samples()} != len(t) {len(self.t)}")
        if any(b < a for a, b in zip(self.t, self.t[1:], strict=False)):
            raise ValueError("t must be monotonically non-decreasing")
        return self


class GripperStream(BaseModel):
    """Gripper position per arm (0 open .. 1 closed) at native rate."""

    arms: dict[Arm, list[float]]
    rate_hz: float = Field(gt=0)
    t: list[float]

    @model_validator(mode="after")
    def _consistent(self) -> GripperStream:
        for arm, values in self.arms.items():
            if len(values) != len(self.t):
                raise ValueError(f"arm {arm} length {len(values)} != len(t) {len(self.t)}")
        return self


class Span(BaseModel):
    """A time interval with a quality digit and a rationale."""

    t0: float
    t1: float
    quality: Quality
    rationale: str = ""
    # Provenance of the span after labels.merge.merge_labels ("" for unmerged spans).
    origin: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.t1 < self.t0:
            raise ValueError("t1 must be >= t0")
        return self


class Label(BaseModel):
    """One label for an episode: an outcome score, dense spans, or event spans."""

    kind: Literal["outcome", "dense", "event"]
    origin: Literal["human", "programmatic", "pseudo", "source"]
    outcome: int | None = Field(default=None, ge=1, le=5)
    spans: list[Span] = Field(default_factory=lambda: list[Span]())


class Episode(BaseModel):
    """One recording on a single episode clock, as produced by a source adapter."""

    source: str
    episode_id: str
    session: str
    task: str
    duration_s: float = Field(ge=0)
    cameras: dict[str, VideoRef]
    force: ForceStream
    gripper: GripperStream | None = None
    labels: list[Label] = Field(default_factory=lambda: list[Label]())
    license: str
    meta: dict[str, Any] = Field(default_factory=dict)


class Window(BaseModel):
    """A training or eval sample; immutable once created."""

    model_config = ConfigDict(frozen=True)

    episode_ref: str
    prompt_kind: PromptKind
    anchor_s: float
    frame_times: tuple[float, ...]
    force_lines: tuple[str, ...]
    target: str
    split: Split

    @model_validator(mode="after")
    def _frame_count(self) -> Window:
        # dense windows always carry 64 positions; outcome samples carry m in [2, 64], even.
        n = len(self.frame_times)
        if self.prompt_kind == "dense" and n != 64:
            raise ValueError("dense windows must have exactly 64 frame_times")
        if self.prompt_kind == "outcome" and not (2 <= n <= 64 and n % 2 == 0):
            raise ValueError("outcome windows must have an even number of frame_times in [2, 64]")
        return self


class ManifestRow(BaseModel):
    """One row per accepted Window in data/manifest/<name>.parquet."""

    window_id: str
    source: str
    episode_id: str
    session: str
    split: Split
    prompt_kind: PromptKind
    anchor_s: float
    frame_hashes: list[str]
    norm_version: str
    label_origin: str
    license: str
    report_version: str


class CheckResult(BaseModel):
    """Outcome of one verification check on one episode."""

    passed: bool
    value: float | None = None
    action: Literal["keep", "drop_episode", "drop_channel", "shift"] = "keep"
    detail: str = ""


class VerifyContext(BaseModel):
    """Minimal context handed to every verification check."""

    source: str
    holdout_ids: frozenset[str] = frozenset()
    license_allowed: list[str] = Field(default_factory=list)
    seed: int = 0


class ArmFeatures(BaseModel):
    """Per-frame force features for one arm; every list has one entry per frame time.

    ``F``, ``delta`` and ``dF`` are raw units out of the resampler and normalized units after
    ``normalize.apply``. ``F`` is NaN for a frame whose window held no samples. ``delta`` is None
    for wrist channels, ``g`` is None when no gripper stream exists, and ``c`` (contact flag,
    0 or 1) is None until the features have been normalized. ``raw_groups`` holds one raw 100 Hz
    slice of the scalar F series (pad: f_a + f_b, wrist: wrench force magnitude) per 0.5 s
    group, always in raw units, so a learned encoder can reuse it later.
    """

    F: list[float]
    delta: list[float] | None = None
    dF: list[float]
    g: list[float] | None = None
    c: list[int] | None = None
    raw_groups: list[list[float]] = Field(default_factory=lambda: list[list[float]]())

    @model_validator(mode="after")
    def _equal_lengths(self) -> ArmFeatures:
        n = len(self.F)
        for name in ("delta", "dF", "g", "c"):
            col = getattr(self, name)
            if col is not None and len(col) != n:
                raise ValueError(f"{name} length {len(col)} != len(F) {n}")
        return self


class ResampledForce(BaseModel):
    """Frame-aligned force features for every arm of one episode window."""

    channel: Literal["pad", "wrist"]
    frame_times: list[float]
    arms: dict[Arm, ArmFeatures]
    norm_version: str | None = None

    @model_validator(mode="after")
    def _lengths(self) -> ResampledForce:
        for arm, feats in self.arms.items():
            if len(feats.F) != len(self.frame_times):
                raise ValueError(
                    f"arm {arm} has {len(feats.F)} frames, expected {len(self.frame_times)}"
                )
        return self
