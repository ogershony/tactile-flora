"""Data models that cross stage boundaries (Episode, Window, ManifestRow, ...).

Owns: every pydantic model that one stage writes to disk and another reads. Timestamps are
float seconds on the episode clock (zero at the first video frame).
Must never: hold configuration (see config.py), import from any sibling stage, or do I/O.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tactile_flora.config import RenderConfig, ResampleConfig

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


@runtime_checkable
class FrameReader(Protocol):
    """Pluggable video decoding used by verification (PyAV in production, in-memory in tests)."""

    def frame_count(self, episode: Episode, role: str) -> int:
        """Return the number of decodable frames of camera ``role``; raise if it cannot decode."""
        ...

    def read_frame(self, episode: Episode, role: str, index: int) -> np.ndarray:
        """Return frame ``index`` of camera ``role`` as an (H, W, 3) uint8 RGB array."""
        ...


class BenchmarkHashRow(BaseModel):
    """One benchmark episode's first, middle and last frame pHashes (16 hex chars each)."""

    source: str
    episode_id: str
    h_first: str
    h_mid: str
    h_last: str


class VerifyThresholds(BaseModel):
    """Thresholds of the verification checks; defaults follow the spec table."""

    rate_tol_frames: float = 2.0  # video duration vs episode duration, in frame periods
    rate_tol_frac: float = 0.2  # median force dt vs declared rate
    noise_floor: float = 1e-4  # minimum channel standard deviation (raw units)
    saturation_frac: float = 0.01
    baseline_sigma: float = 2.0
    baseline_window_s: float = 0.5
    sync_grid_s: float = 0.025
    sync_search_s: float = 1.0
    sync_median_s: float = 0.1  # source-level median |lag|
    sync_max_s: float = 0.25  # per-episode max |lag|
    norm_grasp_tol: float = 0.35  # |median grasp - 1.0|
    norm_free_tol: float = 0.15  # median free space F'
    norm_p99_max: float = 3.0
    label_rate_factor: float = 2.0
    spot_n: int = 20
    spot_min_pass: int = 17
    overlap_max_distance: int = 6


class VerifyContext(BaseModel):
    """Everything the verification checks need; all fields beyond ``source`` have defaults.

    ``norm`` is a force.normalize.NormConstants (typed Any to keep schema free of stage imports).
    ``windows`` maps episode id to its candidate Windows; ``split_of`` maps episode id to its
    assigned split. ``baseline_stats`` maps "session/arm/channel" to (mean, std) of per-episode
    baselines. ``observed_label_rate`` is the source-level fraction of episodes with a bad label.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    holdout_ids: frozenset[str] = frozenset()
    license_allowed: list[str] = Field(default_factory=lambda: list[str]())
    seed: int = 0
    reader: FrameReader | None = None
    norm: Any = None
    resample_cfg: ResampleConfig | None = None
    render_cfg: RenderConfig | None = None
    thresholds: VerifyThresholds = Field(default_factory=VerifyThresholds)
    benchmark_index: list[BenchmarkHashRow] = Field(
        default_factory=lambda: list[BenchmarkHashRow]()
    )
    hash_cache: Any = None
    expected_label_rate: float | None = None
    observed_label_rate: float | None = None
    spot_n: int = 0
    spot_reviewer_pass: int | None = None
    windows: dict[str, list[Window]] = Field(default_factory=lambda: dict[str, list[Window]]())
    split_of: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    baseline_stats: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: dict[str, tuple[float, float]]()
    )


class HoldoutFile(BaseModel):
    """The frozen split file (data/manifest/holdout_ids.json), written once.

    ``splits`` maps "source/episode_id" to the split every manifest row of that episode had when
    the file was frozen; a later build may never change one. Held-out ids are those not in train.
    """

    version: int = 1
    git_hash: str = "unknown"
    manifests: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    splits: dict[str, Split]

    @property
    def held_out_ids(self) -> frozenset[str]:
        """Keys ("source/episode_id") whose frozen split is not train."""
        return frozenset(k for k, v in self.splits.items() if v != "train")


def episode_key(source: str, episode_id: str) -> str:
    """Return the "source/episode_id" key used in the holdout file."""
    return f"{source}/{episode_id}"


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
