"""YAML configuration models and the loader that turns a config file into one of them.

Owns: SourceConfig, ForceConfig, TrainConfig, EvalConfig, GridConfig, BaxterConfig and
load_config. All models forbid extra keys so typos fail at load time.
Must never: be read by stages directly from disk; the CLI loads once and passes models down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ForceMode = Literal["real", "omitted", "zero", "shuffled", "video_blank"]
ForceFormat = Literal["decimal", "code"]


class _Strict(BaseModel):
    """Base for config models: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class SegmentSelect(_Strict):
    """Which segments become episodes in ``episode_unit: segment`` and how they are sampled.

    A segment qualifies when its task is not idle and its last frame is at least
    ``min_duration_s`` after its first. Per split, every failed segment is kept plus up to
    ``success_multiple`` times as many successes, drawn after a ``random.Random(seed)`` shuffle
    of the successes in recording-then-segment order; ``max_total`` caps the split's total
    (successes are what gets cut; 0 or absent = no cap). This is the protocol of
    data/colab/e0b_reassemble_force.py (seed 0, multiple 2, total 450 on the test split).
    """

    min_duration_s: float = Field(default=1.0, ge=0)
    success_multiple: float = Field(default=2.0, ge=0)
    max_total: dict[str, int] = Field(default_factory=lambda: dict[str, int]())
    seed: int = 0


class SourceConfig(_Strict):
    """Per-dataset adapter configuration (configs/sources/<name>.yaml)."""

    name: str
    root: str
    license_allowed: list[str]
    max_windows: int = Field(ge=0)
    max_episodes: int = Field(ge=0)
    # per-prompt-kind train caps; None falls back to max_windows (0 = no cap)
    max_dense_windows: int | None = Field(default=None, ge=0)
    max_outcome_windows: int | None = Field(default=None, ge=0)
    eval_only: bool
    expected_label_rate: float = Field(ge=0, le=1)
    split_rules: dict[str, Any] = Field(default_factory=dict)
    seed: int
    # source-level median |sync lag| gate in seconds; None = VerifyThresholds default (0.1)
    sync_median_s: float | None = Field(default=None, gt=0)
    # source-level bound on the pooled 99th percentile of F'; None = VerifyThresholds default (3)
    norm_p99_max: float | None = Field(default=None, gt=0)
    # source-level band for the pooled (median over episodes) grasp median of F'; None = the
    # spec's [0.8, 1.2] (verify.report.NORM_GRASP_RANGE). Must contain 1.0.
    norm_grasp_range: tuple[float, float] | None = None
    # camera restriction for multi-camera LeRobot ports: the video feature keys to keep, in role
    # order (the first key a port has becomes its scene camera); None = every camera the adapter
    # finds. Honored by rh20t, whose partial downloads carry 2 of the 9-10 cameras per config.
    cameras: list[str] | None = None
    # Episode unit for sources whose recordings carry labelled segments (reassemble_lerobot):
    # "recording" (default) yields one Episode per recording, "segment" one Episode per
    # qualifying segment (see ``SegmentSelect`` and sources/reassemble_lerobot.py).
    episode_unit: Literal["recording", "segment"] = "recording"
    # Segment-mode sampling (ignored in recording mode).
    segment_select: SegmentSelect = Field(default_factory=lambda: SegmentSelect())
    # Name of an already verified source whose recording-level checks and normalization
    # constants this source reuses (data/verified/<verify_from>/): a segment source reads the
    # integrity, sensor-health, sync, normalization and overlap results of its parent recording
    # instead of decoding every mp4 again. Only cheap per-episode checks are re-run.
    verify_from: str | None = None
    # With verify_from: "apply" (default) lets the parent's per-episode drop_episode /
    # drop_channel results drop the child episode too; "record" keeps the child and stores the
    # parent's failure with action "keep" (visible in checks.parquet and report.md, the force
    # then rendered without a sync shift). reassemble_seg uses "record" so that its test split
    # is the complete E0b clip set; see docs/deviations.md section 7.
    inherited_drops: Literal["apply", "record"] = "apply"

    @field_validator("norm_grasp_range")
    @classmethod
    def _band_contains_one(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and not (0 < v[0] <= 1.0 <= v[1]):
            raise ValueError(f"norm_grasp_range {v} must satisfy 0 < low <= 1 <= high")
        return v


class ResampleConfig(_Strict):
    """Resampling geometry for the force channel."""

    half_window_s: float
    frame_period_s: float
    group_period_s: float
    raw_rate_hz: float
    contact_sigma: float


class RenderConfig(_Strict):
    """Text rendering of the force block.

    ``format`` selects the line grammar (force/render.py): ``decimal`` is the byte-exact spec
    format (``header``, ``decimals``), ``code`` the token-cheap single-character format
    (``header_code`` and the thresholds below, all in normalized units: ``levels`` are the F'
    cut points between levels 1/2, 2/3 and 3/4; ``asym_frac`` the |delta'| / F' ratio above which
    a pad dominates; ``trend_up`` / ``trend_down`` / ``sharp_drop`` the dF' per second cut points
    for +, - and !; ``closure`` the gripper g cut points between open/mid and mid/closed).
    ``summary`` adds the per-arm summary line after the code header.
    """

    header: str
    decimals: int
    max_tokens: int
    clip_max: float
    tokenizer_json: str | None = None
    format: ForceFormat = "decimal"
    header_code: str = (
        "Force code per frame (F force level: 0 none, 1 light, 2 median grasp, 3 firm, "
        "4 crush-level; levels are relative to this object's typical grasp (F2 = this object's "
        "median grasp); A pad asymmetry a/b/=; T trend +/0/-/! sharp drop; C contact 0/1; "
        "G gripper o open/m mid/c closed):"
    )
    summary: bool = True
    levels: list[float] = Field(default_factory=lambda: [0.5, 1.5, 2.5])
    asym_frac: float = Field(default=0.3, ge=0)
    trend_up: float = 1.0
    trend_down: float = -1.0
    sharp_drop: float = -2.0
    closure: list[float] = Field(default_factory=lambda: [0.3, 0.6])
    # Optional force-only outcome classifier (force/classifier.py JSON); render.classifier_line
    # writes its success probability under the summary for episodes of the source it was fitted on,
    # out-of-fold for its own training episodes.
    classifier_json: str | None = None

    @field_validator("levels", "closure")
    @classmethod
    def _ascending(cls, v: list[float], info: Any) -> list[float]:
        n = 3 if info.field_name == "levels" else 2
        if len(v) != n or any(b <= a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(f"{info.field_name} must be {n} strictly ascending values, got {v}")
        return v


class EventsConfig(_Strict):
    """Thresholds for the programmatic force-event detectors."""

    grasp_g: float
    grasp_f: float
    grasp_window_s: float
    slip_df: float
    slip_min_s: float
    slip_g_tol: float
    contact_loss_s: float
    regrasp_window_s: float
    crush_f: float
    off_axis_delta: float


class NormalizeConfig(_Strict):
    """Normalization fit settings (force/normalize.py).

    ``grasp_g`` is the gripper closure above which a frame counts as a grasp frame when the
    constants are fitted; a group in which fewer than ``min_group_frac`` of the fitted episodes
    have such a frame falls back to its 75th percentile of g (recorded in the constants as
    ``grasp_rule``).
    """

    grasp_g: float = Field(default=0.6, ge=0.0, le=1.0)
    min_group_frac: float = Field(default=0.25, ge=0.0, le=1.0)


class ForceConfig(_Strict):
    """Force channel configuration (configs/force.yaml)."""

    resample: ResampleConfig
    render: RenderConfig
    events: EventsConfig
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)


class LoraConfig(_Strict):
    """LoRA hyperparameters."""

    rank: int
    alpha: int
    target_modules: list[str]


class TrainConfig(_Strict):
    """LoRA SFT run configuration (configs/train/*.yaml)."""

    manifest: str
    base_model: str
    force_mode: ForceMode
    seed: int
    epochs: int
    learning_rate: float
    warmup_frac: float
    grad_clip: float
    effective_batch: int
    lora: LoraConfig
    output_dir: str
    # Class-balance weight for the failure judge: the loss weight of an OUTCOME sample whose
    # target score is <= train.dataset.FAILURE_SCORE_MAX (2) is multiplied by this; dense samples
    # and successful outcomes keep 1.0. judge*.yaml set 2.2 (2026-10-22; the dry run on
    # train_judge.parquet counts 808 successes / 425 failures); force.yaml / control.yaml keep 1.0.
    failure_weight: float = Field(default=1.0, gt=0)
    # The same for the reward model: the loss weight of a DENSE sample whose Worst digit is <= 2
    # (a window with at least one failure or suboptimal group) is multiplied by this.
    # force_v2.yaml / control_v2.yaml set 3.58 (2026-10-22: train.parquet holds 279 all-normal and
    # 78 non-normal dense windows; v1 at 1.0 collapsed to all-normal).
    dense_failure_weight: float = Field(default=1.0, gt=0)


class BenchmarkSpec(_Strict):
    """One benchmark: a manifest name, prompt kinds and applicable metrics."""

    manifest: str
    prompts: list[Literal["dense", "outcome"]]
    metrics: list[str]
    score_only: bool = False


class EvalConfig(_Strict):
    """Benchmark definitions (configs/eval/benchmarks.yaml)."""

    benchmarks: dict[str, BenchmarkSpec]
    output_dir: str


class GridConfig(_Strict):
    """Ablation grid (configs/eval/grid.yaml): checkpoints and modes per benchmark."""

    checkpoints: list[str]
    modes: dict[str, list[ForceMode]]
    seed: int
    # Optional per-checkpoint cap on modes (e.g. released and control run omitted only);
    # a checkpoint not listed runs every mode its benchmark allows.
    checkpoint_modes: dict[str, list[ForceMode]] = Field(
        default_factory=lambda: dict[str, list[ForceMode]]()
    )


class BaxterTopics(_Strict):
    """ROS topic names used by the Baxter stack."""

    raw: str
    force: str
    health: str
    cameras: dict[str, str]
    gripper_state: dict[str, str]
    policy_status: str
    zero: str = "/tactile/zero"


class BaxterConfig(_Strict):
    """Baxter sensor, serial and topic configuration (configs/baxter.yaml)."""

    serial_port: str
    baud: int
    sample_rate_hz: float
    r_ref_ohm: float
    channels: list[str]
    clock_refit_s: float
    calibration_path: str
    camera_fps: float
    topics: BaxterTopics
    output_dir: str
    # Additive fields (milestone 11); all sensor-dependent values stay UNCONFIRMED until the
    # Amazon part (ASIN B0FGD267GK) has its resistance curve measured.
    sensor_confirmed: bool = False
    adc_max_counts: int = 4095
    saturation_counts: int = 4090
    saturation_frac_max: float = 0.05
    flat_variance: float = 0.25
    health_period_s: float = 1.0
    health_max_age_s: float = 3.0
    gripper_open_max: float = 0.1
    calibration_samples: int = 200
    # tf baxter calibrate (one pad at a time pushed against a kitchen scale): usable pushes required
    # per pad besides the unloaded reading, and the count above which the ESP32 ADC leaves its
    # linear range (about 2.45 V of 3.3 V); points above it are kept but flagged.
    calibration_min_points: int = 4
    linear_max_counts: int = 3000


M = TypeVar("M", bound=BaseModel)


def load_config(path: Path | str, model: type[M]) -> M:
    """Read the YAML file at ``path`` and validate it into ``model``."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return model.model_validate(raw)
