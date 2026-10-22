"""YAML configuration models and the loader that turns a config file into one of them.

Owns: SourceConfig, ForceConfig, TrainConfig, EvalConfig, GridConfig, BaxterConfig and
load_config. All models forbid extra keys so typos fail at load time.
Must never: be read by stages directly from disk; the CLI loads once and passes models down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

ForceMode = Literal["real", "omitted", "zero", "shuffled", "video_blank"]


class _Strict(BaseModel):
    """Base for config models: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


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
    # Opt in for HRDexDB runs with downloaded raw/timestamps/{timestamp,frame_id}.npy.
    hrdexdb_use_video_timestamps: bool = False
    # HRDexDB force representation. "legacy": absolute raw values, the two channels with the
    # largest mean, hand closure proxy assuming larger joint values mean more closed. "tared":
    # per-trial resting offsets (median of the first ``hrdexdb_tare_s`` seconds of the sensor
    # stream) subtracted and floored at zero, the two channels with the largest peak excess,
    # and inspire_dftp closure sign-corrected (Inspire angles are 1000 open, 0 closed).
    hrdexdb_force_representation: Literal["legacy", "tared"] = "legacy"
    hrdexdb_tare_s: float = Field(default=1.0, gt=0)
    seed: int


class ResampleConfig(_Strict):
    """Resampling geometry for the force channel."""

    half_window_s: float
    frame_period_s: float
    group_period_s: float
    raw_rate_hz: float
    contact_sigma: float
    # Floor on the contact threshold as a fraction of the grasp scale: tau is at least
    # baseline + contact_min_frac * (grasp - baseline). 0 keeps the pure sigma rule. Needed for
    # tared integer sensors whose free-space sigma is below one raw unit (HRDexDB inspire_dftp).
    contact_min_frac: float = Field(default=0.0, ge=0, lt=1)


class RenderConfig(_Strict):
    """Text rendering of the force block."""

    header: str
    decimals: int
    max_tokens: int
    clip_max: float
    tokenizer_json: str | None = None
    # Prepend one label-free summary line per arm (contact segments, longest contact, peak,
    # hold duration, hold force, late/early hold ratio, low-force hold fraction) to the block.
    summary: bool = False


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


class ForceConfig(_Strict):
    """Force channel configuration (configs/force.yaml)."""

    resample: ResampleConfig
    render: RenderConfig
    events: EventsConfig


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
    # Amazon part (ASIN B0CZ6L5NMM) is identified.
    sensor_confirmed: bool = False
    adc_max_counts: int = 4095
    saturation_counts: int = 4090
    saturation_frac_max: float = 0.05
    flat_variance: float = 0.25
    health_period_s: float = 1.0
    health_max_age_s: float = 3.0
    gripper_open_max: float = 0.1
    calibration_weights_g: list[float] = Field(
        default_factory=lambda: [100.0, 200.0, 500.0, 1000.0, 2000.0]
    )
    calibration_samples: int = 200


M = TypeVar("M", bound=BaseModel)


def load_config(path: Path | str, model: type[M]) -> M:
    """Read the YAML file at ``path`` and validate it into ``model``."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return model.model_validate(raw)
