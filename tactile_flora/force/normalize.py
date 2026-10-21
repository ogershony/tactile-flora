"""Fitting and applying per-source normalization constants (norm.json, norm_version).

Owns: NormConstants, fit, apply, save, load. The norm_version is a sha256 of the constants.
Must never: fit eval-only sources on training data (callers pass only that source's episodes),
or resample.

Apply maps F to F' = clip((F - baseline) / (grasp - baseline), 0, clip_max); Delta and dF are
divided by the same scale and not clipped. The contact flag is c = 1[F' > tau'] with
tau' = (tau - baseline) / scale, i.e. F > tau in raw units. NaN frames stay NaN in F', Delta
and dF and get c=0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from tactile_flora.config import RenderConfig, ResampleConfig
from tactile_flora.schema import Arm, ArmFeatures, ResampledForce

_GRIPPER_CLOSED = 0.6


class NormError(ValueError):
    """Raised when constants cannot be fitted or loaded."""


class ArmNorm(BaseModel):
    """Constants for one arm, in raw force units."""

    model_config = ConfigDict(extra="forbid")

    baseline: float
    grasp: float
    sigma: float
    tau: float
    scale: float


class NormConstants(BaseModel):
    """Per-source, per-channel normalization constants."""

    model_config = ConfigDict(extra="forbid")

    source: str
    channel: str
    arms: dict[Arm, ArmNorm]

    def version(self) -> str:
        """Return the sha256 hex digest of the canonical JSON of the constants."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _flatten(
    episodes: Iterable[ResampledForce | tuple[ResampledForce, ...]],
) -> Iterable[ResampledForce]:
    for item in episodes:
        if isinstance(item, tuple):
            yield from item
        else:
            yield item


def fit(
    episodes_features: Iterable[ResampledForce | tuple[ResampledForce, ...]],
    cfg: ResampleConfig,
    *,
    source: str,
) -> NormConstants:
    """Fit constants from raw-unit features of one source (tuples are flattened)."""
    free: dict[Arm, list[float]] = {}
    grasp_cands: dict[Arm, list[tuple[float, float]]] = {}
    channel: str | None = None
    for feats in _flatten(episodes_features):
        if channel not in (None, feats.channel):
            raise NormError("mixed channels in one fit")
        channel = feats.channel
        for arm, af in feats.arms.items():
            for i, t in enumerate(feats.frame_times):
                f = af.F[i]
                g = None if af.g is None else af.g[i]
                if np.isnan(f):
                    continue
                if t < cfg.group_period_s and (g is None or g <= _GRIPPER_CLOSED):
                    free.setdefault(arm, []).append(f)
                if g is not None and g > _GRIPPER_CLOSED:
                    grasp_cands.setdefault(arm, []).append((f, g))
    if channel is None or not free:
        raise NormError("no free-space frames to fit baseline")
    arms: dict[Arm, ArmNorm] = {}
    for arm, vals in free.items():
        if len(vals) < 2:
            raise NormError(f"arm {arm}: need >= 2 free-space frames, got {len(vals)}")
        baseline = float(np.median(vals))
        sigma = float(np.std(vals))
        tau = baseline + cfg.contact_sigma * sigma
        held = [f for f, _ in grasp_cands.get(arm, []) if f > tau]
        if not held:
            raise NormError(f"arm {arm}: no grasp frames (g > {_GRIPPER_CLOSED}, F > tau)")
        grasp = float(np.median(held))
        if grasp == baseline:
            raise NormError(f"arm {arm}: grasp equals baseline ({grasp}); cannot normalize")
        arms[arm] = ArmNorm(
            baseline=baseline, grasp=grasp, sigma=sigma, tau=tau, scale=grasp - baseline
        )
    return NormConstants(source=source, channel=channel, arms=arms)


def apply(features: ResampledForce, consts: NormConstants, cfg: RenderConfig) -> ResampledForce:
    """Return normalized features with contact flags and ``norm_version`` set."""
    if features.channel != consts.channel:
        raise NormError(f"channel {features.channel} != constants channel {consts.channel}")
    out: dict[Arm, ArmFeatures] = {}
    for arm, af in features.arms.items():
        if arm not in consts.arms:
            raise NormError(f"no constants for arm {arm}")
        n = consts.arms[arm]
        F = np.asarray(af.F, dtype=np.float64)
        Fn = np.clip((F - n.baseline) / n.scale, 0.0, cfg.clip_max)
        tau_n = (n.tau - n.baseline) / n.scale
        c = [int(v > tau_n) if not np.isnan(v) else 0 for v in Fn]
        out[arm] = ArmFeatures(
            F=[float(v) for v in Fn],
            delta=None if af.delta is None else [d / n.scale for d in af.delta],
            dF=[d / n.scale for d in af.dF],
            g=af.g,
            c=c,
            raw_groups=af.raw_groups,
        )
    return ResampledForce(
        channel=features.channel,
        frame_times=features.frame_times,
        arms=out,
        norm_version=consts.version(),
    )


def save(consts: NormConstants, path: Path | str) -> None:
    """Write constants plus ``norm_version`` to ``path`` (data/verified/<source>/norm.json)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {**consts.model_dump(mode="json"), "norm_version": consts.version()}
    p.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")


def load(path: Path | str) -> NormConstants:
    """Read constants and verify the stored ``norm_version`` against the recomputed hash."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = doc.pop("norm_version", None)
    consts = NormConstants.model_validate(doc)
    if stored != consts.version():
        raise NormError(
            f"norm_version mismatch in {path}: stored {stored}, actual {consts.version()}"
        )
    return consts
