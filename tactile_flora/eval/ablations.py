"""Force-mode transforms (real, omitted, zero, shuffled, video_blank) on resampled features.

Owns: apply_force_mode and AblatedForce. All functions are pure.
Must never: render text or load models.

Features passed in are expected to be normalized. ``video_blank`` leaves the features unchanged
and sets a flag that the prompt builder reads to replace frames with mid-gray. ``zero`` keeps
the arm structure but sets every value (F, D, dF, g, raw groups) to 0 and c to 0.
``shuffled`` takes a donor from ``pool`` (same channel, at least as many frames, not equal to
``features``) chosen with random.Random(seed), keeps the target's frame_times, and truncates.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pydantic import BaseModel

from tactile_flora.config import ForceMode
from tactile_flora.schema import Arm, ArmFeatures, ResampledForce


class AblatedForce(BaseModel):
    """Result of an ablation: features to render (None means no block) and the blank flag."""

    features: ResampledForce | None
    video_blank: bool = False


def _zero(features: ResampledForce) -> ResampledForce:
    arms: dict[Arm, ArmFeatures] = {}
    for arm, af in features.arms.items():
        n = len(af.F)
        arms[arm] = ArmFeatures(
            F=[0.0] * n,
            delta=None if af.delta is None else [0.0] * n,
            dF=[0.0] * n,
            g=None if af.g is None else [0.0] * n,
            c=[0] * n,
            raw_groups=[[0.0] * len(r) for r in af.raw_groups],
        )
    return features.model_copy(update={"arms": arms})


def _shuffle(features: ResampledForce, pool: Sequence[ResampledForce], seed: int) -> ResampledForce:
    n = len(features.frame_times)
    donors = [
        p
        for p in pool
        if p != features
        and p.channel == features.channel
        and len(p.frame_times) >= n
        and set(p.arms) >= set(features.arms)
    ]
    if not donors:
        raise ValueError("shuffled mode needs a different pool element with matching channel")
    donor = random.Random(seed).choice(donors)
    arms: dict[Arm, ArmFeatures] = {}
    for arm in features.arms:
        d = donor.arms[arm]
        arms[arm] = ArmFeatures(
            F=d.F[:n],
            delta=None if d.delta is None else d.delta[:n],
            dF=d.dF[:n],
            g=None if d.g is None else d.g[:n],
            c=None if d.c is None else d.c[:n],
            raw_groups=d.raw_groups[: (n + 1) // 2],
        )
    return features.model_copy(update={"arms": arms})


def apply_force_mode(
    features: ResampledForce,
    mode: ForceMode,
    *,
    pool: Sequence[ResampledForce] = (),
    seed: int,
) -> AblatedForce:
    """Apply ``mode`` to ``features`` and return features plus the video_blank flag."""
    if mode == "real":
        return AblatedForce(features=features)
    if mode == "omitted":
        return AblatedForce(features=None)
    if mode == "zero":
        return AblatedForce(features=_zero(features))
    if mode == "shuffled":
        return AblatedForce(features=_shuffle(features, pool, seed))
    return AblatedForce(features=features, video_blank=True)
