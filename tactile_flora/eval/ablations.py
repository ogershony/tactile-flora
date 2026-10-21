"""Force-mode transforms (real, omitted, zero, shuffled, video_blank) on resampled features.

Owns: apply_force_mode and AblatedForce. All functions are pure.
Must never: render text or load models.

Features passed in are expected to be normalized. ``video_blank`` leaves the features unchanged
and sets a flag that the prompt builder reads to replace frames with mid-gray. ``zero`` keeps
the arm structure but sets every value (F, D, dF, g, raw groups) to 0 and c to 0.
``shuffled`` takes a donor from ``pool`` chosen with random.Random(seed), keeps the target's
frame_times, and truncates. Pool entries are ``PoolEntry`` (features plus source / episode /
prompt_kind identity). When the caller passes ``identity`` (the target's own PoolEntry key) a
donor must come from a DIFFERENT episode of the SAME source and the SAME prompt_kind, with the
same channel and at least as many frames (donors of exactly equal length are preferred).
Without ``identity`` (legacy callers) bare ResampledForce entries are accepted and any element
different from ``features`` with a matching channel qualifies.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pydantic import BaseModel

from tactile_flora.config import ForceMode
from tactile_flora.schema import Arm, ArmFeatures, ResampledForce


class PoolEntry(BaseModel):
    """A shuffle donor candidate: normalized features plus where they came from."""

    features: ResampledForce
    source: str
    episode_id: str
    prompt_kind: str


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


def _shuffle(
    features: ResampledForce,
    pool: Sequence[ResampledForce | PoolEntry],
    seed: int,
    identity: PoolEntry | None,
) -> ResampledForce:
    n = len(features.frame_times)
    donors: list[ResampledForce] = []
    for p in pool:
        if identity is not None:
            if not isinstance(p, PoolEntry):
                continue  # no identity, so "different episode of the same source" is unverifiable
            if (
                p.source != identity.source
                or p.episode_id == identity.episode_id
                or p.prompt_kind != identity.prompt_kind
            ):
                continue
        cand = p.features if isinstance(p, PoolEntry) else p
        if (
            cand != features
            and cand.channel == features.channel
            and len(cand.frame_times) >= n
            and set(cand.arms) >= set(features.arms)
        ):
            donors.append(cand)
    exact = [d for d in donors if len(d.frame_times) == n]
    donors = exact or donors
    if not donors:
        raise ValueError(
            "shuffled mode needs a donor from a different episode of the same source, "
            "same prompt kind and channel"
        )
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
    pool: Sequence[ResampledForce | PoolEntry] = (),
    seed: int,
    identity: PoolEntry | None = None,
) -> AblatedForce:
    """Apply ``mode`` to ``features`` and return features plus the video_blank flag.

    ``identity`` (only its source, episode_id and prompt_kind are read) enables the strict
    same-source / different-episode / same-kind donor rule of the shuffled mode.
    """
    if mode == "real":
        return AblatedForce(features=features)
    if mode == "omitted":
        return AblatedForce(features=None)
    if mode == "zero":
        return AblatedForce(features=_zero(features))
    if mode == "shuffled":
        return AblatedForce(features=_shuffle(features, pool, seed, identity))
    return AblatedForce(features=features, video_blank=True)
