"""Fitting and applying per-source, per-group normalization constants (norm.json, norm_version).

Owns: NormConstants, fit, apply, save, load, norm_group, grasp_mask. The norm_version is a
sha256 of the constants (the whole group mapping).
Must never: fit eval-only sources on training data (callers pass only that source's episodes),
or resample.

Groups: constants are fitted and applied per (source, group). The group of an episode is
``norm_group(episode)``: for PHANTOM the object/task family (the task folder with ``_fail``
stripped, so Carton and Carton_fail share constants), "all" for every other source. A fit over
bare features (no group) yields the single group "all", and ``apply`` with a group that has no
constants falls back to "all" when that group exists (a single-group fit applies to every
episode of its source), else raises NormError. Single-group sources are therefore unchanged.

Grasp frames (the ``grasp`` constant is their median F): frames with g > grasp_g
(``normalize.grasp_g`` in configs/force.yaml, default 0.6) and F > tau. That rule is only
trusted when at least ``normalize.min_group_frac`` (default 0.25) of the group's fitted episodes
contribute such a frame: PHANTOM's median maximum closure is 0.49-0.61 per object, and on the
local subset egg passed g > 0.6 in 6 of 47 episodes, five of them over-squeeze demos, which put
its "median grasp" at the crush force. Below that fraction (or with no such frame at all) the
grasp frames are those with g at or above the group's 75th percentile of g (and F > tau); the
arm's constants then record ``grasp_rule = "p75"`` and ``grasp_g`` = that percentile (otherwise
``grasp_rule = "threshold"`` and ``grasp_g`` = the configured cut). That fallback never raises;
NormError is only raised when no frame has F > tau at all.

Which episodes feed the fit is the caller's choice (verify pass 1 skips episodes whose adapter
set ``meta["exclude_from_norm_fit"]``, PHANTOM's failure demos); the constants are applied to
every episode of the group.

Apply maps F to F' = clip((F - baseline) / (grasp - baseline), 0, clip_max); Delta and dF are
divided by the same scale and not clipped. The contact flag is c = 1[F' > tau'] with
tau' = (tau - baseline) / scale, i.e. F > tau in raw units. NaN frames stay NaN in F', Delta
and dF and get c=0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from tactile_flora.config import RenderConfig, ResampleConfig
from tactile_flora.schema import Arm, ArmFeatures, Episode, ResampledForce

DEFAULT_GROUP = "all"
GRASP_G_DEFAULT = 0.6
MIN_GROUP_FRAC_DEFAULT = 0.25
GRASP_FALLBACK_PERCENTILE = 75.0
GraspRule = Literal["threshold", "p75"]

FitItem = ResampledForce | tuple[ResampledForce, ...] | tuple[str, ResampledForce]


class NormError(ValueError):
    """Raised when constants cannot be fitted or loaded."""


class ArmNorm(BaseModel):
    """Constants for one arm, in raw force units.

    ``grasp_rule`` / ``grasp_g`` record how the grasp frames were chosen: "threshold" means
    g > grasp_g (the configured cut), "p75" means g >= grasp_g where grasp_g is the group's 75th
    percentile of g (the fallback when too few of the group's episodes passed the configured cut).
    ``p99_fprime`` is a fit-time diagnostic: the 99th percentile of the unclipped
    (F - baseline) / scale over every fitted frame of the group, i.e. the tail the clip at
    ``render.clip_max`` and the verification gate ``norm_p99_max`` act on.
    """

    model_config = ConfigDict(extra="forbid")

    baseline: float
    grasp: float
    sigma: float
    tau: float
    scale: float
    grasp_rule: GraspRule = "threshold"
    grasp_g: float = GRASP_G_DEFAULT
    p99_fprime: float | None = None


class NormConstants(BaseModel):
    """Per-source, per-channel normalization constants, one ArmNorm set per group."""

    model_config = ConfigDict(extra="forbid")

    source: str
    channel: str
    groups: dict[str, dict[Arm, ArmNorm]]

    @property
    def arms(self) -> dict[Arm, ArmNorm]:
        """The single-group constants (group "all"); NormError for a grouped fit without it."""
        return self.arms_for(DEFAULT_GROUP)

    def arms_for(self, group: str) -> dict[Arm, ArmNorm]:
        """Constants of ``group``, or of "all" when only a single-group fit exists."""
        if group in self.groups:
            return self.groups[group]
        if DEFAULT_GROUP in self.groups:
            return self.groups[DEFAULT_GROUP]
        raise NormError(f"no constants for group {group!r} (fitted: {sorted(self.groups)})")

    def version(self) -> str:
        """Return the sha256 hex digest of the canonical JSON of the constants."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def norm_group(episode: Episode) -> str:
    """Return the normalization group of an episode: PHANTOM's object/task family, else "all".

    HRDexDB episodes loaded with ``meta["norm_by_hand"]`` (v4, frame-cache mode) are grouped by
    hand (``session``): inspire_dftp and inspire_f1 report force in different, undocumented units.

    PHANTOM: ``meta["task_folder"]`` (``meta["object"]`` if absent) with the ``_fail`` suffix
    removed, so a failure folder shares its object's constants.
    """
    if episode.source == "hrdexdb" and episode.meta.get("norm_by_hand"):
        return str(episode.session)
    if episode.source != "phantom":
        return DEFAULT_GROUP
    family = str(episode.meta.get("task_folder") or episode.meta.get("object") or "")
    return family.removesuffix("_fail") or DEFAULT_GROUP


def _closed(g: np.ndarray, rule: GraspRule, g_cut: float) -> np.ndarray:
    return g > g_cut if rule == "threshold" else g >= g_cut


def grasp_mask(g: Sequence[float] | np.ndarray, n: ArmNorm) -> np.ndarray:
    """Boolean mask of the frames the fit counts as gripper-closed under the arm's grasp rule."""
    return _closed(np.asarray(g, dtype=np.float64), n.grasp_rule, n.grasp_g)


def _grouped(items: Iterable[FitItem]) -> Iterable[tuple[str, ResampledForce]]:
    """Yield (group, features): bare features and feature tuples are group "all"."""
    for item in items:
        if isinstance(item, ResampledForce):
            yield DEFAULT_GROUP, item
            continue
        first = item[0]
        if isinstance(first, str):
            feats = item[1]
            if not isinstance(feats, ResampledForce):
                raise NormError("a (group, features) pair must hold a ResampledForce")
            yield first, feats
            continue
        for feats in item:
            if not isinstance(feats, ResampledForce):
                raise NormError("a features tuple must hold only ResampledForce items")
            yield DEFAULT_GROUP, feats


def _free_samples(
    feats: ResampledForce, af: ArmFeatures, cfg: ResampleConfig, grasp_g: float
) -> list[float]:
    """Raw free-space samples of one arm in one episode (empty if the episode has none).

    Uses ``raw_groups[0]``, the raw 100 Hz slice of the first 0.5 s group (it extends half a
    window past 0.5 s), provided the first frame is inside the first group period and the gripper
    is open (g <= grasp_g) at every frame of group 0. Features built without raw groups fall back
    to the frame medians at t < group_period_s, which understates sigma (documented limitation).
    """
    if not feats.frame_times or feats.frame_times[0] >= cfg.group_period_s:
        return []
    if af.raw_groups and af.raw_groups[0]:
        idx = [i for i in range(min(2, len(feats.frame_times)))]
        if af.g is not None and any(af.g[i] > grasp_g for i in idx):
            return []
        return [v for v in af.raw_groups[0] if not np.isnan(v)]
    out: list[float] = []
    for i, t in enumerate(feats.frame_times):
        g = None if af.g is None else af.g[i]
        if t < cfg.group_period_s and (g is None or g <= grasp_g) and not np.isnan(af.F[i]):
            out.append(af.F[i])
    return out


@dataclass
class _ArmAcc:
    """Per-group, per-arm accumulation over episodes (one (F, g) array pair per episode)."""

    baselines: list[float] = field(default_factory=lambda: list[float]())
    sigmas: list[float] = field(default_factory=lambda: list[float]())
    episodes: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=lambda: list[tuple[np.ndarray, np.ndarray]]()
    )


def _fit_arm(
    acc: _ArmAcc, cfg: ResampleConfig, grasp_g: float, min_group_frac: float, label: str
) -> ArmNorm:
    baseline = float(np.median(acc.baselines))
    sigma = float(np.median(acc.sigmas))
    tau = baseline + cfg.contact_sigma * sigma
    per_ep = [(f, g) for f, g in acc.episodes if f.size]
    force = np.concatenate([f for f, _ in per_ep]) if per_ep else np.empty(0)
    closure = np.concatenate([g for _, g in per_ep]) if per_ep else np.empty(0)
    rule: GraspRule = "threshold"
    g_cut = grasp_g
    contributing = sum(bool((_closed(g, rule, g_cut) & (f > tau)).any()) for f, g in per_ep)
    if per_ep and (contributing == 0 or contributing < min_group_frac * len(per_ep)):
        # documented fallback: too few episodes pass the configured cut (the gripper rarely
        # closes that far on this object, and the few that do are not typical), so take the
        # most-closed quarter of the group's frames instead
        rule, g_cut = "p75", float(np.percentile(closure, GRASP_FALLBACK_PERCENTILE))
    held = force[_closed(closure, rule, g_cut) & (force > tau)] if force.size else force
    if not held.size:
        raise NormError(f"{label}: no grasp frames (F > tau={tau:.3g} while the gripper is closed)")
    grasp = float(np.median(held))
    floor = baseline + cfg.contact_min_frac * (grasp - baseline)
    if cfg.contact_min_frac > 0 and floor > tau:
        tau = floor
        held = force[_closed(closure, rule, g_cut) & (force > tau)]
        if not held.size:
            raise NormError(f"{label}: no grasp frames above the floored tau {tau:.3g}")
        grasp = float(np.median(held))
        sigma = (tau - baseline) / cfg.contact_sigma
    if grasp == baseline:
        raise NormError(f"{label}: grasp equals baseline ({grasp}); cannot normalize")
    scale = grasp - baseline
    return ArmNorm(
        baseline=baseline,
        grasp=grasp,
        sigma=sigma,
        tau=tau,
        scale=scale,
        grasp_rule=rule,
        grasp_g=g_cut,
        p99_fprime=float(np.percentile((force - baseline) / scale, 99)) if force.size else None,
    )


def fit(
    episodes_features: Iterable[FitItem],
    cfg: ResampleConfig,
    *,
    source: str,
    grasp_g: float = GRASP_G_DEFAULT,
    min_group_frac: float = MIN_GROUP_FRAC_DEFAULT,
) -> NormConstants:
    """Fit constants from raw-unit features of one source, per group.

    Items are a ``ResampledForce`` or a tuple of them (group "all", the single-group case) or a
    ``(group, features)`` pair (callers pair each episode's features with
    ``norm_group(episode)``). Free space is the first 0.5 s of every episode with the gripper
    open, taken from the RAW 100 Hz samples in ``ArmFeatures.raw_groups[0]`` (so the features
    alone suffice; no separate stream argument). Per episode and arm the baseline is the median
    and sigma the std of those raw samples; the group constants are the median of the
    per-episode baselines and the median of the per-episode sigmas (robust to between-episode
    baseline spread). Grasp is the median of frame-level F over the grasp frames (module
    docstring: g > ``grasp_g`` and F > tau when at least ``min_group_frac`` of the group's
    episodes have such a frame, else the 75th-percentile fallback). With
    ``cfg.contact_min_frac`` > 0 the threshold is raised to at least that fraction of the
    provisional grasp excess and the grasp median is recomputed above the raised threshold; sigma
    is then reported as (tau - baseline) / contact_sigma so the stored constants stay
    self-consistent (HRDexDB tared integer sensors, origin/main 79c3ea2). The norm_version is the
    sha256 of the resulting constants.
    """
    acc: dict[str, dict[Arm, _ArmAcc]] = {}
    channel: str | None = None
    for group, feats in _grouped(episodes_features):
        if channel not in (None, feats.channel):
            raise NormError("mixed channels in one fit")
        channel = feats.channel
        per_arm = acc.setdefault(group, {})
        for arm, af in feats.arms.items():
            a = per_arm.setdefault(arm, _ArmAcc())
            free = _free_samples(feats, af, cfg, grasp_g)
            if len(free) >= 2:
                a.baselines.append(float(np.median(free)))
                a.sigmas.append(float(np.std(free)))
            if af.g is not None:
                force = np.asarray(af.F, dtype=np.float64)
                closure = np.asarray(af.g, dtype=np.float64)
                ok = ~np.isnan(force)
                a.episodes.append((force[ok], closure[ok]))
    if channel is None:
        raise NormError("no features to fit")
    groups: dict[str, dict[Arm, ArmNorm]] = {}
    for group, per_arm in acc.items():
        fitted: dict[Arm, ArmNorm] = {
            arm: _fit_arm(a, cfg, grasp_g, min_group_frac, f"group {group!r} arm {arm}")
            for arm, a in per_arm.items()
            if a.baselines
        }
        if not fitted:
            raise NormError(f"group {group!r}: no free-space frames to fit baseline")
        groups[group] = fitted
    return NormConstants(source=source, channel=channel, groups=groups)


def apply(
    features: ResampledForce,
    consts: NormConstants,
    cfg: RenderConfig,
    *,
    group: str = DEFAULT_GROUP,
    clip: bool = True,
) -> ResampledForce:
    """Return normalized features with contact flags and ``norm_version`` set.

    ``group`` selects the constants (``consts.arms_for``: the episode's ``norm_group``, falling
    back to "all" for a single-group fit). With ``clip=False`` F' is the unclipped
    (F - baseline) / scale (only for diagnostics such as the normalization check's 99th
    percentile); the lower clip at 0 and upper clip at clip_max are both skipped.
    """
    if features.channel != consts.channel:
        raise NormError(f"channel {features.channel} != constants channel {consts.channel}")
    arms = consts.arms_for(group)
    out: dict[Arm, ArmFeatures] = {}
    for arm, af in features.arms.items():
        if arm not in arms:
            raise NormError(f"no constants for arm {arm} in group {group!r}")
        n = arms[arm]
        F = np.asarray(af.F, dtype=np.float64)
        Fn = (F - n.baseline) / n.scale
        if clip:
            Fn = np.clip(Fn, 0.0, cfg.clip_max)
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
