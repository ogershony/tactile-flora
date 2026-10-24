"""Build an episode's Windows (merged labels, rendered force) for verification and the manifest.

Owns: episode_windows, the glue between labels/ and force/ that verification needs, the sync
shift helpers (``shifted_frame_times``, ``resample_shifted``), the per-source label inputs
(``SourceInputs`` / ``build_source_inputs``) and ``assemble_labels``.
Must never: write files, change labels beyond labels.merge, or run checks.

Label assembly (finding "wire labels"): an episode's labels are
``episode.labels`` + ``adapter.native_labels(id)`` + ``labels.mapping.map_native_labels`` (from the
raw meta and the normalized features) + ``labels.events.detect_all`` (programmatic) + pseudo labels
read back from data/verified/<source>/pseudo.parquet, then ``merge_labels``. Eval-only sources
(mapping.EVAL_ONLY_SOURCES) get their own labels only: no mapping, events or pseudo.

pseudo.parquet contract (written by ``labels.pseudo.run_pseudo``): one row per dense window with
columns episode_id, anchor_s, raw_text, parse_failed, label_json, origin, seed. ``label_json`` is a
``Label`` (kind="dense", origin="pseudo") whose spans are already on the episode clock.

Sync shift: ``ManifestRow.shift_s`` is the clock correction added to the force timestamps, so the
force at frame time t is the raw force at ``shifted_frame_times([t], shift_s)[0] = t - shift_s``.
check_sync reports a lag (positive = force trails the gripper), so ``shift_s = -lag``: the force
is read ``lag`` later in raw time. The gripper and the video are NOT shifted, so use
``resample_shifted`` (not a plain ``resample`` on shifted times, which would shift the gripper
too). Windows built during ``run_verify`` are unshifted (the sync check has not run yet); the
manifest and training paths apply the shift. PHANTOM p99 is always computed unshifted.

Normalization constants are applied per episode with ``normalize.norm_group(episode)`` (PHANTOM
object family, "all" elsewhere); the manifest row records that group.

``meta["prompt_kinds"]`` (``prompt_kinds(episode)``) restricts which windows an episode yields:
segment episodes carry ``["outcome"]`` and get neither dense windows nor programmatic events.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from tactile_flora.config import ForceConfig, ResampleConfig
from tactile_flora.force import normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.labels import events as E
from tactile_flora.labels import mapping
from tactile_flora.labels import windows as W
from tactile_flora.labels.merge import merge_labels
from tactile_flora.schema import (
    Arm,
    ArmFeatures,
    ArmForce,
    CheckResult,
    Episode,
    ForceStream,
    Label,
    ResampledForce,
    Split,
    Window,
)
from tactile_flora.verify import manifest as manifest_mod


def shifted_frame_times(frame_times: Sequence[float], shift_s: float) -> list[float]:
    """Return the raw-force query times ``t - shift_s`` for episode-clock ``frame_times``."""
    return [float(t) - shift_s for t in frame_times]


def resample_shifted(
    episode: Episode, frame_times: Sequence[float], cfg: ResampleConfig, shift_s: float = 0.0
) -> ResampledForce:
    """Resample ``episode`` at ``frame_times`` with force read at ``t - shift_s``.

    The gripper is sampled at the unshifted times and ``frame_times`` of the result are the
    episode-clock times, so rendering and labels stay on the episode clock.
    """
    times = [float(t) for t in frame_times]
    if shift_s == 0.0:
        return resample(episode.force, episode.gripper, times, cfg)
    feats = resample(episode.force, episode.gripper, shifted_frame_times(times, shift_s), cfg)
    arms: dict[Arm, ArmFeatures] = dict(feats.arms)
    if episode.gripper is not None:
        unshifted = resample(episode.force, episode.gripper, times, cfg)
        arms = {a: af.model_copy(update={"g": unshifted.arms[a].g}) for a, af in arms.items()}
    return ResampledForce(channel=feats.channel, frame_times=times, arms=arms)


class LabelSource(Protocol):
    """The part of a SourceAdapter that label assembly uses."""

    name: str

    def load(self, episode_id: str) -> Episode:
        """Load an episode."""
        ...

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return the source's own labels."""
        ...


@dataclass
class SourceInputs:
    """Everything beyond the Episode that window building needs for one source."""

    adapter: LabelSource
    phantom_p99: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    phantom_peak_p5: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    pseudo: dict[str, list[Label]] = field(default_factory=lambda: dict[str, list[Label]]())
    results: dict[str, dict[str, CheckResult]] = field(
        default_factory=lambda: dict[str, dict[str, CheckResult]]()
    )

    @property
    def source(self) -> str:
        """The adapter's source name."""
        return self.adapter.name

    def shift_of(self, episode_id: str) -> float:
        """Clock correction (shift_s) for an episode from its saved sync check."""
        return manifest_mod.shift_of(self.results.get(episode_id, {}))


ALL_PROMPT_KINDS: frozenset[str] = frozenset({"dense", "outcome"})


def prompt_kinds(episode: Episode) -> frozenset[str]:
    """Prompt kinds an episode may produce: ``meta["prompt_kinds"]`` when set, else both.

    Segment episodes (reassemble_lerobot segment mode) set ``["outcome"]``: no dense windows
    are built and no programmatic event labels are detected for them.
    """
    kinds = episode.meta.get("prompt_kinds")
    return frozenset(str(k) for k in kinds) if kinds else ALL_PROMPT_KINDS


def _frame_grid(duration_s: float, period: float) -> list[float]:
    return [k * period for k in range(int(duration_s / period) + 1)]


def read_pseudo(path: Path) -> dict[str, list[Label]]:
    """Read pseudo.parquet (see the module docstring) into Labels per episode id."""
    if not path.exists():
        return {}
    import pandas as pd

    out: dict[str, list[Label]] = {}
    for row in pd.read_parquet(path).sort_values(["episode_id", "anchor_s"]).to_dict("records"):
        lab = Label.model_validate_json(str(row["label_json"]))
        out.setdefault(str(row["episode_id"]), []).append(lab)
    return out


def build_source_inputs(
    adapter: LabelSource,
    force_cfg: ForceConfig,
    consts: normalize.NormConstants | None,
    verified_dir: Path,
    ids: Sequence[str],
    results: dict[str, dict[str, CheckResult]] | None = None,
) -> SourceInputs:
    """Gather per-source label inputs: PHANTOM per-object stats, pseudo labels and check results.

    The PHANTOM per-object p99 of F' and 5th percentile of per-episode peak F' are computed once
    here over the success-set episodes (features normalized with each episode's norm_group).
    """
    inputs = SourceInputs(
        adapter=adapter,
        pseudo={}
        if adapter.name in mapping.EVAL_ONLY_SOURCES
        else read_pseudo(verified_dir / "pseudo.parquet"),
        results=results or {},
    )
    if adapter.name == "phantom" and consts is not None:

        def _pairs():
            for eid in ids:
                ep = adapter.load(eid)
                times = _frame_grid(ep.duration_s, force_cfg.resample.frame_period_s)
                feats = resample(ep.force, ep.gripper, times, force_cfg.resample)
                yield (
                    ep,
                    normalize.apply(
                        feats, consts, force_cfg.render, group=normalize.norm_group(ep)
                    ),
                )

        inputs.phantom_p99, inputs.phantom_peak_p5 = mapping.phantom_success_stats(_pairs())
    return inputs


def prepare_episode(episode: Episode, inputs: SourceInputs | None) -> Episode | None:
    """Apply drop_channel: remove failed arms from the ForceStream (None if none are left).

    See ``manifest.dropped_arms`` for how the failed arm is read from the check detail.
    """
    if inputs is None:
        return episode
    res = inputs.results.get(episode.episode_id)
    if not res:
        return episode
    drop = manifest_mod.dropped_arms(res, list(episode.force.arms))
    if not drop:
        return episode
    keep: dict[Arm, ArmForce] = {a: d for a, d in episode.force.arms.items() if a not in drop}
    if not keep:
        return None
    stream = ForceStream(
        channel=episode.force.channel, arms=keep, rate_hz=episode.force.rate_hz, t=episode.force.t
    )
    return episode.model_copy(update={"force": stream})


def assemble_labels(
    episode: Episode,
    cfg: ForceConfig,
    consts: normalize.NormConstants | None,
    inputs: SourceInputs,
    shift_s: float = 0.0,
) -> list[Label]:
    """Return the unmerged labels of an episode (see the module docstring for the recipe)."""
    labels = list(episode.labels)
    for lab in inputs.adapter.native_labels(episode.episode_id):
        if lab not in labels:
            labels.append(lab)
    if episode.source in mapping.EVAL_ONLY_SOURCES:
        return labels
    feats: ResampledForce | None = None
    if consts is not None:
        times = _frame_grid(episode.duration_s, cfg.resample.frame_period_s)
        feats = normalize.apply(
            resample_shifted(episode, times, cfg.resample, shift_s),
            consts,
            cfg.render,
            group=normalize.norm_group(episode),
        )
    obj = str(episode.meta.get("object", ""))
    if episode.source in mapping.MAPPED_SOURCES:  # other sources (tests) keep native labels only
        labels += mapping.map_native_labels(
            episode.source,
            episode,
            features=feats,
            phantom_p99=inputs.phantom_p99.get(obj),
            phantom_peak_p5=inputs.phantom_peak_p5.get(obj),
        )
    outcomes = [lab.outcome for lab in labels if lab.kind == "outcome" and lab.outcome]
    task_complete = bool(outcomes) and merge_outcome(labels) >= 4
    if feats is not None and "dense" in prompt_kinds(episode):
        labels += E.detect_all(feats, cfg.events, task_complete=task_complete)
    labels += inputs.pseudo.get(episode.episode_id, [])
    return labels


def merge_outcome(labels: Sequence[Label]) -> int:
    """Outcome score the merge would pick (0 when there is none)."""
    merged = merge_labels(list(labels))
    return next((lab.outcome or 0 for lab in merged if lab.kind == "outcome"), 0)


def _lines(
    episode: Episode,
    consts: normalize.NormConstants,
    cfg: ForceConfig,
    times: Sequence[float],
    shift_s: float,
) -> list[str]:
    feats = resample_shifted(episode, times, cfg.resample, shift_s)
    norm = normalize.apply(feats, consts, cfg.render, group=normalize.norm_group(episode))
    return render.render_lines(norm, list(times), episode.force.channel, cfg=cfg.render)


def episode_windows(
    episode: Episode,
    cfg: ForceConfig,
    consts: normalize.NormConstants | None,
    split: Split,
    *,
    with_force: bool = True,
    inputs: SourceInputs | None = None,
) -> list[Window]:
    """Return the episode's dense and outcome windows with force lines rendered from ``consts``.

    With ``with_force`` False or no constants the force lines are empty (the control condition).
    With ``inputs`` the labels are assembled from native, mapped, programmatic and pseudo labels,
    failed arms are removed and the episode's sync shift is applied; without it only
    ``episode.labels`` are used.
    """
    prepared = prepare_episode(episode, inputs)
    if prepared is None:
        return []
    episode = prepared
    shift_s = inputs.shift_of(episode.episode_id) if inputs is not None else 0.0
    labels = (
        assemble_labels(episode, cfg, consts, inputs, shift_s)
        if inputs is not None
        else episode.labels
    )
    merged = merge_labels(labels)
    kinds = prompt_kinds(episode)
    if not with_force or consts is None:
        wins = W.make_windows(episode, None, merged, split)
        return [w for w in wins if w.prompt_kind in kinds]
    dense = (
        {
            a: _lines(episode, consts, cfg, W.dense_frame_times(a), shift_s)
            for a in W.dense_anchors(episode.duration_s)
        }
        if "dense" in kinds
        else {}
    )
    cam = episode.cameras.get("scene")
    outcome: list[str] = []
    if cam is not None and "outcome" in kinds:
        idx = W.outcome_frame_indices(cam.n_frames, cam.fps)
        outcome = _lines(episode, consts, cfg, [i / cam.fps for i in idx], shift_s)
    wins = W.make_windows(episode, dense, merged, split, outcome_force_lines=outcome)
    return [w for w in wins if w.prompt_kind in kinds]


__all__ = [
    "SourceInputs",
    "assemble_labels",
    "build_source_inputs",
    "episode_windows",
    "prepare_episode",
    "prompt_kinds",
    "read_pseudo",
    "resample_shifted",
    "shifted_frame_times",
]
