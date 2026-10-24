"""Converting native source labels to FLORA Label objects (origin="source").

Owns: one pure mapper per source, following the label-mapping table of the spec, and the
dispatcher ``map_native_labels``. Raw native values are read from ``episode.meta``:

- phantom: ``success`` (bool), ``task_folder`` (str, ``*_fail`` means failure), ``failure_demo``
  (bool, optional), ``object`` (str) and optionally ``peak_force_p99`` (float). The per-object
  success-set statistics of F' are COMPUTED by ``phantom_success_stats`` (called once per source
  by the verify pipeline) and passed in as ``phantom_p99`` (99th percentile of every success
  frame) and ``phantom_peak_p5`` (5th percentile of the per-episode peak). A failure demo gets
  one failure span: the crush rule (first frame of F' above p99 to the end, "excessive force on
  the <object>") or, when its peak F' is below the success set's peak p5 instead, the
  under-grasp rule (first contact to the end, "insufficient grip force on the <object>", the
  too-weak class; the whole episode if it never made contact). The outcome label carries the
  same span, so the episode-level rationale names the same failure class.
- rh20t: ``rating`` (int -1..9; -1 means unrated and gives no label) and
  ``object_moved_toward_goal`` (bool). The latter cannot be derived from the recorded data: it is
  an OPERATOR-PROVIDED annotation in meta and defaults to False, so a rating of 1 maps to 2 unless
  an operator marks the object as moved toward the goal.
- reassemble: ``action_success`` (list[bool]) and ``segments`` (list of {t0, t1, text, success});
  dense failure spans come from the failed segments (there are no anomaly annotations).
- airoa: ``success`` (bool), ``failure_cause`` (str | None) and ``failed_segments`` (list of
  {t0, t1, cause}).
- sharpadex: nothing (labels come from pseudo-labeling and programmatic events).

A key that is absent yields no label of that kind. Mappers never filter episodes and never use
force thresholds except the PHANTOM dense rule, which needs normalized features.
Must never: read raw datasets or run on Baxter data (its labels are human).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from tactile_flora.schema import Episode, Label, ResampledForce, Span

# Registered sources that are evaluation-only: their labels are their own (human or native) and
# never get mapped, programmatic or pseudo labels added.
EVAL_ONLY_SOURCES: frozenset[str] = frozenset({"baxter", "hrdexdb", "softvtbench"})


# Training sources with a mapper in ``map_native_labels``.
MAPPED_SOURCES: frozenset[str] = frozenset({"phantom", "rh20t", "reassemble", "airoa", "sharpadex"})


def rh20t_outcome(rating: int, moved_toward_goal: bool) -> int | None:
    """RH20T rating to outcome: 0 -> 1; 1 -> 2 (3 if the object moved toward the goal); 2-9 -> 5.

    -1 is "unrated" and gives None (no label).
    """
    if rating == -1:
        return None
    if rating == 0:
        return 1
    if rating == 1:
        return 3 if moved_toward_goal else 2
    if 2 <= rating <= 9:
        return 5
    raise ValueError(f"RH20T rating must be in -1..9, got {rating}")


def _outcome(score: int, spans: list[Span] | None = None) -> Label:
    return Label(kind="outcome", origin="source", outcome=score, spans=spans or [])


def _dense(spans: list[Span]) -> Label:
    return Label(kind="dense", origin="source", spans=spans)


def _failure_spans(items: list[dict[str, Any]], text_key: str) -> list[Span]:
    return [
        Span(t0=float(d["t0"]), t1=float(d["t1"]), quality=1, rationale=str(d.get(text_key, "")))
        for d in items
    ]


def _phantom_failed(m: dict[str, Any]) -> bool:
    return (
        str(m.get("task_folder", "")).endswith("_fail")
        or m.get("success") is False
        or bool(m.get("failure_demo", False))
    )


PHANTOM_CRUSH_RATIONALE = "excessive force on the {object}"
PHANTOM_UNDERGRASP_RATIONALE = "insufficient grip force on the {object}"
PHANTOM_PEAK_PERCENTILE = 5.0


def phantom_success_stats(
    episodes_features: Iterable[tuple[Episode, ResampledForce]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-object (p99 of F', p5 of per-episode peak F') over the SUCCESS-set episodes.

    ``episodes_features`` pairs each episode with its NORMALIZED features; failure demos are
    skipped. The first mapping pools every finite F' of every arm per ``meta["object"]`` and
    takes the 99th percentile (the crush threshold); the second takes each success episode's
    peak F' over its arms and the 5th percentile of those peaks (the under-grasp threshold).
    """
    pooled: dict[str, list[np.ndarray]] = {}
    peaks: dict[str, list[float]] = {}
    for ep, feats in episodes_features:
        if _phantom_failed(ep.meta):
            continue
        obj = str(ep.meta.get("object", ""))
        parts = pooled.setdefault(obj, [])
        for af in feats.arms.values():
            f = np.asarray(af.F, dtype=np.float64)
            parts.append(f[np.isfinite(f)])
        peak = _peak(feats)
        if peak is not None:
            peaks.setdefault(obj, []).append(peak)
    p99: dict[str, float] = {}
    for obj, parts in pooled.items():
        allf = np.concatenate(parts) if parts else np.empty(0)
        if allf.size:
            p99[obj] = float(np.percentile(allf, 99))
    peak_p5 = {obj: float(np.percentile(v, PHANTOM_PEAK_PERCENTILE)) for obj, v in peaks.items()}
    return p99, peak_p5


def phantom_success_p99(
    episodes_features: Iterable[tuple[Episode, ResampledForce]],
) -> dict[str, float]:
    """Per-object 99th percentile of normalized F' over the success set (see the stats)."""
    return phantom_success_stats(episodes_features)[0]


def _peak(feats: ResampledForce) -> float | None:
    """Largest finite F' over every arm and frame, None when there is none."""
    vals = [
        float(np.nanmax(f))
        for af in feats.arms.values()
        if (f := np.asarray(af.F, dtype=np.float64)).size and np.isfinite(f).any()
    ]
    return max(vals) if vals else None


def _first_contact(feats: ResampledForce) -> float | None:
    """Time of the first frame whose contact flag is set on any arm (None without one)."""
    for i, t in enumerate(feats.frame_times):
        if any(af.c is not None and af.c[i] == 1 for af in feats.arms.values()):
            return t
    return None


def _phantom_failure_span(
    episode: Episode, features: ResampledForce, p99: float | None, peak_p5: float | None
) -> Span | None:
    """Return the crush span (first F' over p99 to the end), else the under-grasp span or None."""
    obj = episode.meta.get("object", "object")
    if p99 is not None:
        for i, t in enumerate(features.frame_times):
            if any(af.F[i] > p99 for af in features.arms.values()):
                why = PHANTOM_CRUSH_RATIONALE.format(object=obj)
                return Span(t0=t, t1=episode.duration_s, quality=1, rationale=why)
    if peak_p5 is not None:
        peak = _peak(features)
        if peak is not None and peak < peak_p5:
            t0 = _first_contact(features)
            why = PHANTOM_UNDERGRASP_RATIONALE.format(object=obj)
            t0 = 0.0 if t0 is None else t0
            return Span(t0=t0, t1=episode.duration_s, quality=1, rationale=why)
    return None


def map_phantom(
    episode: Episode,
    *,
    features: ResampledForce | None,
    p99: float | None,
    peak_p5: float | None = None,
) -> list[Label]:
    """PHANTOM: success -> 5, failure -> 1; failure demos get one failure span (module docstring).

    The span is only emitted for failure demos (``meta["failure_demo"]``, success False or a
    ``*_fail`` folder): the crush rule first (first F' over ``p99`` to the end), else the
    under-grasp rule (peak F' below ``peak_p5``: first contact to the end). It is attached to
    the dense label and, as the rationale, to the outcome label. Successful demos get an outcome
    label only.
    """
    m = episode.meta
    labels: list[Label] = []
    failed = _phantom_failed(m)
    p99 = p99 if p99 is not None else m.get("peak_force_p99")
    span = (
        _phantom_failure_span(episode, features, p99, peak_p5)
        if failed and features is not None
        else None
    )
    if "success" in m or "task_folder" in m or "failure_demo" in m:
        labels.append(_outcome(1 if failed else 5, [span] if span is not None else None))
    if span is not None:
        labels.append(_dense([span]))
    return labels


def map_rh20t(episode: Episode) -> list[Label]:
    """RH20T: outcome from the rating rule (see ``rh20t_outcome``); no dense label."""
    if "rating" not in episode.meta:
        return []
    moved = bool(episode.meta.get("object_moved_toward_goal", False))
    score = rh20t_outcome(int(episode.meta["rating"]), moved)
    return [] if score is None else [_outcome(score)]


def map_reassemble(episode: Episode) -> list[Label]:
    """REASSEMBLE: any failed action -> 1 else 5; failed segments -> dense failure spans.

    The rationale of a span is the segment text. ``meta["segments"]`` items need t0, t1 and
    success; segments without a usable interval are skipped.
    """
    m = episode.meta
    labels: list[Label] = []
    actions = m.get("action_success")
    if actions:
        labels.append(_outcome(5 if all(actions) else 1))
    failed = [
        s
        for s in m.get("segments", [])
        if not s.get("success", True) and "t0" in s and "t1" in s and s["t1"] >= s["t0"]
    ]
    if failed:
        labels.append(_dense(_failure_spans(failed, "text")))
    return labels


def map_airoa(episode: Episode) -> list[Label]:
    """AIRoA: episode flag -> 5 or 1 (failure cause kept as the outcome rationale span)."""
    m = episode.meta
    labels: list[Label] = []
    if "success" in m:
        ok = bool(m["success"])
        cause = m.get("failure_cause")
        spans = (
            [Span(t0=0.0, t1=episode.duration_s, quality=1, rationale=str(cause))]
            if not ok and cause
            else []
        )
        labels.append(_outcome(5 if ok else 1, spans))
    if "failed_segments" in m:
        labels.append(_dense(_failure_spans(list(m["failed_segments"]), "cause")))
    return labels


def map_native_labels(
    source: str,
    episode: Episode,
    *,
    features: ResampledForce | None = None,
    phantom_p99: float | None = None,
    phantom_peak_p5: float | None = None,
) -> list[Label]:
    """Dispatch to the per-source mapper (``features`` and ``phantom_*`` stats are PHANTOM only).

    Eval-only sources (EVAL_ONLY_SOURCES) are a no-op; an unknown name still raises.
    """
    if source in EVAL_ONLY_SOURCES:
        return []
    if source == "phantom":
        return map_phantom(episode, features=features, p99=phantom_p99, peak_p5=phantom_peak_p5)
    if source == "rh20t":
        return map_rh20t(episode)
    if source == "reassemble":
        return map_reassemble(episode)
    if source == "airoa":
        return map_airoa(episode)
    if source == "sharpadex":
        return []
    raise ValueError(f"no native label mapping for source {source!r}")
