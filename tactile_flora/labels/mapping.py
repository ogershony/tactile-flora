"""Converting native source labels to FLORA Label objects (origin="source").

Owns: one pure mapper per source, following the label-mapping table of the spec, and the
dispatcher ``map_native_labels``. Raw native values are read from ``episode.meta``:

- phantom: ``success`` (bool), ``task_folder`` (str, ``*_fail`` means failure), ``failure_demo``
  (bool, optional), ``object`` (str) and optionally ``peak_force_p99`` (float). The per-object
  success-set 99th percentile of F' is COMPUTED by ``phantom_success_p99`` (called once per source
  by the verify pipeline) and passed in as ``phantom_p99``.
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


def phantom_success_p99(
    episodes_features: Iterable[tuple[Episode, ResampledForce]],
) -> dict[str, float]:
    """Per-object 99th percentile of normalized F' over the SUCCESS-set episodes of a source.

    ``episodes_features`` pairs each episode with its NORMALIZED features; failure demos are
    skipped, every finite F' of every arm is pooled per ``meta["object"]``.
    """
    pooled: dict[str, list[np.ndarray]] = {}
    for ep, feats in episodes_features:
        if _phantom_failed(ep.meta):
            continue
        obj = str(ep.meta.get("object", ""))
        for af in feats.arms.values():
            f = np.asarray(af.F, dtype=np.float64)
            pooled.setdefault(obj, []).append(f[np.isfinite(f)])
    out: dict[str, float] = {}
    for obj, parts in pooled.items():
        allf = np.concatenate(parts) if parts else np.empty(0)
        if allf.size:
            out[obj] = float(np.percentile(allf, 99))
    return out


def map_phantom(
    episode: Episode, *, features: ResampledForce | None, p99: float | None
) -> list[Label]:
    """PHANTOM: success -> 5, failure -> 1; failure demos get a span from the first F' over p99.

    The p99 failure span is only emitted for failure demos (``meta["failure_demo"]``, success
    False or a ``*_fail`` folder); successful demos get an outcome label only.
    """
    m = episode.meta
    labels: list[Label] = []
    failed = _phantom_failed(m)
    if "success" in m or "task_folder" in m or "failure_demo" in m:
        labels.append(_outcome(1 if failed else 5))
    p99 = p99 if p99 is not None else m.get("peak_force_p99")
    if failed and features is not None and p99 is not None:
        hit: float | None = None
        for i, t in enumerate(features.frame_times):
            if any(af.F[i] > p99 for af in features.arms.values()):
                hit = t
                break
        if hit is not None:
            why = f"excessive force on the {m.get('object', 'object')}"
            labels.append(_dense([Span(t0=hit, t1=episode.duration_s, quality=1, rationale=why)]))
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
) -> list[Label]:
    """Dispatch to the per-source mapper (``features`` and ``phantom_p99`` are PHANTOM only).

    Eval-only sources (EVAL_ONLY_SOURCES) are a no-op; an unknown name still raises.
    """
    if source in EVAL_ONLY_SOURCES:
        return []
    if source == "phantom":
        return map_phantom(episode, features=features, p99=phantom_p99)
    if source == "rh20t":
        return map_rh20t(episode)
    if source == "reassemble":
        return map_reassemble(episode)
    if source == "airoa":
        return map_airoa(episode)
    if source == "sharpadex":
        return []
    raise ValueError(f"no native label mapping for source {source!r}")
