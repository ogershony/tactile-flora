"""Converting native source labels to FLORA Label objects (origin="source").

Owns: one pure mapper per source, following the label-mapping table of the spec, and the
dispatcher ``map_native_labels``. Raw native values are read from ``episode.meta``:

- phantom: ``success`` (bool), ``task_folder`` (str, ``*_fail`` means failure), ``object`` (str)
  and optionally ``peak_force_p99`` (float, the object's success-set 99th percentile of F').
- rh20t: ``rating`` (int 0-9) and ``object_moved_toward_goal`` (bool, default False).
- reassemble: ``action_success`` (list[bool]) and ``anomalies`` (list of {t0, t1, text}).
- airoa: ``success`` (bool), ``failure_cause`` (str | None) and ``failed_segments`` (list of
  {t0, t1, cause}).
- sharpadex: nothing (labels come from pseudo-labeling and programmatic events).

A key that is absent yields no label of that kind. Mappers never filter episodes and never use
force thresholds except the PHANTOM dense rule, which needs normalized features.
Must never: read raw datasets or run on Baxter data (its labels are human).
"""

from __future__ import annotations

from typing import Any

from tactile_flora.schema import Episode, Label, ResampledForce, Span


def rh20t_outcome(rating: int, moved_toward_goal: bool) -> int:
    """RH20T rating to outcome: 0 -> 1; 1 -> 2 (3 if the object moved toward the goal); 2-9 -> 5."""
    if rating == 0:
        return 1
    if rating == 1:
        return 3 if moved_toward_goal else 2
    if 2 <= rating <= 9:
        return 5
    raise ValueError(f"RH20T rating must be in 0..9, got {rating}")


def _outcome(score: int, spans: list[Span] | None = None) -> Label:
    return Label(kind="outcome", origin="source", outcome=score, spans=spans or [])


def _dense(spans: list[Span]) -> Label:
    return Label(kind="dense", origin="source", spans=spans)


def _failure_spans(items: list[dict[str, Any]], text_key: str) -> list[Span]:
    return [
        Span(t0=float(d["t0"]), t1=float(d["t1"]), quality=1, rationale=str(d.get(text_key, "")))
        for d in items
    ]


def map_phantom(
    episode: Episode, *, features: ResampledForce | None, p99: float | None
) -> list[Label]:
    """PHANTOM: success -> 5, ``*_fail`` -> 1; dense failure from the first F' over the p99."""
    m = episode.meta
    labels: list[Label] = []
    failed = str(m.get("task_folder", "")).endswith("_fail") or m.get("success") is False
    if "success" in m or "task_folder" in m:
        labels.append(_outcome(1 if failed else 5))
    p99 = p99 if p99 is not None else m.get("peak_force_p99")
    if features is not None and p99 is not None:
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
    return [_outcome(rh20t_outcome(int(episode.meta["rating"]), moved))]


def map_reassemble(episode: Episode) -> list[Label]:
    """REASSEMBLE: any failed action -> 1 else 5; anomaly annotations -> failure spans."""
    m = episode.meta
    labels: list[Label] = []
    actions = m.get("action_success")
    if actions:
        labels.append(_outcome(5 if all(actions) else 1))
    if "anomalies" in m:
        labels.append(_dense(_failure_spans(list(m["anomalies"]), "text")))
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
    """Dispatch to the per-source mapper (``features`` and ``phantom_p99`` are PHANTOM only)."""
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
