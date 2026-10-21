"""Build an episode's Windows (merged labels, rendered force) for verification and the manifest.

Owns: episode_windows, the glue between labels/ and force/ that verification needs.
Must never: write files, change labels beyond labels.merge, or run checks.
"""

from __future__ import annotations

from collections.abc import Sequence

from tactile_flora.config import ForceConfig
from tactile_flora.force import normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.labels import windows as W
from tactile_flora.labels.merge import merge_labels
from tactile_flora.schema import Episode, Split, Window


def _lines(
    episode: Episode, consts: normalize.NormConstants, cfg: ForceConfig, times: Sequence[float]
) -> list[str]:
    feats = resample(episode.force, episode.gripper, times, cfg.resample)
    norm = normalize.apply(feats, consts, cfg.render)
    return render.render_lines(norm, times, episode.force.channel)


def episode_windows(
    episode: Episode,
    cfg: ForceConfig,
    consts: normalize.NormConstants | None,
    split: Split,
    *,
    with_force: bool = True,
) -> list[Window]:
    """Return the episode's dense and outcome windows with force lines rendered from ``consts``.

    With ``with_force`` False or no constants the force lines are empty (the control condition).
    """
    merged = merge_labels(episode.labels)
    if not with_force or consts is None:
        return W.make_windows(episode, None, merged, split)
    dense = {
        a: _lines(episode, consts, cfg, W.dense_frame_times(a))
        for a in W.dense_anchors(episode.duration_s)
    }
    cam = episode.cameras.get("scene")
    outcome: list[str] = []
    if cam is not None:
        idx = W.outcome_frame_indices(cam.n_frames, cam.fps)
        outcome = _lines(episode, consts, cfg, [i / cam.fps for i in idx])
    return W.make_windows(episode, dense, merged, split, outcome_force_lines=outcome)
