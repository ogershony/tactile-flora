"""Shared builders for the verification tests."""

from __future__ import annotations

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.force import normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.schema import Episode, VerifyContext, Window
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.synthetic_video import SyntheticFrameReader

FORCE_CFG = load_config("configs/force.yaml", ForceConfig)
TIMES = [i * 0.25 for i in range(10)]


def fit_consts(episodes: list[Episode]) -> normalize.NormConstants:
    """Fit constants over ``episodes`` at 4 Hz frames."""
    feats = [resample(e.force, e.gripper, TIMES, FORCE_CFG.resample) for e in episodes]
    return normalize.fit(feats, FORCE_CFG.resample, source="synthetic")


def outcome_window(
    ep: Episode, consts: normalize.NormConstants, split: str = "train", with_force: bool = True
) -> Window:
    """An outcome window over the episode's 10 frames with rendered force lines."""
    feats = normalize.apply(
        resample(ep.force, ep.gripper, TIMES, FORCE_CFG.resample), consts, FORCE_CFG.render
    )
    lines = render.render_lines(feats, TIMES, "pad") if with_force else []
    return Window(
        episode_ref=f"synthetic/{ep.episode_id}",
        prompt_kind="outcome",
        anchor_s=0.0,
        frame_times=tuple(TIMES),
        force_lines=tuple(lines),
        target="Score: 5",
        split=split,  # type: ignore[arg-type]
    )


def clean_episodes(n: int = 4) -> list[Episode]:
    """``n`` clean pad episodes."""
    return [make_pad_episode(s) for s in range(n)]


def make_ctx(episodes: list[Episode], **over: object) -> VerifyContext:
    """A context under which every clean synthetic episode passes every check."""
    consts = fit_consts(episodes)
    base: dict[str, object] = {
        "source": "synthetic",
        "license_allowed": ["CC0"],
        "reader": SyntheticFrameReader(),
        "norm": consts,
        "resample_cfg": FORCE_CFG.resample,
        "render_cfg": FORCE_CFG.render,
        "hash_cache": {},
        "expected_label_rate": 0.3,
        "observed_label_rate": 0.3,
        "spot_n": 20,
        "spot_reviewer_pass": 18,
        "windows": {e.episode_id: [outcome_window(e, consts)] for e in episodes},
    }
    base.update(over)
    return VerifyContext(**base)  # type: ignore[arg-type]
