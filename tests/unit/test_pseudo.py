import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.labels.pseudo import PseudoRefused, pseudo_label
from tactile_flora.labels.windows import dense_frame_times
from tactile_flora.schema import Window


def cfg(eval_only: bool) -> SourceConfig:
    return SourceConfig(
        name="sharpadex", root="r", license_allowed=[], max_windows=1, max_episodes=1,
        eval_only=eval_only, expected_label_rate=0.5, seed=0,
    )  # fmt: skip


def win(ep: str, anchor: float = 10.0) -> Window:
    return Window(
        episode_ref=f"sharpadex/{ep}", prompt_kind="dense", anchor_s=anchor,
        frame_times=tuple(dense_frame_times(anchor)), force_lines=(), target="", split="train",
    )  # fmt: skip


ANSWER = (
    "Worst: 2\nQuality:\n" + "3" * 4 + "2" * 4 + "3" * 24 + "\nRationales:\n5-8 the hand hesitates"
)


def test_pseudo_parses_runner_output() -> None:
    calls: list[Window] = []

    def runner(w: Window) -> str:
        calls.append(w)
        return ANSWER

    (lab,) = pseudo_label(cfg(False), [win("a")], {"zzz"}, runner)
    assert lab.origin == "pseudo" and lab.kind == "dense" and len(calls) == 1
    (s,) = lab.spans
    assert (s.t0, s.t1, s.quality, s.rationale) == (10.0, 12.0, 2, "the hand hesitates")


def test_refuses_eval_only() -> None:
    def runner(w: Window) -> str:
        raise AssertionError("must not run")

    with pytest.raises(PseudoRefused):
        pseudo_label(cfg(True), [win("a")], set(), runner)


def test_refuses_holdout_before_running() -> None:
    calls: list[Window] = []

    def runner(w: Window) -> str:
        calls.append(w)
        return ANSWER

    with pytest.raises(PseudoRefused):
        pseudo_label(cfg(False), [win("a"), win("held")], {"held"}, runner)
    assert calls == []
