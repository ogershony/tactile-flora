import pytest

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.labels.windows import dense_frame_times, outcome_frame_indices
from tactile_flora.prompts import templates as T
from tactile_flora.prompts.flora import (
    build_prompt,
    force_sentence_for,
    mosaic_layout,
    prompt_diff,
    to_messages,
)
from tactile_flora.schema import Window


def _dense() -> Window:
    return Window(
        episode_ref="e",
        prompt_kind="dense",
        anchor_s=10.0,
        frame_times=tuple(dense_frame_times(10.0)),
        force_lines=("t=8.00 F=0.1", "t=8.25 F=0.2"),
        target="x",
        split="train",
    )


def _outcome() -> Window:
    idx = outcome_frame_indices(120, 30.0)
    return Window(
        episode_ref="e",
        prompt_kind="outcome",
        anchor_s=0.0,
        frame_times=tuple(i / 30.0 for i in idx),
        force_lines=("a",),
        target="x",
        split="train",
    )


def test_force_vs_control_differ_only_in_block() -> None:
    for w in (_dense(), _outcome()):
        assert prompt_diff(w, task="stack cups", camera_roles=["scene", "wrist_left"])
        block = "H\nl1"
        f = build_prompt(w, block)
        c = build_prompt(w, None)
        assert T.FORCE_SENTENCE in f.system and T.FORCE_SENTENCE not in c.system
        assert f.user_text_after_video == "\n" + block + c.user_text_after_video
        assert f.system.endswith("\n" + T.FORCE_SENTENCE)
        assert f.system.replace("\n" + T.FORCE_SENTENCE, "") == c.system


def test_code_sentence_and_prompt_diff_with_block() -> None:
    dec = load_config("configs/force.yaml", ForceConfig).render
    code = load_config("configs/force.code.yaml", ForceConfig).render
    assert force_sentence_for(dec) == T.FORCE_SENTENCE
    assert force_sentence_for(code) == T.FORCE_SENTENCE_CODE
    assert T.FORCE_SENTENCE_CODE != T.FORCE_SENTENCE and "\n" not in T.FORCE_SENTENCE_CODE
    for w in (_dense(), _outcome()):
        block = "\n".join([code.header_code, "Summary L: no contact. R: none.", *w.force_lines])
        assert prompt_diff(w, block=block, force_sentence=T.FORCE_SENTENCE_CODE, task="t")
        f = build_prompt(w, block, force_sentence=T.FORCE_SENTENCE_CODE, task="t")
        c = build_prompt(w, None, force_sentence=T.FORCE_SENTENCE_CODE, task="t")
        assert f.system == c.system + "\n" + T.FORCE_SENTENCE_CODE
        assert T.FORCE_SENTENCE not in f.system
        assert f.user_text_after_video == "\n" + block + c.user_text_after_video
        # the default sentence is the decimal one, so a code block without the kwarg differs
        assert build_prompt(w, block, task="t").system == c.system + "\n" + T.FORCE_SENTENCE


def test_outcome_modes_and_blank() -> None:
    w = _outcome()
    assert build_prompt(w, None, with_rationale=False).system == T.OUTCOME_SYSTEM_SCORE_ONLY
    assert build_prompt(w, None).system == T.OUTCOME_SYSTEM_RATIONALE
    assert build_prompt(w, None, video_blank=True).video.blank


def test_messages_structure() -> None:
    m = to_messages(build_prompt(_dense(), "blk", task="t"))
    assert [x["role"] for x in m] == ["system", "user"]
    kinds = [c["type"] for c in m[1]["content"]]
    assert kinds == ["text", "video", "text"]
    assert m[1]["content"][0]["text"] == "Task: t\n\nSynchronized multiview clip:\n"
    assert m[1]["content"][2]["text"].startswith("\nblk\n\nThe clip contains 64")


def test_verbatim_templates() -> None:
    assert T.TEMPLATES_VERIFIED_AGAINST_PAPER
    assert T.OUTCOME_SYSTEM_SCORE_ONLY.endswith(
        "exactly.\nDo not provide a rationale. Return exactly:\nScore: <1-5>"
    )
    assert T.OUTCOME_SYSTEM_RATIONALE.endswith("Rationale: <rationale>\n\nScore: <1-5>")
    assert T.DENSE_SYSTEM.startswith("You are a dense action-quality reward model")
    assert T.DENSE_SYSTEM.endswith("in time order>")


def test_user_text_verbatim() -> None:
    d = build_prompt(_dense(), None, task="a   b\nc")
    assert d.user_text_before_video == "Task: a b c\n\nSynchronized multiview clip:\n"
    assert "run from t=8.000 to t=23.750 seconds of the episode." in d.user_text_after_video
    assert d.user_text_after_video.endswith("rationales for non-normal runs.")
    o = build_prompt(_outcome(), None, task="x")
    assert o.user_text_before_video == "Task: x\n\nComplete trajectory video:\n"
    assert o.user_text_after_video == "\n\nEvaluate the terminal success of the task."


def test_mosaic_layouts() -> None:
    assert mosaic_layout(["scene"]) == ((1, 1), ["scene"])
    assert mosaic_layout(["wrist_left", "scene"]) == ((1, 2), ["scene", "wrist_left"])
    assert mosaic_layout(["a", "b", "scene"])[0] == (1, 3)
    assert mosaic_layout(["a", "b", "c", "scene"]) == ((2, 2), ["scene", "a", "b", "c"])
    assert mosaic_layout(list("abcde"))[0] == (2, 2)
    with pytest.raises(ValueError):
        mosaic_layout([])
