import pytest

from tactile_flora.labels.windows import dense_frame_times, outcome_frame_indices
from tactile_flora.prompts import templates as T
from tactile_flora.prompts.flora import build_prompt, mosaic_layout, prompt_diff, to_messages
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
        assert "H" not in c.user_text_after_video.split("\n")[0:1]
        assert f.user_text_after_video.startswith(block + "\n")
        assert f.system.replace("\n" + T.FORCE_SENTENCE, "") == c.system


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
    assert m[1]["content"][0]["text"] == "Task: t"
    assert m[1]["content"][2]["text"].startswith("blk\n")


def test_mosaic_layouts() -> None:
    assert mosaic_layout(["scene"]) == ((1, 1), ["scene"])
    assert mosaic_layout(["wrist_left", "scene"]) == ((1, 2), ["scene", "wrist_left"])
    assert mosaic_layout(["a", "b", "scene"])[0] == (1, 3)
    assert mosaic_layout(["a", "b", "c", "scene"]) == ((2, 2), ["scene", "a", "b", "c"])
    assert mosaic_layout(list("abcde"))[0] == (2, 2)
    with pytest.raises(ValueError):
        mosaic_layout([])
