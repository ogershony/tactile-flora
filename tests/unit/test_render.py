from pathlib import Path

import pytest

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.force.render import (
    TokenBudgetError,
    budget_check,
    load_tokenizer,
    render_block,
    render_lines,
    token_count,
)
from tactile_flora.schema import ArmFeatures, ResampledForce

CFG = load_config("configs/force.yaml", ForceConfig)


def _pad() -> ResampledForce:
    return ResampledForce(
        channel="pad",
        frame_times=[3.5, 3.75],
        arms={
            "L": ArmFeatures(
                F=[0.62, 0.5], delta=[-0.08, 0.0], dF=[0.41, -0.004], g=[0.31, 0.3], c=[1, 1]
            ),
            "R": ArmFeatures(
                F=[0.0, float("nan")], delta=[0.0, 0.0], dF=[0.0, 0.0], g=[1.0, 1.0], c=[0, 1]
            ),
        },
    )


def test_pad_exact() -> None:
    lines = render_lines(_pad(), [3.5, 3.75], "pad")
    assert lines[0] == (
        "t=3.50 pad L F=0.62 D=-0.08 dF=0.41 g=0.31 c=1 | pad R F=0.00 D=0.00 dF=0.00 g=1.00 c=0"
    )
    # -0.004 renders 0.00 not -0.00; NaN renders 0.00 with c=0
    assert lines[1] == (
        "t=3.75 pad L F=0.50 D=0.00 dF=0.00 g=0.30 c=1 | pad R F=0.00 D=0.00 dF=0.00 g=1.00 c=0"
    )


def test_wrist_absent_arm_negative_df() -> None:
    f = ResampledForce(
        channel="wrist",
        frame_times=[3.75],
        arms={"L": ArmFeatures(F=[0.31], dF=[-0.12], g=[0.52], c=[1])},
    )
    lines = render_lines(f, [3.75], "wrist")
    assert lines == ["t=3.75 wrist L F=0.31 dF=-0.12 g=0.52 c=1 | R none"]
    block = render_block(lines, CFG.render)
    assert block == ("Force (normalized: 0 free space, 1 median grasp, max 3):\n" + lines[0])
    assert render_block([], CFG.render) == ""


def test_budget() -> None:
    lines = render_lines(_pad(), [3.5, 3.75], "pad") * 4
    block = render_block(lines, CFG.render)
    assert budget_check(block, CFG.render.max_tokens) == token_count(block)
    with pytest.raises(TokenBudgetError):
        budget_check(block * 10, CFG.render.max_tokens)

    class Fake:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    assert budget_check(block, CFG.render.max_tokens, Fake()) == len(block.split())


def _one_arm_line() -> str:
    f = ResampledForce(
        channel="pad",
        frame_times=[23.75],
        arms={"L": ArmFeatures(F=[0.62], delta=[-0.08], dF=[0.41], g=[0.31], c=[1])},
    )
    return render_lines(f, [23.75], "pad")[0]


def test_fallback_accepts_64_one_arm_pad_lines() -> None:
    block = render_block([_one_arm_line()] * 64, CFG.render)
    assert CFG.render.max_tokens == 3000
    assert budget_check(block, CFG.render.max_tokens) < 3000


def test_tokenizer_json_loader_and_use(tmp_path: Path) -> None:
    from tokenizers import Tokenizer, models

    assert load_tokenizer(CFG.render) is None
    path = tmp_path / "tok.json"
    Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(path))
    cfg = CFG.render.model_copy(update={"tokenizer_json": str(path)})
    tok = load_tokenizer(cfg)
    assert tok is not None
    block = render_block([_one_arm_line()] * 64, CFG.render)
    assert budget_check(block, 5, tok) == 1  # whole text is one unknown word
    with pytest.raises(TokenBudgetError):
        budget_check(block, 5)


def test_time_never_negative_zero() -> None:
    f = ResampledForce(
        channel="wrist",
        frame_times=[-0.001],
        arms={"L": ArmFeatures(F=[0.1], dF=[0.0], g=None, c=[0])},
    )
    assert render_lines(f, [-0.001], "wrist")[0].startswith("t=0.00 ")
