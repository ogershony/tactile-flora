"""The token-cheap ``code`` force format: lines, header, summary, budget; decimal path untouched."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tactile_flora.config import ForceConfig, RenderConfig, load_config
from tactile_flora.force.render import (
    FALLBACK_TOKENS_PER_CHAR,
    FALLBACK_TOKENS_PER_CHAR_CODE,
    TokenBudgetError,
    budget_check,
    code_summary,
    fallback_tokens_per_char,
    parse_code_line,
    render_block,
    render_lines,
    token_count,
)
from tactile_flora.schema import ArmFeatures, ResampledForce

ROOT = Path(__file__).resolve().parents[2]
DEC = load_config(ROOT / "configs" / "force.yaml", ForceConfig).render
CODE = load_config(ROOT / "configs" / "force.code.yaml", ForceConfig).render
NAN = float("nan")

PAD_TIMES = [3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25]
PAD_LINES = [
    "t=3.50 L F0 A= T0 C0 Go | R F0 A= T0 C0 Go",
    "t=3.75 L F1 A= T+ C1 Gm | R F0 A= T0 C0 Go",
    "t=4.00 L F2 Aa T0 C1 Gc | R F0 A= T0 C0 Go",
    "t=4.25 L F3 Ab T- C1 Gc | R F0 A= T0 C0 Go",
    "t=4.50 L F4 A= T! C1 Gc | R F0 A= T0 C0 Go",
    "t=4.75 L F0 A= T0 C0 Gc | R F0 A= T0 C0 Go",  # NaN frame while the gripper is closed
    "t=5.00 L F2 A= T0 C1 Gm | R F0 A= T0 C0 Go",
    "t=5.25 L F1 A= T0 C1 Go | R F0 A= T0 C0 Go",
]
PAD_SUMMARY = (
    "Summary L: first contact t=3.75; peak F4 at t=4.50; contact lost t=4.75 while closed; "
    "sharp drops 1; final F1. R: no contact."
)


def _pad() -> ResampledForce:
    """Every level, every A / T / G symbol, a NaN frame, and a no-contact right arm."""
    n = len(PAD_TIMES)
    return ResampledForce(
        channel="pad",
        frame_times=PAD_TIMES,
        arms={
            "L": ArmFeatures(
                F=[0.0, 0.4, 1.0, 2.0, 3.0, NAN, 0.6, 0.2],
                delta=[0.0, 0.0, 0.4, -0.7, 0.5, NAN, 0.1, 0.0],
                dF=[0.0, 1.5, 0.5, -1.5, -2.0, NAN, 1.0, -1.0],
                g=[0.1, 0.45, 0.7, 1.0, 0.6, 0.6, 0.3, 0.0],
                c=[0, 1, 1, 1, 1, 0, 1, 1],
            ),
            "R": ArmFeatures(F=[0.0] * n, delta=[0.0] * n, dF=[0.0] * n, g=[0.0] * n, c=[0] * n),
        },
    )


def _wrist() -> ResampledForce:
    return ResampledForce(
        channel="wrist",
        frame_times=[10.0, 10.25, 10.5],
        arms={"L": ArmFeatures(F=[0.0, 1.6, 2.4], dF=[0.0, 3.0, -2.5], g=None, c=[0, 1, 1])},
    )


def test_pad_lines_exact() -> None:
    assert render_lines(_pad(), PAD_TIMES, "pad", cfg=CODE) == PAD_LINES


def test_wrist_lines_exact_no_gripper_absent_arm() -> None:
    lines = render_lines(_wrist(), [10.0, 10.25, 10.5], "wrist", cfg=CODE)
    assert lines == [
        "t=10.00 L F0 T0 C0 Go | R none",
        "t=10.25 L F3 T+ C1 Go | R none",
        "t=10.50 L F3 T! C1 Go | R none",
    ]
    assert code_summary(lines) == (
        "Summary L: first contact t=10.25; peak F3 at t=10.25; no contact loss; sharp drops 1; "
        "final F3. R: none."
    )


def test_header_summary_and_block() -> None:
    lines = render_lines(_pad(), PAD_TIMES, "pad", cfg=CODE)
    assert code_summary(lines) == PAD_SUMMARY
    assert CODE.header_code.startswith("Force code per frame (F force level: 0 none")
    # the legend states that levels are relative to the episode's own object (per-group norm)
    clause = "levels are relative to this object's typical grasp (F2 = this object's median grasp)"
    assert f"; {clause}; " in CODE.header_code
    assert RenderConfig.model_fields["header_code"].default == CODE.header_code
    assert render_block(lines, CODE) == "\n".join([CODE.header_code, PAD_SUMMARY, *lines])
    no_summary = CODE.model_copy(update={"summary": False})
    assert render_block(lines, no_summary) == "\n".join([CODE.header_code, *lines])
    assert render_block([], CODE) == ""  # control condition


def test_summary_release_is_not_a_loss_and_mid_closure() -> None:
    # contact ends while the gripper is open: a release, not a loss
    rel = [
        "t=0.00 L F2 A= T0 C1 Gc | R none",
        "t=0.25 L F0 A= T0 C0 Go | R none",
    ]
    assert code_summary(rel) == (
        "Summary L: first contact t=0.00; peak F2 at t=0.00; no contact loss; sharp drops 0; "
        "final F0. R: none."
    )
    mid = [
        "t=0.00 L F3 A= T0 C1 Gm | R none",
        "t=0.25 L F3 A= T! C1 Gm | R none",
        "t=0.50 L F0 A= T! C0 Gm | R none",
        "t=0.75 L F3 A= T0 C1 Gc | R none",
    ]
    assert code_summary(mid) == (
        "Summary L: first contact t=0.00; peak F3 at t=0.00; contact lost t=0.50 while mid; "
        "sharp drops 2; final F3. R: none."
    )


def test_parse_code_line_roundtrip_and_rejects_decimal() -> None:
    t, arms = parse_code_line(PAD_LINES[3])
    assert t == "t=4.25"
    left = arms["L"]
    assert left is not None and left == (3, "b", "-", 1, "c")
    assert arms["R"] == (0, "=", "0", 0, "o")
    _, wrist = parse_code_line("t=10.25 L F3 T+ C1 Go | R none")
    assert wrist["L"] is not None and wrist["L"].asym is None and wrist["R"] is None
    decimal = render_lines(_pad(), PAD_TIMES, "pad")[0]
    with pytest.raises(ValueError, match="not a code line"):
        parse_code_line(decimal)
    with pytest.raises(ValueError, match="not a code line"):
        render_block([decimal], CODE)


def test_decimal_path_unchanged_with_or_without_cfg() -> None:
    f = _pad()
    plain = render_lines(f, PAD_TIMES, "pad")
    assert render_lines(f, PAD_TIMES, "pad", cfg=DEC) == plain
    assert render_lines(f, PAD_TIMES, "pad", cfg=None) == plain
    assert plain[0] == (
        "t=3.50 pad L F=0.00 D=0.00 dF=0.00 g=0.10 c=0 | pad R F=0.00 D=0.00 dF=0.00 g=0.00 c=0"
    )
    assert render_block(plain, DEC) == "\n".join([DEC.header, *plain])
    assert fallback_tokens_per_char(None) == fallback_tokens_per_char(DEC) == 0.75
    assert FALLBACK_TOKENS_PER_CHAR == 0.75 and FALLBACK_TOKENS_PER_CHAR_CODE == 0.72


def _two_arm_lines(t0: float) -> list[str]:
    """64 two-arm pad lines with every field at its two-token spelling."""
    return [f"t={t0 + i * 0.25:.2f} L F4 Aa T! C1 Gm | R F1 Ab T- C1 Gm" for i in range(64)]


def test_budget_64_two_arm_pad_lines_under_code_budget() -> None:
    fb = fallback_tokens_per_char(CODE)
    assert fb == FALLBACK_TOKENS_PER_CHAR_CODE
    for t0 in (0.0, 1000.0):  # four-digit frame times are the worst realistic case
        block = render_block(_two_arm_lines(t0), CODE)
        n = budget_check(block, CODE.max_tokens, fallback=fb)
        assert n == token_count(block, fallback=fb) < CODE.max_tokens == 2500
    # the same 64 two-arm frames in the decimal format do not fit the decimal budget
    f = _pad()
    tiled = ResampledForce(
        channel="pad",
        frame_times=[3.5 + 0.25 * i for i in range(64)],
        arms={
            arm: ArmFeatures(
                F=af.F * 8,
                delta=None if af.delta is None else af.delta * 8,
                dF=af.dF * 8,
                g=None if af.g is None else af.g * 8,
                c=None if af.c is None else af.c * 8,
            )
            for arm, af in f.arms.items()
        },
    )
    dec = render_lines(tiled, tiled.frame_times, "pad", cfg=DEC)
    assert len(dec) == 64
    with pytest.raises(TokenBudgetError):
        budget_check(render_block(dec, DEC), DEC.max_tokens)
    code = render_block(render_lines(tiled, tiled.frame_times, "pad", cfg=CODE), CODE)
    assert budget_check(code, CODE.max_tokens, fallback=fb) < CODE.max_tokens
    # measured 2026-10-21 (Qwen3.5 tokenizer): 2,200 real tokens for the t>=1000 block; the
    # estimate must stay above the real count
    assert token_count(render_block(_two_arm_lines(1000.0), CODE), fallback=fb) >= 2200


def test_fallback_is_per_line_maximum() -> None:
    worst = "t=1000.00 L F4 T! C1 Gm | R F1 T- C1 Gm"  # 28 tokens / 39 chars measured
    assert math.isclose(28 / len(worst), 0.718, abs_tol=0.001)
    assert 28 / len(worst) <= FALLBACK_TOKENS_PER_CHAR_CODE


def test_levels_and_closure_validated() -> None:
    base = CODE.model_dump()
    for key, bad in (("levels", [0.5, 0.5, 2.5]), ("levels", [0.5, 1.5]), ("closure", [0.6, 0.3])):
        with pytest.raises(ValueError, match="strictly ascending"):
            RenderConfig(**{**base, key: bad})


def test_code_line_grammar() -> None:
    cell = r"(?:[LR] none|[LR] F[0-4](?: A[ab=])? T[+0!-] C[01] G[omc])"
    pat = re.compile(rf"^t=\d+\.\d\d {cell} \| {cell}$")
    for line in render_lines(_pad(), PAD_TIMES, "pad", cfg=CODE):
        assert pat.match(line), line
    for line in render_lines(_wrist(), [10.0, 10.25, 10.5], "wrist", cfg=CODE):
        assert pat.match(line), line
