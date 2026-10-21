"""Rendering normalized features to the force text block, one line per frame.

Owns: render_lines, render_block, token_count, budget_check, TokenBudgetError.
Must never: resample, normalize, or decide the ablation mode.

Format: two decimals, arms in L then R order, an absent arm written ``L none`` / ``R none``.
NaN values (empty resample windows) render as 0.00 with c=0. A missing gripper renders g=0.00.
Negative values keep their minus sign; values that round to zero render as 0.00, never -0.00.

Token budget: real runs pass the Qwen tokenizer (any object with ``encode(str)`` returning a
sequence, e.g. ``tokenizers.Tokenizer`` wrapped, or a transformers tokenizer) to budget_check.
With none, a conservative estimate of FALLBACK_TOKENS_PER_CHAR tokens per character is used, so no
tokenizer download is needed in tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from tactile_flora.config import RenderConfig
from tactile_flora.schema import Arm, ArmFeatures, ResampledForce

FALLBACK_TOKENS_PER_CHAR = 0.8
"""Conservative tokens-per-character ratio used when no tokenizer is passed (see token_count)."""


class TokenBudgetError(ValueError):
    """Raised when a rendered block exceeds the token budget."""


class Tokenizer(Protocol):
    """Anything whose ``encode`` returns a sized sequence of tokens."""

    def encode(self, text: str) -> Sequence[int]:
        """Tokenize ``text``."""
        ...


def _fmt(x: float | None) -> str:
    v = 0.0 if x is None or math.isnan(x) else x
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


def _arm_text(arm: Arm, channel: str, af: ArmFeatures, i: int) -> str:
    if af.c is None:
        raise ValueError("features must be normalized (c missing) before rendering")
    nan = math.isnan(af.F[i])
    parts = [f"{channel} {arm}", f"F={_fmt(af.F[i])}"]
    if channel == "pad":
        parts.append(f"D={_fmt(None if af.delta is None else af.delta[i])}")
    parts.append(f"dF={_fmt(af.dF[i])}")
    parts.append(f"g={_fmt(None if af.g is None else af.g[i])}")
    parts.append(f"c={0 if nan else af.c[i]}")
    return " ".join(parts)


def render_lines(features: ResampledForce, frame_times: Sequence[float], channel: str) -> list[str]:
    """Render one text line per frame time."""
    if channel != features.channel:
        raise ValueError(f"channel {channel} != features channel {features.channel}")
    if len(frame_times) != len(features.frame_times):
        raise ValueError("frame_times length differs from features")
    lines: list[str] = []
    for i, t in enumerate(frame_times):
        cells: list[str] = []
        for arm in ("L", "R"):
            af = features.arms.get(arm)
            if af is None:
                cells.append(f"{arm} none")
            else:
                cells.append(_arm_text(arm, channel, af, i))
        lines.append(f"t={t:.2f} " + " | ".join(cells))
    return lines


def render_block(lines: Sequence[str], cfg: RenderConfig) -> str:
    """Join lines under the header; an empty ``lines`` (control) gives an empty string."""
    if not lines:
        return ""
    return "\n".join([cfg.header, *lines])


def token_count(block: str, tokenizer: Tokenizer | None = None) -> int:
    """Count tokens with ``tokenizer``, or estimate from characters (rounded up).

    The fallback ratio FALLBACK_TOKENS_PER_CHAR was measured on the exact line format with the
    Qwen3 tokenizer (digits tokenize one per character): a two-arm pad line is 68 tokens for
    91 characters, a one-arm pad line 40 for 57. The estimate is deliberately conservative.
    """
    if tokenizer is None:
        return -(-int(len(block) * FALLBACK_TOKENS_PER_CHAR * 100) // 100)
    return len(tokenizer.encode(block))


def budget_check(block: str, max_tokens: int, tokenizer: Tokenizer | None = None) -> int:
    """Return the token count, raising TokenBudgetError unless it is under ``max_tokens``."""
    n = token_count(block, tokenizer)
    if n >= max_tokens:
        raise TokenBudgetError(f"force block uses {n} tokens, budget is under {max_tokens}")
    return n
