"""Rendering normalized features to the force text block, one line per frame.

Owns: render_lines, summary_lines, render_block, token_count, budget_check, load_tokenizer,
TokenBudgetError.
Must never: resample, normalize, or decide the ablation mode.

Summary lines (``RenderConfig.summary``): one ``summary <arm>:`` line per arm placed between
the header and the frame lines. They are pure functions of the same normalized features the
frame lines show (never of labels): contact segments (runs of c=1, single-frame gaps merged),
longest and total contact seconds, peak F, the hold phase (frames whose closure g is at least
80% of the window's maximum, or the contact frames when g is absent), its duration, median F,
late/early median ratio (last quarter over first quarter of the hold) and the fraction of hold
frames with F under 20% of the hold median. HRDexDB analysis (2026-10-22) showed these
trial-relative statistics carry more outcome signal than absolute levels: successful grasps
hold one sustained plateau while failures grip weakly, re-grasp, or lose force late.

Format: two decimals, arms in L then R order, an absent arm written ``L none`` / ``R none``.
NaN values (empty resample windows) render as 0.00 with c=0. A missing gripper renders g=0.00
(open; deliberately not distinguished from a present open gripper).
Negative values keep their minus sign; values that round to zero render as 0.00, never -0.00,
and the same holds for the frame time ``t``.

Token budget: real runs pass the Qwen tokenizer (any object with ``encode(str)`` returning a
sequence, e.g. ``tokenizers.Tokenizer`` wrapped, or a transformers tokenizer) to budget_check.
With none, an estimate of FALLBACK_TOKENS_PER_CHAR tokens per character is used, so no tokenizer
download is needed in tests. Real runs set ``render.tokenizer_json`` in configs/force.yaml to the
Qwen3.5 tokenizer.json path; load_tokenizer(cfg) then returns it for budget_check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from tactile_flora.config import RenderConfig
from tactile_flora.schema import Arm, ArmFeatures, ResampledForce

FALLBACK_TOKENS_PER_CHAR = 0.75
"""Tokens-per-character ratio used when no tokenizer is passed (see token_count).

Measured with the real tokenizer: 0.70 (40/57, one-arm pad), 0.75 (68/91, two-arm pad), 0.68
(34/50, one-arm wrist). 0.75 is the maximum, so the fallback never rejects what the real
tokenizer accepts.
"""


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


def load_tokenizer(cfg: RenderConfig) -> Tokenizer | None:
    """Load the ``tokenizers.Tokenizer`` at ``cfg.tokenizer_json``; None when it is unset."""
    if cfg.tokenizer_json is None:
        return None
    from tokenizers import Tokenizer as HfTokenizer

    class _Wrapped:
        def __init__(self, tok: HfTokenizer) -> None:
            self._tok = tok

        def encode(self, text: str) -> Sequence[int]:
            return self._tok.encode(text).ids

    return _Wrapped(HfTokenizer.from_file(cfg.tokenizer_json))


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
        lines.append(f"t={_fmt(t)} " + " | ".join(cells))
    return lines


def _runs(flags: Sequence[int]) -> list[tuple[int, int]]:
    """Inclusive index runs of 1s, merging gaps of exactly one frame."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _seconds(times: Sequence[float], a: int, b: int) -> float:
    """Duration covered by frames a..b inclusive: span plus one median frame step."""
    if len(times) < 2:
        return 0.0
    step = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    return float(times[b] - times[a]) + step


def summary_lines(features: ResampledForce) -> list[str]:
    """One label-free ``summary <arm>:`` line per arm (see the module docstring)."""
    times = list(features.frame_times)
    out: list[str] = []
    for arm in ("L", "R"):
        af = features.arms.get(arm)
        if af is None:
            continue
        if af.c is None:
            raise ValueError("features must be normalized (c missing) before summarizing")
        F = np.nan_to_num(np.asarray(af.F, dtype=np.float64))
        c = [int(v) for v in af.c]
        runs = _runs(c)
        longest = max((_seconds(times, a, b) for a, b in runs), default=0.0)
        total = sum(_seconds(times, a, b) for a, b in runs)
        if af.g is not None and max(af.g) > 0:
            g = np.asarray(af.g, dtype=np.float64)
            hold = g >= 0.8 * float(g.max())
        else:
            hold = np.asarray(c, dtype=bool)
        idx = np.flatnonzero(hold)
        if idx.size:
            hf = F[idx]
            median = float(np.median(hf))
            q = max(1, idx.size // 4)
            early = float(np.median(hf[:q]))
            late_early = float(np.median(hf[-q:]) / early) if early > 0 else 0.0
            low = float(np.mean(hf < 0.2 * median)) if median > 0 else 1.0
            hold_s = _seconds(times, int(idx[0]), int(idx[-1]))
        else:
            median = late_early = hold_s = 0.0
            low = 1.0
        out.append(
            f"summary {arm}: contacts={len(runs)} longest={_fmt(longest)}s "
            f"total={_fmt(total)}s peak={_fmt(float(F.max()) if F.size else 0.0)} "
            f"hold={_fmt(hold_s)}s holdF={_fmt(median)} late/early={_fmt(late_early)} "
            f"low={int(round(100 * low))}%"
        )
    return out


def render_block(
    lines: Sequence[str], cfg: RenderConfig, features: ResampledForce | None = None
) -> str:
    """Join lines under the header; an empty ``lines`` (control) gives an empty string.

    With ``cfg.summary`` and ``features`` the summary lines go between the header and the
    frame lines; summary without features is an error (callers must pass what they rendered).
    """
    if not lines:
        return ""
    if not cfg.summary:
        return "\n".join([cfg.header, *lines])
    if features is None:
        raise ValueError("render.summary needs the rendered features")
    return "\n".join([cfg.header, *summary_lines(features), *lines])


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
