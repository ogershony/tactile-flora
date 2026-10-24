"""Rendering normalized features to the force text block, one line per frame.

Owns: render_lines, render_block, summary_lines, code_summary, classifier_line, parse_code_line,
token_count, budget_check, fallback_tokens_per_char, load_tokenizer, TokenBudgetError.
Must never: resample, normalize, or decide the ablation mode.

Two line formats, selected by ``RenderConfig.format``:

``decimal`` (default, the spec's byte-exact format): two decimals, arms in L then R order, an
absent arm written ``L none`` / ``R none``. NaN values (empty resample windows) render as 0.00
with c=0. A missing gripper renders g=0.00 (open; deliberately not distinguished from a present
open gripper). Negative values keep their minus sign; values that round to zero render as 0.00,
never -0.00, and the same holds for the frame time ``t``. Block = ``header`` + [summary lines] +
lines.

Decimal summary lines (``RenderConfig.summary``, HRDexDB v3): one ``summary <arm>:`` line per arm
placed between the header and the frame lines. They are pure functions of the same normalized
features the frame lines show (never of labels): contact segments (runs of c=1, single-frame gaps
merged), longest and total contact seconds, peak F, the hold phase (frames whose closure g is at
least 80% of the window's maximum, or the contact frames when g is absent), its duration, median
F, late/early median ratio (last quarter over first quarter of the hold) and the fraction of hold
frames with F under 20% of the hold median. HRDexDB analysis (2026-10-22) showed these
trial-relative statistics carry more outcome signal than absolute levels: successful grasps
hold one sustained plateau while failures grip weakly, re-grasp, or lose force late. With
``classifier_json`` the scorer arm's line ends in the force-only classifier's P(success).

``code`` (token-cheap; the Qwen tokenizer splits digits one per character, so a decimal line
costs 40-68 tokens and a code line 18-28): every field is one letter plus one character, so
each is 1-2 tokens. Per arm ``F<level> [A<asym>] T<trend> C<contact> G<closure>``::

    t=3.50 L F3 A= T+ C1 Gc | R F0 A= T0 C0 Go      (pad)
    t=3.75 L F2 T0 C1 Gc | R none                    (wrist: no A field)

F is the force level 0-4: 0 when c=0 or F' is NaN, else 1/2/3/4 by ``levels`` (F' below
levels[0], levels[1], levels[2], or at/above levels[2]). A (pad only) is the pad asymmetry:
``=`` when F' <= 0 (or NaN) or |delta'| <= asym_frac * F', ``a`` when delta' is above that
(pad a dominant), ``b`` when below minus that. T is the trend of dF' (normalized per second):
``!`` at or below ``sharp_drop``, ``-`` below ``trend_down``, ``+`` above ``trend_up``, else
``0`` (NaN gives ``0``). C is the contact flag (0 for NaN frames). G is the gripper closure:
``o`` below closure[0], ``m`` below closure[1], else ``c``; ``o`` when there is no gripper or
g is NaN. Block = ``header_code`` + [summary] + [classifier line] + lines.

Compact code block (``RenderConfig.frame_lines`` false, added 2026-10-23): ``header_compact`` +
summary + classifier line, no per-frame lines. About 60-90 tokens instead of 1.3-2.7k, and
insensitive to force/video clock offsets of a few seconds (HRDexDB trials have them), because
nothing in it is tied to a frame of the video.

Summary line (``RenderConfig.summary``): one line per block, per arm, from the same window::

    Summary L: first contact t=4.25; peak F4 at t=5.00; contact lost t=7.75 while closed;
    sharp drops 1; final F0. R: no contact.

first contact = first frame with C=1; peak = the maximum level and its first time; contact
lost = first frame after first contact with C=0 while G is m or c (``no contact loss``
otherwise; an opening gripper releasing the object is not a loss); sharp drops = number of
frames with T=``!``; final = the level at the last frame. An arm with no C=1 frame is
``no contact``; an absent arm is ``none``.

Design choice: the summary is derived from the code lines themselves (``parse_code_line`` is
the exact inverse of the line grammar), not from the features. ``render_block(lines, cfg)``
therefore stays a pure function of the lines and the config, ``Window.force_lines`` stays a
tuple of per-frame strings, and every path that holds only the lines (manifest windows, the
verify prompt-diff check, training from stored lines) produces byte-identical blocks to the
eval harness, which holds the features. The classifier line is the one part that needs the
features; ``render_block`` adds it only when the caller passes features, source and episode id.

Token budget: real runs pass the Qwen tokenizer (any object with ``encode(str)`` returning a
sequence, e.g. ``tokenizers.Tokenizer`` wrapped, or a transformers tokenizer) to budget_check.
With none, an estimate of ``fallback`` tokens per character is used (FALLBACK_TOKENS_PER_CHAR
for decimal, FALLBACK_TOKENS_PER_CHAR_CODE for code; ``fallback_tokens_per_char(cfg)`` picks),
so no tokenizer download is needed in tests. Real runs set ``render.tokenizer_json`` in
configs/force.yaml to the Qwen3.5 tokenizer.json path; load_tokenizer(cfg) then returns it for
budget_check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple, Protocol

import numpy as np

from tactile_flora.config import RenderConfig
from tactile_flora.force.classifier import ForceScorer, load_scorer
from tactile_flora.schema import Arm, ArmFeatures, ResampledForce

FALLBACK_TOKENS_PER_CHAR = 0.75
"""Tokens-per-character ratio used for the decimal format when no tokenizer is passed.

Measured with the real tokenizer: 0.70 (40/57, one-arm pad), 0.75 (68/91, two-arm pad), 0.68
(34/50, one-arm wrist). 0.75 is the maximum, so the fallback never rejects what the real
tokenizer accepts.
"""

FALLBACK_TOKENS_PER_CHAR_CODE = 0.72
"""Tokens-per-character ratio used for the code format when no tokenizer is passed.

Measured 2026-10-21 with the Qwen3.5 tokenizer.json (Qwen3.5-9B snapshot; assumed to share the
vocabulary of the FLORA Qwen3.5-27B base, to be confirmed by hashing the released tokenizer.json):
every field is 2 tokens (``ĠF`` + digit) and every time digit is 1 token, so the ratio rises
with the number of time digits. Per line: 0.667 (28/42, two-arm pad ``t=3.50``), 0.651 (28/43,
two-arm pad ``t=23.75``), 0.636 (21/33, one-arm pad), 0.621 (18/29, one-arm wrist), 0.694
(25/36, two-arm wrist ``t=3.75``), 0.718 (28/39, two-arm wrist ``t=1000.00``, the worst
realistic line: four-digit frame times, every field 2 tokens). Each newline is one token; the
header is 67 tokens for 192 characters (0.35) and the summary line about 0.38, so whole blocks
measure 0.60-0.68. 0.72 is the per-line maximum, so the estimate stays conservative.
"""

_CLOSURE_WORD = {"o": "open", "m": "mid", "c": "closed"}


class TokenBudgetError(ValueError):
    """Raised when a rendered block exceeds the token budget."""


class Tokenizer(Protocol):
    """Anything whose ``encode`` returns a sized sequence of tokens."""

    def encode(self, text: str) -> Sequence[int]:
        """Tokenize ``text``."""
        ...


class CodeFields(NamedTuple):
    """The decoded fields of one arm cell of a code line (``asym`` is None for wrist)."""

    level: int
    asym: str | None
    trend: str
    contact: int
    closure: str


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


def fallback_tokens_per_char(cfg: RenderConfig | None) -> float:
    """Return the tokens-per-character estimate for ``cfg.format`` (decimal if ``cfg`` is None)."""
    if cfg is not None and cfg.format == "code":
        return FALLBACK_TOKENS_PER_CHAR_CODE
    return FALLBACK_TOKENS_PER_CHAR


# --- decimal format -----------------------------------------------------------------------------


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


# --- code format --------------------------------------------------------------------------------


def _level(f: float, contact: int, levels: Sequence[float]) -> int:
    if contact == 0 or math.isnan(f):
        return 0
    for k, cut in enumerate(levels):
        if f < cut:
            return k + 1
    return len(levels) + 1


def _asym(f: float, delta: float | None, asym_frac: float) -> str:
    if delta is None or math.isnan(f) or math.isnan(delta) or f <= 0:
        return "="
    bound = asym_frac * f
    if delta > bound:
        return "a"
    if delta < -bound:
        return "b"
    return "="


def _trend(df: float, cfg: RenderConfig) -> str:
    if math.isnan(df):
        return "0"
    if df <= cfg.sharp_drop:
        return "!"
    if df < cfg.trend_down:
        return "-"
    if df > cfg.trend_up:
        return "+"
    return "0"


def _closure(g: float | None, closure: Sequence[float]) -> str:
    if g is None or math.isnan(g):
        return "o"
    if g < closure[0]:
        return "o"
    if g < closure[1]:
        return "m"
    return "c"


def _code_arm_text(arm: Arm, channel: str, af: ArmFeatures, i: int, cfg: RenderConfig) -> str:
    if af.c is None:
        raise ValueError("features must be normalized (c missing) before rendering")
    f = af.F[i]
    contact = 0 if math.isnan(f) else af.c[i]
    parts = [arm, f"F{_level(f, contact, cfg.levels)}"]
    if channel == "pad":
        parts.append(f"A{_asym(f, None if af.delta is None else af.delta[i], cfg.asym_frac)}")
    parts.append(f"T{_trend(af.dF[i], cfg)}")
    parts.append(f"C{contact}")
    parts.append(f"G{_closure(None if af.g is None else af.g[i], cfg.closure)}")
    return " ".join(parts)


def parse_code_line(line: str) -> tuple[str, dict[Arm, CodeFields | None]]:
    """Decode a code line into its ``t=...`` text and per-arm fields (None for an absent arm).

    The exact inverse of the code grammar; raises ValueError on anything else (for example a
    decimal line rendered under a code config).
    """
    head, sep, rest = line.partition(" ")
    if not head.startswith("t=") or not sep:
        raise ValueError(f"not a code line: {line!r}")
    cells = rest.split(" | ")
    arms: dict[Arm, CodeFields | None] = {}
    for cell, expect in zip(cells, ("L", "R"), strict=True):
        toks = cell.split(" ")
        if toks[0] != expect:
            raise ValueError(f"not a code line (arm order): {line!r}")
        if toks[1:] == ["none"]:
            arms[expect] = None
            continue
        fields: dict[str, str] = {}
        for tok in toks[1:]:
            if len(tok) != 2 or tok[0] in fields:
                raise ValueError(f"not a code line (field {tok!r}): {line!r}")
            fields[tok[0]] = tok[1]
        try:
            arms[expect] = CodeFields(
                level=int(fields["F"]),
                asym=fields.get("A"),
                trend=fields["T"],
                contact=int(fields["C"]),
                closure=fields["G"],
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"not a code line ({exc}): {line!r}") from None
        if set(fields) - {"F", "A", "T", "C", "G"}:
            raise ValueError(f"not a code line (unknown field): {line!r}")
    return head, arms


def code_summary(lines: Sequence[str]) -> str:
    """Build the per-arm summary line of a code block (definitions in the module docstring)."""
    parsed = [parse_code_line(ln) for ln in lines]
    parts: list[str] = []
    for arm in ("L", "R"):
        frames = [(t, arms[arm]) for t, arms in parsed]
        if all(f is None for _, f in frames):
            parts.append(f"{arm}: none.")
            continue
        present = [(t, f) for t, f in frames if f is not None]
        first = next((k for k, (_, f) in enumerate(present) if f.contact == 1), None)
        if first is None:
            parts.append(f"{arm}: no contact.")
            continue
        peak = max(f.level for _, f in present)
        peak_t = next(t for t, f in present if f.level == peak)
        lost = next(
            (
                (t, f.closure)
                for t, f in present[first + 1 :]
                if f.contact == 0 and f.closure in ("m", "c")
            ),
            None,
        )
        lost_text = (
            f"contact lost {lost[0]} while {_CLOSURE_WORD[lost[1]]}" if lost else "no contact loss"
        )
        drops = sum(f.trend == "!" for _, f in present)
        parts.append(
            f"{arm}: first contact {present[first][0]}; peak F{peak} at {peak_t}; {lost_text}; "
            f"sharp drops {drops}; final F{present[-1][1].level}."
        )
    return "Summary " + " ".join(parts)


# --- lines and block ----------------------------------------------------------------------------


def render_lines(
    features: ResampledForce,
    frame_times: Sequence[float],
    channel: str,
    *,
    cfg: RenderConfig | None = None,
) -> list[str]:
    """Render one text line per frame time in ``cfg.format`` (decimal when ``cfg`` is None)."""
    if channel != features.channel:
        raise ValueError(f"channel {channel} != features channel {features.channel}")
    if len(frame_times) != len(features.frame_times):
        raise ValueError("frame_times length differs from features")
    code = cfg is not None and cfg.format == "code"
    lines: list[str] = []
    for i, t in enumerate(frame_times):
        cells: list[str] = []
        for arm in ("L", "R"):
            af = features.arms.get(arm)
            if af is None:
                cells.append(f"{arm} none")
            elif code:
                assert cfg is not None
                cells.append(_code_arm_text(arm, channel, af, i, cfg))
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


@lru_cache(maxsize=4)
def _scorer(path: str) -> ForceScorer:
    return load_scorer(path)


def summary_lines(
    features: ResampledForce,
    scorer: ForceScorer | str | None = None,
    episode_id: str | None = None,
) -> list[str]:
    """One label-free ``summary <arm>:`` line per arm (see the module docstring).

    With ``scorer`` (a ForceScorer or the path of its JSON) the arm it was fitted for also
    shows ``P(success)`` from the force-only classifier; ``episode_id`` selects the
    out-of-fold model for training episodes.
    """
    times = list(features.frame_times)
    out: list[str] = []
    model = _scorer(scorer) if isinstance(scorer, str) else scorer
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
        line = (
            f"summary {arm}: contacts={len(runs)} longest={_fmt(longest)}s "
            f"total={_fmt(total)}s peak={_fmt(float(F.max()) if F.size else 0.0)} "
            f"hold={_fmt(hold_s)}s holdF={_fmt(median)} late/early={_fmt(late_early)} "
            f"low={int(round(100 * low))}%"
        )
        if model is not None and model.arm == arm:
            line += f" force-only classifier P(success)={_fmt(model.score(features, episode_id))}"
        out.append(line)
    return out


def classifier_line(
    cfg: RenderConfig, features: ResampledForce, source: str, episode_id: str
) -> str | None:
    """``Force-only classifier P(success)=0.xx`` for ``cfg.classifier_json``, else None.

    ``cfg.classifier_json`` may list several scorers separated by commas (one per source); the
    first fitted on ``source`` whose arm is present is used, and None is returned when there is
    none. ``features`` are
    the ones being rendered (another episode's under the shuffled ablation); ``episode_id`` picks
    the out-of-fold model for the scorer's own training episodes.
    """
    if not cfg.classifier_json:
        return None
    for path in cfg.classifier_json.split(","):
        scorer = _scorer(path.strip())
        if source == scorer.source and scorer.arm in features.arms:
            return f"Force-only classifier P(success)={_fmt(scorer.score(features, episode_id))}"
    return None


def render_block(
    lines: Sequence[str],
    cfg: RenderConfig,
    features: ResampledForce | None = None,
    episode_id: str | None = None,
    *,
    source: str | None = None,
) -> str:
    """Join lines under the format's header; an empty ``lines`` (control) gives an empty string.

    ``features`` are the normalized features that ``lines`` were rendered from (another episode's
    under the shuffled ablation), ``episode_id`` and ``source`` the episode being prompted.
    decimal: with ``cfg.summary`` the ``summary_lines`` of ``features`` (required) go between the
    header and the frame lines; ``cfg.classifier_json`` ends the scorer arm's line in its
    out-of-fold P(success) (HRDexDB v3).
    code: ``header_code`` (``header_compact`` when ``cfg.frame_lines`` is false), the
    ``code_summary`` of the lines when ``cfg.summary``, the ``classifier_line`` when features,
    source and episode id are given, then the frame lines unless ``cfg.frame_lines`` is false.
    """
    if not lines:
        return ""
    if cfg.format == "decimal":
        if not cfg.summary:
            return "\n".join([cfg.header, *lines])
        if features is None:
            raise ValueError("render.summary needs the rendered features")
        summary = summary_lines(features, cfg.classifier_json, episode_id)
        return "\n".join([cfg.header, *summary, *lines])
    head = [cfg.header_code if cfg.frame_lines else cfg.header_compact]
    if cfg.summary:
        head.append(code_summary(lines))
    if features is not None and source is not None and episode_id is not None:
        line = classifier_line(cfg, features, source, episode_id)
        if line:
            head.append(line)
    return "\n".join([*head, *(lines if cfg.frame_lines else [])])


def token_count(
    block: str, tokenizer: Tokenizer | None = None, *, fallback: float = FALLBACK_TOKENS_PER_CHAR
) -> int:
    """Count tokens with ``tokenizer``, or estimate from characters (rounded up).

    The fallback ratio FALLBACK_TOKENS_PER_CHAR was measured on the exact decimal line format
    with the Qwen3 tokenizer (digits tokenize one per character): a two-arm pad line is 68 tokens
    for 91 characters, a one-arm pad line 40 for 57. Pass ``fallback`` (see
    ``fallback_tokens_per_char``) for the code format. The estimate is deliberately conservative.
    """
    if tokenizer is None:
        return -(-int(len(block) * fallback * 100) // 100)
    return len(tokenizer.encode(block))


def budget_check(
    block: str,
    max_tokens: int,
    tokenizer: Tokenizer | None = None,
    *,
    fallback: float = FALLBACK_TOKENS_PER_CHAR,
) -> int:
    """Return the token count, raising TokenBudgetError unless it is under ``max_tokens``."""
    n = token_count(block, tokenizer, fallback=fallback)
    if n >= max_tokens:
        raise TokenBudgetError(f"force block uses {n} tokens, budget is under {max_tokens}")
    return n
