r"""Answer grammar for FLORA outputs and the parsers that read it.

Owns: the ONE definition of the dense and outcome answer grammars (regex constants below),
ParsedDense / ParsedOutcome (pydantic models kept here, they never cross a stage boundary
on disk), parse_dense, parse_outcome, dense_regex, and parsed_to_label.
Must never: build prompts, or accept a Worst that disagrees with the minimum digit.

Dense grammar (``labels/windows.py`` builds targets from the same constants)::

    Worst: [123]\nQuality:\n[123]{32}\nRationales:(?:\n<g>-<g> [^\n]{1,240})*

This is exactly the released FLORA decoding regex. Group numbers <g> are 1-based (1..32). The
digits sit on their own line after ``Quality:``; a rationale line has a mandatory space and
1-240 non-newline characters. Outcome: ``Score: [1-5]`` or
``Rationale: [^\n]{1,512}\n\nScore: [1-5]`` (blank line). Only surrounding whitespace is
tolerated (the model text is stripped before matching).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from tactile_flora.schema import Label, Quality, Span

N_GROUPS = 32
GROUP_S = 0.5  # one group = 2 frames at 4 FPS
WINDOW_PRE_S = 2.0  # window spans [anchor - 2, anchor + 14)

# --- the single grammar definition -------------------------------------------------------
GROUP_NUM = r"(?:[1-9]|[12][0-9]|3[0-2])"
MAX_DENSE_RATIONALE = 240
MAX_OUTCOME_RATIONALE = 512
DENSE_BODY = (
    rf"Worst: [123]\nQuality:\n[123]{{{N_GROUPS}}}\nRationales:"
    rf"(?:\n{GROUP_NUM}-{GROUP_NUM} [^\n]{{1,{MAX_DENSE_RATIONALE}}})*"
)
RATIONALE_LINE = re.compile(rf"({GROUP_NUM})-({GROUP_NUM}) ([^\n]{{1,{MAX_DENSE_RATIONALE}}})")
DENSE_RE = re.compile(DENSE_BODY)
OUTCOME_RATIONALE_BODY = rf"Rationale: [^\n]{{1,{MAX_OUTCOME_RATIONALE}}}\n\nScore: [1-5]"
OUTCOME_SCORE_BODY = r"Score: [1-5]"
OUTCOME_RATIONALE_RE = re.compile(OUTCOME_RATIONALE_BODY)
OUTCOME_SCORE_RE = re.compile(OUTCOME_SCORE_BODY)


class ParsedDense(BaseModel):
    """A parsed dense answer; all-normal with parse_failed=True when the answer was rejected."""

    worst: int
    digits: list[int]
    rationales: list[tuple[int, int, str]] = Field(default_factory=lambda: [])
    parse_failed: bool = False


class ParsedOutcome(BaseModel):
    """A parsed outcome answer; score is None when parsing failed."""

    score: int | None
    rationale: str = ""
    parse_failed: bool = False


def dense_regex() -> str:
    """Return the dense grammar as a regex string (no anchors) for constrained decoding."""
    return DENSE_BODY


def outcome_regex(with_rationale: bool) -> str:
    """Return the outcome grammar as a regex string for constrained decoding."""
    return OUTCOME_RATIONALE_BODY if with_rationale else OUTCOME_SCORE_BODY


def _dense_failure() -> ParsedDense:
    return ParsedDense(worst=3, digits=[3] * N_GROUPS, rationales=[], parse_failed=True)


def parse_dense(text: str) -> ParsedDense:
    """Parse a dense answer; reject to the all-normal replacement on any violation."""
    if DENSE_RE.fullmatch(text.strip()) is None:
        return _dense_failure()
    lines = text.strip().split("\n")
    worst = int(lines[0].removeprefix("Worst: "))
    digits = [int(c) for c in lines[2]]
    if worst != min(digits):
        return _dense_failure()
    rationales: list[tuple[int, int, str]] = []
    for line in lines[4:]:
        r = RATIONALE_LINE.fullmatch(line)
        if r is None:  # unreachable given DENSE_RE, kept as a guard
            return _dense_failure()
        first, last = int(r.group(1)), int(r.group(2))
        if last < first:
            return _dense_failure()
        rationales.append((first, last, r.group(3).strip()))
    return ParsedDense(worst=worst, digits=digits, rationales=rationales)


def parse_outcome(text: str) -> ParsedOutcome:
    """Parse an outcome answer in either rationale or score-only mode."""
    t = text.strip()
    if OUTCOME_RATIONALE_RE.fullmatch(t) is not None:
        rationale, _, score = t.removeprefix("Rationale: ").partition("\n\nScore: ")
        return ParsedOutcome(score=int(score), rationale=rationale.strip())
    if OUTCOME_SCORE_RE.fullmatch(t) is not None:
        return ParsedOutcome(score=int(t.removeprefix("Score: ")))
    return ParsedOutcome(score=None, parse_failed=True)


def parsed_to_label(
    parsed: ParsedDense | ParsedOutcome,
    anchor: float,
    origin: str,
) -> Label:
    """Convert a parsed answer to a Label; ``origin`` is supplied by the caller.

    Dense: one Span per maximal run of equal non-normal digits, timed on the episode clock
    from ``anchor`` (group g covers [anchor - 2 + 0.5 (g - 1), anchor - 2 + 0.5 g)); the
    rationale is the one whose range contains the run, else "". Outcome: ``outcome=score``.
    """
    if origin not in ("human", "programmatic", "pseudo", "source"):
        raise ValueError(f"bad origin {origin!r}")
    if isinstance(parsed, ParsedOutcome):
        return Label(kind="outcome", origin=origin, outcome=parsed.score)  # type: ignore[arg-type]
    spans: list[Span] = []
    g = 0
    d = parsed.digits
    while g < len(d):
        h = g
        while h + 1 < len(d) and d[h + 1] == d[g]:
            h += 1
        if d[g] != 3:
            first, last = g + 1, h + 1
            text = next((t for a, b, t in parsed.rationales if a <= first and last <= b), "")
            q: Quality = 1 if d[g] == 1 else 2
            t0 = anchor - WINDOW_PRE_S + GROUP_S * (first - 1)
            spans.append(
                Span(t0=t0, t1=t0 + GROUP_S * (last - first + 1), quality=q, rationale=text)
            )
        g = h + 1
    return Label(kind="dense", origin=origin, spans=spans)  # type: ignore[arg-type]
