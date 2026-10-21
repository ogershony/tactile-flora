import random

from tactile_flora.labels.windows import to_flora_dense_target, to_flora_outcome_target
from tactile_flora.prompts.parse import (
    dense_regex,
    parse_dense,
    parse_outcome,
    parsed_to_label,
)
from tactile_flora.schema import Label, Span

WORDS = ["slips", "drops", "force", "cup", "gripper", "left", "re-grasps", "object"]


def _spans(rng: random.Random, duration: float) -> list[Span]:
    out: list[Span] = []
    for _ in range(rng.randint(0, 5)):
        t0 = rng.uniform(0, duration)
        t1 = t0 + rng.uniform(0, 6)
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 4)))
        out.append(Span(t0=t0, t1=t1, quality=rng.choice([1, 2, 3]), rationale=text))
    return out


def test_dense_round_trip() -> None:
    rng = random.Random(1234)
    for _ in range(1000):
        duration = rng.uniform(1, 60)
        anchor = rng.choice([0.0, 10.0, rng.uniform(0, 40)])
        spans = _spans(rng, duration)
        text = to_flora_dense_target(spans, anchor, duration)
        p = parse_dense(text)
        assert not p.parse_failed, text
        assert p.worst == min(p.digits) and len(p.digits) == 32
        lines = text.split("\n")
        assert lines[1] == "Quality: " + "".join(map(str, p.digits))
        assert len(p.rationales) == len(lines) - 3
        for (a, b, t), line in zip(p.rationales, lines[3:], strict=True):
            assert line == (f"{a}-{b} {t}" if t else f"{a}-{b}")
        # label conversion reproduces the digits
        lab = parsed_to_label(p, anchor, "human")
        assert lab.kind == "dense"
        assert len(lab.spans) == len(p.rationales)


def test_all_normal_target() -> None:
    t = to_flora_dense_target([], 0.0, 30.0)
    assert t == "Worst: 3\nQuality: " + "3" * 32 + "\nRationales:"
    assert not parse_dense(t).parse_failed


def test_worst_mismatch_rejected() -> None:
    bad = "Worst: 2\nQuality: " + "3" * 31 + "1\nRationales:\n32-32 x"
    p = parse_dense(bad)
    assert p.parse_failed and p.digits == [3] * 32 and p.worst == 3 and p.rationales == []


def test_malformed_rejected() -> None:
    for bad in [
        "",
        "Worst: 3\nQuality: " + "3" * 31 + "\nRationales:",
        "Worst: 3\nQuality: " + "3" * 32 + "\nRationales:\n40-41 x",
        "Worst: 1\nQuality: " + "1" * 32 + "\nRationales:\n5-3 x",
        "Worst: 4\nQuality: " + "4" * 32 + "\nRationales:",
    ]:
        assert parse_dense(bad).parse_failed


def test_regex_string_matches() -> None:
    import re

    t = to_flora_dense_target([Span(t0=1, t1=3, quality=1, rationale="drops")], 0.0, 30.0)
    assert re.fullmatch(dense_regex(), t)


def test_outcome_round_trip() -> None:
    lab = Label(
        kind="outcome",
        origin="human",
        outcome=2,
        spans=[Span(t0=0, t1=1, quality=1, rationale="a b")],
    )
    p = parse_outcome(to_flora_outcome_target(lab, True))
    assert p.score == 2 and p.rationale == "a b" and not p.parse_failed
    p = parse_outcome(to_flora_outcome_target(lab, False))
    assert p.score == 2 and p.rationale == ""
    assert parsed_to_label(p, 0.0, "pseudo").outcome == 2


def test_outcome_failure() -> None:
    for bad in ["Score: 6", "Score: ", "hello", "Rationale: x"]:
        p = parse_outcome(bad)
        assert p.parse_failed and p.score is None
