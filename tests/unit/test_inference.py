import math
import sys

import numpy as np
import pytest

from tactile_flora.eval import inference as I
from tactile_flora.prompts.flora import build_prompt, to_messages
from tactile_flora.prompts.parse import dense_regex, parse_dense, parse_outcome
from tactile_flora.schema import Window
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.synthetic_video import SyntheticFrameReader


def window(kind: str) -> Window:
    n = 64 if kind == "dense" else 8
    return Window(
        episode_ref="synthetic/pad-0", prompt_kind=kind, anchor_s=0.0,  # type: ignore[arg-type]
        frame_times=tuple(i * 0.25 for i in range(n)), force_lines=(), target="", split="val",
    )  # fmt: skip


def test_expected_score_renormalizes_digits() -> None:
    lp = {"4": math.log(0.3), "5": math.log(0.3), "x": math.log(0.4), " 1": math.log(0.0 + 1e-30)}
    assert I.expected_score(lp) == pytest.approx(4.5)
    assert I.progress(4.5) == pytest.approx(0.875)
    assert I.progress(1.0) == 0.0
    with pytest.raises(ValueError):
        I.expected_score({"a": -1.0})


def test_stub_outcome_formats_and_logprobs() -> None:
    stub = I.StubBackend(outcome_score=4)
    for rationale, prefix in ((False, "Score: 4"), (True, "Rationale:")):
        b = build_prompt(window("outcome"), None, with_rationale=rationale)
        res = stub.generate(to_messages(b), b.video, regex=None, max_new_tokens=12, top_logprobs=20)
        assert res.text.startswith(prefix) and parse_outcome(res.text).score == 4
        assert res.score_token_logprobs is not None
        assert set(res.score_token_logprobs) == set("12345")
        e = I.expected_score(res.score_token_logprobs)
        assert e == pytest.approx(3.875)  # 0.9 * 4 + 0.025 * (1 + 2 + 3 + 5)
    custom = I.StubBackend(logprobs={"1": math.log(0.5), "5": math.log(0.5)})
    b = build_prompt(window("outcome"), None, with_rationale=False)
    r = custom.generate(to_messages(b), b.video, regex=None, max_new_tokens=12, top_logprobs=20)
    assert I.expected_score(r.score_token_logprobs or {}) == pytest.approx(3.0)
    assert stub.calls == 2 and custom.calls == 1


def test_stub_dense_is_valid_and_deterministic() -> None:
    b = build_prompt(window("dense"), None)
    kw = {"regex": dense_regex(), "max_new_tokens": 640, "top_logprobs": 0}
    r1 = I.StubBackend().generate(to_messages(b), b.video, **kw)  # type: ignore[arg-type]
    r2 = I.StubBackend().generate(to_messages(b), b.video, **kw)  # type: ignore[arg-type]
    assert r1.text == r2.text and r1.score_token_logprobs is None
    parsed = parse_dense(r1.text)
    assert not parsed.parse_failed and parsed.digits == [3] * 32
    bad = I.StubBackend(dense_digits="3" * 4 + "1" * 2 + "3" * 26)
    p = parse_dense(bad.generate(to_messages(b), b.video, **kw).text)  # type: ignore[arg-type]
    assert not p.parse_failed and p.worst == 1 and p.rationales[0][:2] == (5, 6)


def test_hf_backend_importable_without_heavy_deps() -> None:
    assert not any(m in sys.modules for m in ("vllm", "outlines"))
    hf = I.HFBackend("released")
    assert hf.name == "hf" and hf.adapter is None
    assert I.HFBackend("data/runs/force").adapter == "data/runs/force"
    assert isinstance(I.make_backend("vllm"), I.VLLMBackend)
    assert isinstance(hf, I.Backend) and isinstance(I.StubBackend(), I.Backend)
    b = build_prompt(window("outcome"), None)
    with pytest.raises(RuntimeError):  # fails before any heavy import
        hf.generate(to_messages(b), b.video, regex=None, max_new_tokens=12, top_logprobs=20)
    with pytest.raises(ValueError):
        I.make_backend("nope")


def test_mosaic_frames_layout_and_blank() -> None:
    ep = make_pad_episode(0)
    b = build_prompt(window("outcome"), None, camera_roles=("scene",), fps=4.0)
    clip = I.mosaic_frames(ep, b.video, SyntheticFrameReader())
    assert clip.shape == (8, 96, 96, 3) and clip.dtype == np.uint8 and clip.std() > 0
    blank = I.mosaic_frames(ep, b.video.model_copy(update={"blank": True}), SyntheticFrameReader())
    assert blank.shape == clip.shape and set(np.unique(blank)) == {128}
