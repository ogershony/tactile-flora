# pyright: reportMissingImports=false
"""Model backends for FLORA-Rewarder inference: a Protocol, a stub, an HF and a vLLM backend.

Owns: Backend, GenerateResult, StubBackend, HFBackend, VLLMBackend, expected_score, progress,
mosaic_frames and the decoding-budget constants.
Must never: import torch / transformers / vllm / outlines at module level (the tests, the CLI
help and CI import this module on machines without them), build prompts, or parse answers.

MILESTONE 7 GATE (not yet passed). HFBackend and VLLMBackend are written from the released
inference code and the paper but have NOT been run against the real checkpoint: the first
real run is milestone 7, "the released model produces a parsable dense output on one window"
on the MI210s (ROCm; Gated DeltaNet kernels come from the Triton flash-linear-attention
library, gfx90a is supported but not the tested path) with the cloud RTX Pro 6000 as the
fallback. Record the torch / ROCm / transformers / FLA versions that worked in the README.
Released requirements (nvidia/Flora-Rewarder batch_utils/requirements.txt): vllm 0.24.0,
transformers 5.8.1, qwen-vl-utils 0.0.14 (plus decord 0.6.0, av >= 15, Pillow < 13). The
released vLLM run uses bf16, ``gdn_prefill_backend="triton"``, prefix caching, structured
outputs by regex, ``mm_processor_kwargs={"do_sample_frames": False}`` and per-video metadata
{total_num_frames, fps, frames_indices, duration, do_sample_frames: False} so the timestamp
tokens are absolute episode seconds. Everything here mirrors those settings.

Decoding (paper): greedy, thinking disabled, 640 new tokens for the dense grammar, 12 for
score-only outcomes, 160 for outcome with rationale, top-20 logprobs at the Score token.
Constrained decoding in HFBackend uses ``outlines`` if importable; otherwise the text is
validated afterwards by prompts/parse.py (a violation becomes a parse failure, which is what
the paper does with an invalid answer).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from tactile_flora.prompts.flora import VideoSpec
from tactile_flora.schema import Episode, FrameReader

DENSE_MAX_NEW_TOKENS = 640
SCORE_ONLY_MAX_NEW_TOKENS = 12
RATIONALE_MAX_NEW_TOKENS = 160
TOP_LOGPROBS = 20
DIGITS = ("1", "2", "3", "4", "5")
MODEL_ID = "nvidia/Flora-Rewarder"
BLANK_GRAY = 128
# Substrings of the released system prompts, used by the stub to tell the tasks apart.
_DENSE_MARK = "dense action-quality reward model"
_SCORE_ONLY_MARK = "Do not provide a rationale"


class GenerateResult(BaseModel):
    """One generation: the text, logprobs at the Score token (digit -> logprob) and latency."""

    text: str
    score_token_logprobs: dict[str, float] | None = None
    latency_s: float = 0.0


@runtime_checkable
class Backend(Protocol):
    """A model that answers one FLORA prompt.

    ``episode`` is the source of the video frames for backends that decode video (the stub
    ignores it). ``regex`` is the answer grammar for constrained decoding or validation.
    """

    name: str

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        video: VideoSpec,
        *,
        regex: str | None,
        max_new_tokens: int,
        top_logprobs: int,
        episode: Episode | None = None,
    ) -> GenerateResult:
        """Return the greedy answer for ``messages`` over the video described by ``video``."""
        ...


def expected_score(logprobs: Mapping[str, float]) -> float:
    """E[s] from score-token logprobs: keep digits 1-5, renormalize, take the mean.

    Raises ValueError if no digit 1-5 is present. Keys may carry whitespace (" 4").
    """
    mass = dict.fromkeys(DIGITS, 0.0)
    for tok, lp in logprobs.items():
        key = tok.strip()
        if key in mass:
            mass[key] += math.exp(lp)
    total = sum(mass.values())
    if total <= 0.0:
        raise ValueError("no digit 1-5 among the score-token logprobs")
    return sum(int(k) * v for k, v in mass.items()) / total


def progress(expected: float) -> float:
    """Rescale E[s] in [1, 5] to a [0, 1] progress value: (E - 1) / 4."""
    return (expected - 1.0) / 4.0


def dense_answer(digits: str) -> str:
    """Return a grammar-valid dense answer for 32 quality ``digits`` (one rationale per run)."""
    if len(digits) != 32 or set(digits) - set("123"):
        raise ValueError("digits must be 32 characters from 1, 2, 3")
    lines = [f"Worst: {min(int(c) for c in digits)}", "Quality:", digits, "Rationales:"]
    g = 0
    while g < 32:
        h = g
        while h + 1 < 32 and digits[h + 1] == digits[g]:
            h += 1
        if digits[g] != "3":
            lines.append(f"{g + 1}-{h + 1} stub rationale")
        g = h + 1
    return "\n".join(lines)


class StubBackend:
    """Deterministic backend for tests and wiring checks; never touches video or a model.

    Dense prompts get ``dense_digits`` (default all normal); outcome prompts get
    ``outcome_score`` in the format the prompt asked for, with logprobs over "1".."5" from
    ``logprobs`` (default: mass 0.9 on the answer, the rest spread evenly). ``text_fn``
    replaces the answer text entirely (it receives the messages and the VideoSpec).
    """

    name = "stub"

    def __init__(
        self,
        *,
        outcome_score: int = 5,
        logprobs: Mapping[str, float] | None = None,
        dense_digits: str = "3" * 32,
        text_fn: Callable[[Sequence[Mapping[str, Any]], VideoSpec], str] | None = None,
    ) -> None:
        """Fix the answers; ``logprobs`` of None derives them from ``outcome_score``."""
        if not 1 <= outcome_score <= 5:
            raise ValueError("outcome_score must be 1..5")
        self.outcome_score = outcome_score
        self.logprobs = dict(logprobs) if logprobs is not None else None
        self.dense_digits = dense_digits
        self.text_fn = text_fn
        self.calls = 0

    def _logprobs(self) -> dict[str, float]:
        if self.logprobs is not None:
            return dict(self.logprobs)
        rest = math.log(0.1 / 4)
        return {d: (math.log(0.9) if int(d) == self.outcome_score else rest) for d in DIGITS}

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        video: VideoSpec,
        *,
        regex: str | None,
        max_new_tokens: int,
        top_logprobs: int,
        episode: Episode | None = None,
    ) -> GenerateResult:
        """Return the fixed answer for the prompt kind found in the system message."""
        self.calls += 1
        system = str(messages[0]["content"][0]["text"])
        dense = _DENSE_MARK in system
        if self.text_fn is not None:
            text = self.text_fn(messages, video)
        elif dense:
            text = dense_answer(self.dense_digits)
        elif _SCORE_ONLY_MARK in system:
            text = f"Score: {self.outcome_score}"
        else:
            text = f"Rationale: stub rationale\n\nScore: {self.outcome_score}"
        lp = self._logprobs() if (not dense and top_logprobs > 0) else None
        return GenerateResult(text=text, score_token_logprobs=lp, latency_s=0.0)


def mosaic_frames(
    episode: Episode,
    video: VideoSpec,
    reader: FrameReader,
) -> np.ndarray:
    """Tile the cameras of ``video.roles`` into one (T, H, W, 3) uint8 mosaic clip.

    Each view is resized to a ``cell_px`` square-bounded cell (aspect kept, padded with black)
    and placed row-major in ``video.layout`` (scene first = upper left). ``video.blank``
    replaces every frame with mid-gray of the same shape (the video_blank ablation).
    """
    from PIL import Image

    rows, cols = video.layout
    cell = video.cell_px
    frames = np.zeros((len(video.frame_indices), rows * cell, cols * cell, 3), dtype=np.uint8)
    if video.blank:
        frames[:] = BLANK_GRAY
        return frames
    for t, idx in enumerate(video.frame_indices):
        for k, role in enumerate(video.roles):
            img = Image.fromarray(reader.read_frame(episode, role, idx))
            img.thumbnail((cell, cell))
            r, c = divmod(k, cols)
            frames[t, r * cell : r * cell + img.height, c * cell : c * cell + img.width] = (
                np.asarray(img)
            )
    return frames


def _video_metadata(video: VideoSpec, n_frames: int) -> dict[str, Any]:
    """Frame-clock metadata so timestamp tokens are absolute episode seconds."""
    return {
        "total_num_frames": int(n_frames),
        "fps": float(video.fps),
        "frames_indices": [int(i) for i in video.frame_indices],
        "duration": float(n_frames) / float(video.fps),
        "do_sample_frames": False,
    }


def _scene_frames(episode: Episode) -> int:
    cam = episode.cameras.get("scene") or next(iter(episode.cameras.values()))
    return cam.n_frames


def _merge_digit_logprobs(entries: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Merge (token text, logprob) pairs into digit -> logprob (log-sum-exp over duplicates)."""
    out: dict[str, float] = {}
    for tok, lp in entries:
        key = tok.strip()
        if key in DIGITS:
            out[key] = float(np.logaddexp(out[key], lp)) if key in out else float(lp)
    return out


class HFBackend:
    """Hugging Face transformers backend for nvidia/Flora-Rewarder (Qwen3.5-27B).

    Heavy imports happen in ``load`` / ``generate`` only. Loads
    Qwen3_5ForConditionalGeneration in bf16 with the Gated DeltaNet parameters promoted to
    fp32 (the released inference precision). A LoRA adapter directory (``adapter``) is merged
    on top for trained checkpoints.
    """

    name = "hf"

    def __init__(
        self,
        checkpoint: str = "released",
        reader: FrameReader | None = None,
        *,
        model_id: str = MODEL_ID,
        adapter: str | None = None,
        device_map: str = "auto",
    ) -> None:
        """Remember what to load; nothing heavy happens until the first ``generate``."""
        self.checkpoint = checkpoint
        self.reader = reader
        self.model_id = model_id
        self.adapter = (
            adapter if adapter is not None else (None if checkpoint == "released" else checkpoint)
        )
        self.device_map = device_map
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load processor and model (idempotent). Requires the [train] extra."""
        if self._model is not None:
            return
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # type: ignore[attr-defined]

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            self.model_id, dtype=torch.bfloat16, device_map=self.device_map
        )
        # Gated DeltaNet parameters stay fp32 (spec: "bf16 elsewhere"); matched by module name.
        for name, param in model.named_parameters():  # pyright: ignore[reportUnknownMemberType]
            if any(k in name for k in ("linear_attn", "delta", "A_log", "dt_bias")):
                param.data = param.data.float()
        if self.adapter is not None:
            from peft import PeftModel  # pyright: ignore[reportMissingImports]

            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload()
        model.eval()
        self._model = model

    def _processor_logits(self, regex: str | None) -> list[Any]:
        """Regex logits processors via outlines when available, else none (post-hoc check)."""
        if regex is None:
            return []
        try:
            from outlines.models.transformers import TransformerTokenizer  # type: ignore[import-not-found]
            from outlines.processors import RegexLogitsProcessor  # type: ignore[import-not-found]
        except ImportError:
            return []
        tok = TransformerTokenizer(self._processor.tokenizer)
        return [RegexLogitsProcessor(regex, tok)]  # pyright: ignore[reportUnknownVariableType]

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        video: VideoSpec,
        *,
        regex: str | None,
        max_new_tokens: int,
        top_logprobs: int,
        episode: Episode | None = None,
    ) -> GenerateResult:
        """Greedy-decode one answer; ``score_token_logprobs`` holds digits found in the top-k."""
        if episode is None or self.reader is None:
            raise RuntimeError("HFBackend needs an episode and a FrameReader to read video")
        import torch  # pyright: ignore[reportMissingImports]

        self.load()
        frames = mosaic_frames(episode, video, self.reader)
        prompt = self._processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        meta = _video_metadata(video, _scene_frames(episode))
        inputs = self._processor(
            text=[prompt],
            videos=[frames],
            videos_kwargs={"video_metadata": [meta], "do_sample_frames": False},
            return_tensors="pt",
        ).to(self._model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=top_logprobs > 0,
                return_dict_in_generate=True,
                logits_processor=self._processor_logits(regex) or None,
            )
        latency = time.perf_counter() - t0
        new_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
        tok = self._processor.tokenizer
        text = str(tok.decode(new_ids, skip_special_tokens=True))
        lp: dict[str, float] | None = None
        if top_logprobs > 0 and getattr(out, "scores", None):
            for pos in range(len(new_ids) - 1, -1, -1):  # last generated digit = the score
                if tok.decode(new_ids[pos]).strip() in DIGITS:
                    logp = torch.log_softmax(out.scores[pos][0].float(), dim=-1)
                    top = torch.topk(logp, top_logprobs)
                    lp = _merge_digit_logprobs(
                        [
                            (tok.decode(int(i)), float(v))
                            for v, i in zip(top.values, top.indices, strict=True)
                        ]
                    )
                    break
        return GenerateResult(text=text, score_token_logprobs=lp or None, latency_s=latency)


class VLLMBackend:
    """vLLM backend mirroring the released engine (offline ``LLM`` with regex structured output).

    Sketch, same status as HFBackend (unrun until milestone 7). vllm is imported inside
    ``load``; install vllm==0.24.0 alongside transformers==5.8.1 and qwen-vl-utils==0.0.14.
    """

    name = "vllm"

    def __init__(
        self,
        checkpoint: str = "released",
        reader: FrameReader | None = None,
        *,
        model_id: str = MODEL_ID,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Remember engine settings; the engine is created on first use."""
        self.checkpoint = checkpoint
        self.reader = reader
        self.model_id = model_id if checkpoint == "released" else checkpoint
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._llm: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Create the vLLM engine (bf16, triton GDN prefill, video-only multimodal limit)."""
        if self._llm is not None:
            return
        from transformers import AutoProcessor  # pyright: ignore[reportMissingImports]
        from vllm import LLM  # type: ignore[import-not-found]

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            limit_mm_per_prompt={"image": 0, "video": 1},
            mm_processor_kwargs={"do_sample_frames": False},
            gdn_prefill_backend="triton",
            enable_prefix_caching=True,
            max_logprobs=TOP_LOGPROBS,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        video: VideoSpec,
        *,
        regex: str | None,
        max_new_tokens: int,
        top_logprobs: int,
        episode: Episode | None = None,
    ) -> GenerateResult:
        """Greedy, regex-constrained generation through vLLM with top-k logprobs."""
        if episode is None or self.reader is None:
            raise RuntimeError("VLLMBackend needs an episode and a FrameReader to read video")
        self.load()
        from vllm import SamplingParams  # type: ignore[import-not-found]
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore[import-not-found]

        frames = mosaic_frames(episode, video, self.reader)
        prompt = self._processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            logprobs=top_logprobs or None,
            structured_outputs=StructuredOutputsParams(regex=regex) if regex else None,
        )
        meta = _video_metadata(video, _scene_frames(episode))
        t0 = time.perf_counter()
        (res,) = self._llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"video": (frames, meta)},
                    "mm_processor_kwargs": {"do_sample_frames": False},
                }
            ],
            params,
            use_tqdm=False,
        )
        latency = time.perf_counter() - t0
        comp = res.outputs[0]
        tok = self._llm.get_tokenizer()
        lp: dict[str, float] | None = None
        if top_logprobs and comp.logprobs:
            for token, cands in zip(reversed(comp.token_ids), reversed(comp.logprobs), strict=True):
                if tok.decode([token]).strip() in DIGITS:
                    lp = _merge_digit_logprobs(
                        [(tok.decode([i]), float(c.logprob)) for i, c in cands.items()]
                    )
                    break
        return GenerateResult(text=comp.text, score_token_logprobs=lp or None, latency_s=latency)


def make_backend(
    kind: str, checkpoint: str = "released", reader: FrameReader | None = None
) -> Backend:
    """Return the backend named ``kind`` ("stub", "hf" or "vllm") for ``checkpoint``."""
    if kind == "stub":
        return StubBackend()
    if kind == "hf":
        return HFBackend(checkpoint, reader)
    if kind == "vllm":
        return VLLMBackend(checkpoint, reader)
    raise ValueError(f"unknown backend {kind!r}; choose stub, hf or vllm")
