# pyright: reportMissingImports=false
"""Model backends for FLORA-Rewarder inference: a Protocol, a stub, an HF and a vLLM backend.

Owns: Backend, GenerateResult, StubBackend, HFBackend, VLLMBackend, expected_score, progress,
clip_array and the decoding-budget constants.
Must never: import torch / transformers / vllm / outlines at module level (the tests, the CLI
help and CI import this module on machines without them), build prompts, or parse answers.

MILESTONE 7 PASSED (2026-10-21). HFBackend generated parsable dense and outcome answers,
with force omitted and included, on one real PHANTOM episode across two MI210s. Tested:
torch 2.12.1+rocm7.1, HIP 7.1.52802, transformers 5.8.1, Triton ROCm 3.7.1. DeltaNet used
the PyTorch fallback; flash-linear-attention and causal-conv1d were not installed. This
checks loading and output parsing, not model quality, production training or the fast kernels.
VLLMBackend remains untested. Details and artifacts are in scripts/cluster/README.md.
Released requirements (nvidia/Flora-Rewarder batch_utils/requirements.txt): vllm 0.24.0,
transformers 5.8.1, qwen-vl-utils 0.0.14 (plus decord 0.6.0, av >= 15, Pillow < 13). The
released vLLM run uses bf16, ``gdn_prefill_backend="triton"``, prefix caching, structured
outputs by regex, ``mm_processor_kwargs={"do_sample_frames": False}`` and per-video metadata
{total_num_frames, fps, frames_indices, duration} so the timestamp
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
from tactile_flora.video.mosaic import (
    VideoMetadataDict,
    clip_frames,
    source_frame_count,
    video_metadata,
)
from tactile_flora.video.readers import AutoReader

DENSE_MAX_NEW_TOKENS = 640
SCORE_ONLY_MAX_NEW_TOKENS = 12
RATIONALE_MAX_NEW_TOKENS = 160
TOP_LOGPROBS = 20
DIGITS = ("1", "2", "3", "4", "5")
MODEL_ID = "nvidia/Flora-Rewarder"
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


def clip_array(episode: Episode, video: VideoSpec, reader: FrameReader) -> np.ndarray:
    """Return the (T, H, W, 3) uint8 clip for ``video`` via video.mosaic.clip_frames."""
    frames = clip_frames(
        episode,
        reader,
        video.frame_times,
        video.roles,
        blank=video.blank,
        cell_px=video.cell_px,
    )
    return np.stack(frames)


def _video_metadata(video: VideoSpec, episode: Episode) -> VideoMetadataDict:
    return video_metadata(video.frame_indices, video.fps, source_frame_count(episode))


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
        self.reader: FrameReader = reader if reader is not None else AutoReader()
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
        if episode is None:
            raise RuntimeError("HFBackend needs the episode to read video")
        import torch  # pyright: ignore[reportMissingImports]

        self.load()
        frames = clip_array(episode, video, self.reader)
        prompt = self._processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        meta = _video_metadata(video, episode)
        inputs = self._processor(
            text=[prompt],
            videos=[frames],
            videos_kwargs={"video_metadata": [meta], "do_sample_frames": False},
            return_tensors="pt",
        ).to(self._model.device)
        tok = self._processor.tokenizer
        # The released generation config names endoftext, while its chat tokenizer ends
        # assistant turns with im_end. Stop on either instead of generating more turns.
        eos = self._model.generation_config.eos_token_id
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
            eos_ids.append(tok.eos_token_id)
        t0 = time.perf_counter()
        # DeltaNet parameters stay fp32 while activations and the remaining weights are
        # bf16. Autocast reconciles projection inputs without changing stored weights.
        with (
            torch.no_grad(),
            torch.autocast(device_type=self._model.device.type, dtype=torch.bfloat16),
        ):
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids or None,
                pad_token_id=tok.pad_token_id,
                output_scores=top_logprobs > 0,
                return_dict_in_generate=True,
                logits_processor=self._processor_logits(regex) or None,
            )
        latency = time.perf_counter() - t0
        new_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
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

    Untested; milestone 7 was run through HFBackend. vllm is imported inside
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
        self.reader: FrameReader = reader if reader is not None else AutoReader()
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
        if episode is None:
            raise RuntimeError("VLLMBackend needs the episode to read video")
        self.load()
        from vllm import SamplingParams  # type: ignore[import-not-found]
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore[import-not-found]

        frames = clip_array(episode, video, self.reader)
        prompt = self._processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            logprobs=top_logprobs or None,
            structured_outputs=StructuredOutputsParams(regex=regex) if regex else None,
        )
        meta = _video_metadata(video, episode)
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
