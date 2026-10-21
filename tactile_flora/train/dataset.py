"""Turning a manifest into tokenized samples with answer-only loss masks.

Owns: Sample, WindowItem, the Tokenizer Protocol, QwenTokenizerAdapter (lazy transformers),
build_samples, token_histogram, estimate_wallclock and adapter_windows_source.
Must never: read raw datasets directly (episodes come only through a ``windows_source``, by
default the manifest's source adapters re-run through verify.windowing), write files, or import
torch/transformers at module import time.

Windows: the manifest stores rows, not Windows (verify/manifest.py has no windows parquet), so
each row is resolved to a ``WindowItem`` by ``windows_source(row)``. The default rebuilds the
episode's windows with ``verify.windowing.episode_windows`` and the fitted ``norm.json``
(read-only use of verify) and matches on ``window_id``.

Force mode: with ``item.features`` (normalized ResampledForce) the mode goes through
``eval.ablations.apply_force_mode`` and ``force.render``. Without features only ``real`` (the
window's own force lines) and ``omitted`` are possible. Every sample asserts, via
``prompts.flora.prompt_diff``, that it differs from the control prompt only in the force block.

Masking: the prompt (chat template with generation prompt) and prompt+answer are tokenized
separately; labels are -100 everywhere except the answer tokens. Samples are returned sorted by
n_tokens (packing-free; batch size 1 per GPU, gradient accumulation in sft.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from tactile_flora.config import ForceConfig, ForceMode
from tactile_flora.eval.ablations import apply_force_mode
from tactile_flora.force import normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.prompts.flora import VideoSpec, build_prompt, prompt_diff, to_messages
from tactile_flora.schema import Episode, ManifestRow, ResampledForce, Window
from tactile_flora.verify import manifest as manifest_mod
from tactile_flora.video.mosaic import VISUAL_TOKEN_CAP

IGNORE_INDEX = -100


class Tokenizer(Protocol):
    """Chat tokenizer; the real one is QwenTokenizerAdapter, tests use a stub."""

    def encode_prompt(self, messages: Sequence[dict[str, Any]], video: VideoSpec) -> list[int]:
        """Token ids of the chat-templated prompt, ending at the generation prompt."""
        ...

    def encode_with_answer(
        self, messages: Sequence[dict[str, Any]], answer: str, video: VideoSpec
    ) -> list[int]:
        """Token ids of the same prompt followed by the answer (and end-of-turn)."""
        ...


class WindowItem(BaseModel):
    """One resolved manifest row: the Window plus what the prompt builder needs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: Window
    task: str = ""
    camera_roles: tuple[str, ...] = ("scene",)
    fps: float | None = None
    features: ResampledForce | None = None  # normalized features at window.frame_times
    episode: Episode | None = None  # where the video frames come from


WindowsSource = Callable[[ManifestRow], WindowItem | None]


class Sample(BaseModel):
    """One tokenized training sample."""

    window_id: str
    messages: list[dict[str, Any]]
    video: VideoSpec
    episode: Episode | None = None
    force_block: str | None
    target: str
    input_ids: list[int]
    labels: list[int]
    n_tokens: int


def answer_labels(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> list[int]:
    """Labels for ``full_ids``: -100 over the prompt prefix, the token id over the answer."""
    n = len(prompt_ids)
    if len(full_ids) <= n or list(full_ids[:n]) != list(prompt_ids):
        raise ValueError(
            "prompt+answer tokens must extend the prompt tokens with a non-empty answer"
        )
    return [IGNORE_INDEX] * n + list(full_ids[n:])


def _block(item: WindowItem, mode: ForceMode, pool: Sequence[ResampledForce], seed: int,
           force_cfg: ForceConfig) -> tuple[list[str], bool]:  # fmt: skip
    """Return (force lines to render, video_blank); empty lines mean no block."""
    w = item.window
    if item.features is None:
        if mode == "omitted":
            return [], False
        if mode in ("real", "video_blank"):
            if not w.force_lines:
                raise ValueError(f"{w.episode_ref}: force_mode {mode} but the window has no force")
            return list(w.force_lines), mode == "video_blank"
        raise ValueError(f"force_mode {mode} needs resampled features (WindowItem.features)")
    ab = apply_force_mode(item.features, mode, pool=pool, seed=seed)
    if ab.features is None:
        return [], ab.video_blank
    lines = render.render_lines(ab.features, w.frame_times, item.features.channel)
    return lines, ab.video_blank


def build_samples(
    manifest_path: Path | str,
    force_mode: ForceMode,
    force_cfg: ForceConfig,
    data_root: Path,
    *,
    tokenizer: Tokenizer,
    windows_source: WindowsSource | None = None,
    seed: int = 0,
    splits: Sequence[str] = ("train",),
) -> list[Sample]:
    """Build tokenized samples for every manifest row in ``splits``, sorted by n_tokens."""
    rows = [r for r in manifest_mod.read_manifest(manifest_path) if r.split in splits]
    source = windows_source or adapter_windows_source(data_root, force_cfg)
    items: list[tuple[ManifestRow, WindowItem]] = []
    for row in rows:
        item = source(row)
        if item is not None:
            items.append((row, item))
    pool = [it.features for _, it in items if it.features is not None]
    samples: list[Sample] = []
    for row, item in items:
        w = item.window
        lines, blank = _block(item, force_mode, pool, seed, force_cfg)
        block = render.render_block(lines, force_cfg.render) if lines else None
        kw: dict[str, Any] = {
            "task": item.task,
            "camera_roles": item.camera_roles,
            "fps": item.fps,
        }
        if block is not None:
            prompt_diff(
                w.model_copy(update={"force_lines": tuple(lines)}),
                header=force_cfg.render.header,
                task=item.task,
                camera_roles=item.camera_roles,
            )
        bundle = build_prompt(w, block, video_blank=blank, **kw)
        messages = to_messages(bundle)
        prompt_ids = tokenizer.encode_prompt(messages, bundle.video)
        full_ids = tokenizer.encode_with_answer(messages, w.target, bundle.video)
        samples.append(
            Sample(
                window_id=row.window_id,
                messages=messages,
                video=bundle.video,
                episode=item.episode,
                force_block=block,
                target=w.target,
                input_ids=list(full_ids),
                labels=answer_labels(prompt_ids, full_ids),
                n_tokens=len(full_ids),
            )
        )
    return sorted(samples, key=lambda s: (s.n_tokens, s.window_id))


def token_histogram(samples: Sequence[Sample]) -> dict[str, float]:
    """Summary of sample lengths: min, p50, p90, max and total tokens."""
    if not samples:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "total": 0.0}
    n = np.asarray([s.n_tokens for s in samples], dtype=np.float64)
    return {
        "min": float(n.min()),
        "p50": float(np.percentile(n, 50)),
        "p90": float(np.percentile(n, 90)),
        "max": float(n.max()),
        "total": float(n.sum()),
    }


def estimate_wallclock(samples: Sequence[Sample], s_per_window: float) -> float:
    """Estimated seconds for one epoch: one optimizer-visible window costs ``s_per_window``."""
    return len(samples) * s_per_window


def adapter_windows_source(data_root: Path, force_cfg: ForceConfig) -> WindowsSource:
    """Default ``windows_source``: rebuild windows with the manifest's source adapter.

    Reads data_root/verified/<source>/norm.json; resolves rows by ``window_id``. Episodes are
    cached one at a time per source (rows are sorted by episode in the manifest).
    """
    from tactile_flora.sources.base import get_source
    from tactile_flora.verify.windowing import episode_windows

    adapters: dict[str, Any] = {}
    consts: dict[str, normalize.NormConstants] = {}
    cache: dict[tuple[str, str], dict[str, WindowItem]] = {}

    def source(row: ManifestRow) -> WindowItem | None:
        key = (row.source, row.episode_id)
        if key not in cache:
            cache.clear()
            if row.source not in adapters:
                adapters[row.source] = get_source(row.source)()
                consts[row.source] = normalize.load(
                    data_root / "verified" / row.source / "norm.json"
                )
            ep = adapters[row.source].load(row.episode_id)
            c = consts[row.source]
            cam = ep.cameras.get("scene")
            out: dict[str, WindowItem] = {}
            for w in episode_windows(ep, force_cfg, c, row.split):
                feats = normalize.apply(
                    resample(ep.force, ep.gripper, list(w.frame_times), force_cfg.resample),
                    c,
                    force_cfg.render,
                )
                wid = manifest_mod.window_id(ep.source, ep.episode_id, w.prompt_kind, w.anchor_s)
                out[wid] = WindowItem(
                    window=w,
                    task=ep.task,
                    camera_roles=tuple(ep.cameras) or ("scene",),
                    fps=cam.fps if cam is not None else None,
                    features=feats,
                    episode=ep,
                )
            cache[key] = out
        return cache[key].get(row.window_id)

    return source


class QwenTokenizerAdapter:
    """Qwen3.5 processor behind the Tokenizer Protocol; loaded lazily, never used in tests.

    The chat template renders a single video placeholder; it is expanded to ``visual_tokens``
    video pad tokens so lengths include the clip (real frames are decoded only at train time,
    where the processor produces the exact count). Answer tokens are found by tokenizing the
    prompt and the prompt+answer separately. Unverified against the real checkpoint.
    """

    def __init__(self, model_id: str, visual_tokens: int = VISUAL_TOKEN_CAP) -> None:
        self.model_id = model_id
        self.visual_tokens = visual_tokens
        self._processor: Any = None

    @property
    def processor(self) -> Any:
        """The transformers AutoProcessor (imported and loaded on first use)."""
        if self._processor is None:
            from transformers import AutoProcessor  # type: ignore[import-not-found]

            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    def _text(self, messages: Sequence[dict[str, Any]], answer: str | None) -> str:
        msgs = list(messages)
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        text: str = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=answer is None
        )
        return text

    def _ids(self, text: str, video: VideoSpec) -> list[int]:
        pad = "<|video_pad|>"
        n = min(self.visual_tokens, VISUAL_TOKEN_CAP)
        text = text.replace(pad, pad * n, 1) if pad in text else text
        ids: list[int] = self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def encode_prompt(self, messages: Sequence[dict[str, Any]], video: VideoSpec) -> list[int]:
        """Prompt ids with the generation prompt."""
        return self._ids(self._text(messages, None), video)

    def encode_with_answer(
        self, messages: Sequence[dict[str, Any]], answer: str, video: VideoSpec
    ) -> list[int]:
        """Prompt+answer ids."""
        return self._ids(self._text(messages, answer), video)
