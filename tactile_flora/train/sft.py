"""The LoRA SFT loop and the run directory it writes.

Owns: train (entry point), train_steps (the loop body), make_scheduler, the Backend Protocol and
HFBackend (the real Qwen3.5 + LoRA + FSDP model, imported lazily), TrainRefused.
Must never: start without holdout_ids.json or with a stale report_version; read raw datasets;
import torch, transformers or peft at module import time (they are the [train] extra).

Run directory ``<cfg.output_dir>/<run_name>/``: config.yaml (resolved copy), manifest_hash.txt,
git_hash.txt, loss.csv (every optimizer step), log.jsonl (loss, lr, tok/s, GPU memory every 10
steps) and adapter/. ``--dry-run`` writes dry_run.json (token histogram, ETA) instead of training.

Refusals: no data/manifest/holdout_ids.json, or any manifest row whose report_version differs
from the current full git hash. ``allow_stale_manifest`` (CLI ``--allow-stale-manifest``) skips
only the second check and is DEV-ONLY: a committed-then-edited repo never produces a publishable
run with it.

Loss: cross-entropy over answer tokens only (labels != -100), mean per micro-batch, divided by
the accumulation count; gradient clip, AdamW, cosine schedule with linear warmup. Samples are
length-sorted by dataset.py, grouped into global batches of ``effective_batch`` (similar lengths
per step), the groups shuffled with ``cfg.seed``, and each rank takes a stride of its group.

GPU wheels: install the [train] extra from the torch index matching the machine, selected by
the environment variable ``TF_TORCH_INDEX`` (for example https://download.pytorch.org/whl/rocm6.2
on the MI210s or .../cu124 on the cloud GPU); see README "Training".
"""

from __future__ import annotations

import copy
import csv
import importlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from tactile_flora.config import ForceConfig, TrainConfig
from tactile_flora.schema import FrameReader
from tactile_flora.train.dataset import (
    IGNORE_INDEX,
    QwenTokenizerAdapter,
    Sample,
    Tokenizer,
    WindowsSource,
    build_samples,
    estimate_wallclock,
    token_histogram,
)
from tactile_flora.verify import full_git_hash, holdout_path, manifest
from tactile_flora.video.mosaic import clip_frames, source_frame_count, video_metadata
from tactile_flora.video.readers import SourceRootReader

log = logging.getLogger("tactile_flora.train")

DEFAULT_S_PER_WINDOW = 9.0  # spec: 8 to 10 s per window on 5x MI210
LOG_EVERY = 10


class TrainRefused(RuntimeError):
    """Raised when a precondition (holdout file, fresh manifest) is not met."""


class Backend(Protocol):
    """What the loop needs from a model stack; HFBackend is the real one."""

    model: Any
    device: Any
    world_size: int
    rank: int

    def trainable_parameters(self) -> list[Any]:
        """Parameters with requires_grad (LoRA only)."""
        ...

    def make_batch(self, sample: Sample) -> dict[str, Any]:
        """Tensors for one sample: input_ids and labels of shape (1, L) plus model extras."""
        ...

    def save_adapter(self, path: Path) -> None:
        """Write the adapter weights to ``path``."""
        ...


BackendFactory = Callable[[TrainConfig], Backend]


def _torch() -> Any:
    return importlib.import_module("torch")


def make_scheduler(optimizer: Any, total_steps: int, warmup_frac: float) -> Any:
    """Linear warmup over ``warmup_frac`` of the steps, then cosine decay to zero."""
    warm = max(1, round(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return _torch().optim.lr_scheduler.LambdaLR(optimizer, factor)


def _answer_loss(logits: Any, labels: Any, kept: int | None) -> Any:
    """Mean cross-entropy over answer tokens; ``kept`` is the number of trailing logits kept."""
    torch = _torch()
    if kept is not None:
        labels = labels[:, -kept:]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def _first_answer(labels: Any) -> int:
    idx = (labels[0] != IGNORE_INDEX).nonzero()
    return int(idx[0, 0])


def train_steps(
    model: Any,
    batches: Iterable[dict[str, Any]],
    optimizer: Any,
    scheduler: Any,
    *,
    accum: int,
    grad_clip: float,
    run_dir: Path | None = None,
    log_every: int = LOG_EVERY,
    device: Any = None,
    keep_logits: bool = False,
    params: Sequence[Any] | None = None,
    autocast_bf16: bool = False,
) -> list[float]:
    """Run the loop over ``batches`` (one micro-batch each); return the loss of every step.

    Every ``accum`` micro-batches make one optimizer step (a trailing partial group steps too).
    ``keep_logits`` passes ``logits_to_keep`` so only the answer suffix is projected to the
    vocabulary (HF models). With ``run_dir`` it writes loss.csv and log.jsonl.
    """
    torch = _torch()
    model.train()
    clip_params = list(params) if params is not None else list(model.parameters())
    losses: list[float] = []
    csv_fh = jsonl_fh = None
    writer: Any = None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_fh = open(run_dir / "loss.csv", "w", newline="", encoding="utf-8")  # noqa: SIM115
        writer = csv.writer(csv_fh)
        writer.writerow(["step", "loss", "lr"])
        jsonl_fh = open(run_dir / "log.jsonl", "w", encoding="utf-8")  # noqa: SIM115
    it: Iterator[dict[str, Any]] = iter(batches)
    step = 0
    tokens = 0
    t_last = time.monotonic()
    try:
        while True:
            group = list(itertools.islice(it, accum))
            if not group:
                break
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for batch in group:
                if device is not None:
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                labels = batch["labels"]
                weight = float(batch.get("loss_weight", 1.0))
                inputs = {k: v for k, v in batch.items() if k not in ("labels", "loss_weight")}
                kept: int | None = None
                if keep_logits:
                    kept = labels.size(1) - max(0, _first_answer(labels) - 1)
                    inputs["logits_to_keep"] = kept
                with torch.autocast(
                    device_type=torch.device(device or "cpu").type,
                    dtype=torch.bfloat16,
                    enabled=autocast_bf16,
                ):
                    out = model(**inputs)
                    logits = getattr(out, "logits", out)
                    loss = _answer_loss(logits, labels, kept) * weight / len(group)
                loss.backward()
                total += float(loss.detach())
                tokens += int(labels.size(1))
            clip = getattr(model, "clip_grad_norm_", None)
            if callable(clip):
                clip(grad_clip)  # FSDP computes the norm across shards, including nested wrappers.
            else:
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
            optimizer.step()
            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            step += 1
            losses.append(total)
            if writer is not None and csv_fh is not None:
                writer.writerow([step, f"{total:.6f}", f"{lr:.3e}"])
                csv_fh.flush()
            if jsonl_fh is not None and step % log_every == 0:
                now = time.monotonic()
                mem = (
                    torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                )
                rec = {
                    "step": step,
                    "loss": total,
                    "lr": lr,
                    "tok_s": tokens / max(now - t_last, 1e-9),
                    "gpu_mem_gb": mem,
                }
                jsonl_fh.write(json.dumps(rec) + "\n")
                jsonl_fh.flush()
                log.info("step %d loss %.4f lr %.2e tok/s %.0f", step, total, lr, rec["tok_s"])
                t_last, tokens = now, 0
    finally:
        for fh in (csv_fh, jsonl_fh):
            if fh is not None:
                fh.close()
    return losses


def ordered_groups(
    samples: Sequence[Sample], effective_batch: int, seed: int
) -> list[list[Sample]]:
    """Chunk length-sorted samples into global batches, then shuffle the chunks with ``seed``."""
    groups = [
        list(samples[i : i + effective_batch]) for i in range(0, len(samples), effective_batch)
    ]
    tail = groups.pop() if groups and len(groups[-1]) < effective_batch else None
    random.Random(seed).shuffle(groups)
    if tail:
        groups.append(tail)  # keep a partial group last so accumulation cannot cross its boundary
    return groups


def _batches(
    groups: Sequence[Sequence[Sample]], backend: Backend, epochs: int
) -> Iterator[dict[str, Any]]:
    for _ in range(epochs):
        for g in groups:
            padded = math.ceil(len(g) / backend.world_size) * backend.world_size
            for pos in range(backend.rank, padded, backend.world_size):
                batch = backend.make_batch(g[pos % len(g)])
                # Every rank must run the same collectives, even for an uneven final group.
                # Zero-weight padding contributes no gradient; scaling preserves the true mean.
                batch["loss_weight"] = padded / len(g) if pos < len(g) else 0.0
                yield batch


def _check_preconditions(
    cfg: TrainConfig, data_root: Path, allow_stale: bool, version: str
) -> Path:
    hp = holdout_path(data_root)
    if not hp.exists():
        raise TrainRefused(f"{hp} does not exist; run `tf manifest freeze-holdout` first")
    mp = Path(cfg.manifest)
    if not mp.exists():
        raise TrainRefused(f"manifest {mp} does not exist")
    versions = {r.report_version for r in manifest.read_manifest(mp)}
    if not versions:
        raise TrainRefused(f"manifest {mp} has no rows")
    if versions != {version} and not allow_stale:
        raise TrainRefused(
            f"manifest report_version {sorted(versions)} != current git hash {version}; "
            "rebuild the manifest (or --allow-stale-manifest, dev only)"
        )
    return mp


def train(
    cfg: TrainConfig,
    force_cfg: ForceConfig,
    data_root: Path,
    *,
    dry_run: bool,
    backend_factory: BackendFactory | None = None,
    run_name: str | None = None,
    allow_stale_manifest: bool = False,
    tokenizer: Tokenizer | None = None,
    windows_source: WindowsSource | None = None,
    eval_only_sources: Collection[str] | None = None,
    s_per_window: float = DEFAULT_S_PER_WINDOW,
    report_version: str | None = None,
) -> Path:
    """Run (or dry-run) one SFT run and return its run directory."""
    version = report_version or full_git_hash()
    mp = _check_preconditions(cfg, data_root, allow_stale_manifest, version)
    run_dir = Path(cfg.output_dir) / (run_name or f"sft-{cfg.force_mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    tok = tokenizer or QwenTokenizerAdapter(cfg.base_model)
    samples = build_samples(
        mp, cfg.force_mode, force_cfg, data_root, tokenizer=tok, windows_source=windows_source,
        seed=cfg.seed, eval_only_sources=eval_only_sources,
    )  # fmt: skip
    hist = token_histogram(samples)
    eta_s = estimate_wallclock(samples, s_per_window) * cfg.epochs
    if rank == 0:
        print(f"samples: {len(samples)}  force_mode: {cfg.force_mode}")
        print("tokens: " + "  ".join(f"{k}={v:.0f}" for k, v in hist.items()))
        print(f"estimated wall-clock: {eta_s / 3600:.2f} h at {s_per_window:g} s/window")
    if dry_run:
        if rank == 0:
            (run_dir / "dry_run.json").write_text(
                json.dumps(
                    {"n_samples": len(samples), "histogram": hist, "eta_s": eta_s}, indent=2
                ),
                encoding="utf-8",
            )
        return run_dir
    if not samples:
        raise TrainRefused("no training samples")

    if rank == 0:
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(), sort_keys=True), encoding="utf-8"
        )
        (run_dir / "manifest_hash.txt").write_text(
            manifest.manifest_hash(mp) + "\n", encoding="utf-8"
        )
        (run_dir / "git_hash.txt").write_text(version + "\n", encoding="utf-8")

    torch = _torch()
    torch.manual_seed(cfg.seed)
    backend = (backend_factory or HFBackend)(cfg)
    params = backend.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    groups = ordered_groups(samples, cfg.effective_batch, cfg.seed)
    accum = max(1, math.ceil(cfg.effective_batch / backend.world_size))
    total_steps = len(groups) * cfg.epochs
    scheduler = make_scheduler(optimizer, total_steps, cfg.warmup_frac)
    train_steps(
        backend.model,
        _batches(groups, backend, cfg.epochs),
        optimizer,
        scheduler,
        accum=accum,
        grad_clip=cfg.grad_clip,
        run_dir=run_dir if backend.rank == 0 else None,
        device=backend.device,
        keep_logits=isinstance(backend, HFBackend),
        params=params,
        autocast_bf16=isinstance(backend, HFBackend),
    )
    adapter_dir = run_dir / "adapter"
    if backend.rank == 0:
        shutil.rmtree(adapter_dir, ignore_errors=True)
    if backend.world_size > 1:
        torch.distributed.barrier()
    backend.save_adapter(adapter_dir)
    return run_dir


class HFBackend:
    """Qwen3.5-27B + LoRA on one or many GPUs. Imports torch/transformers/peft lazily.

    Unexercised on this machine (no [train] extra); launch multi-GPU under ``torchrun``.
    Steps: load bf16, cast Gated DeltaNet parameters (names containing ``linear_attn``) to fp32,
    freeze everything, attach LoRA to language-model attention/MLP projections only (a regex over
    ``cfg.lora.target_modules`` restricted to ``language_model``), so the vision encoder and
    merger stay frozen; enable gradient checkpointing; wrap with FSDP when world_size > 1,
    otherwise move to the single device. Mixed bf16/fp32 parameters need FSDP
    ``use_orig_params=True``. Video frames come from ``video_loader``; the default decodes
    ``sample.episode`` through video.mosaic.clip_frames with ``reader`` (the same path eval uses)
    and passes the source frame indices and fps to the processor as video metadata.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        video_loader: Callable[[Sample], Any] | None = None,
        reader: FrameReader | None = None,
    ) -> None:
        torch = _torch()
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        self.cfg = cfg
        self.reader: FrameReader = reader if reader is not None else SourceRootReader()
        self.video_loader = video_loader or self.load_clip
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group("nccl")
        self.rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
        self.world_size = (
            int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
        )
        self.device = torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu"
        self.processor = transformers.AutoProcessor.from_pretrained(cfg.base_model)
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            cfg.base_model, dtype=torch.bfloat16
        )
        model.config.use_cache = False
        for name, p in model.named_parameters():
            p.requires_grad = False
            if "linear_attn" in name:
                p.data = p.data.float()
        targets = "|".join(cfg.lora.target_modules)
        lcfg = peft.LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            target_modules=rf".*language_model.*\.({targets})",
        )
        model = peft.get_peft_model(model, lcfg)
        # Capture canonical PEFT export keys before nested FSDP wrappers change the module tree.
        self._adapter_saved_keys = set(peft.get_peft_model_state_dict(model))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        if self.world_size > 1:
            fsdp = importlib.import_module("torch.distributed.fsdp")
            peft_policy = peft.utils.other.fsdp_auto_wrap_policy(model)

            def wrap_policy(module: Any, recurse: bool, nonwrapped_numel: int) -> bool:
                # PEFT separates trainable fp32 LoRA leaves from frozen bf16 blocks.
                # DeltaNet also needs its own fp32 FSDP unit inside each decoder block.
                return type(module).__name__ == "Qwen3_5GatedDeltaNet" or peft_policy(
                    module, recurse, nonwrapped_numel
                )

            model = fsdp.FullyShardedDataParallel(
                model,
                use_orig_params=True,
                device_id=self.device,
                auto_wrap_policy=wrap_policy,
                sync_module_states=True,
                limit_all_gathers=True,
            )
        else:
            model = model.to(self.device)
        self.model = model

    def trainable_parameters(self) -> list[Any]:
        """LoRA parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def load_clip(self, sample: Sample) -> Any:
        """Default video_loader: the (T, H, W, 3) uint8 mosaic clip of ``sample``."""
        if sample.episode is None:
            raise ValueError(f"{sample.window_id}: sample has no episode to read video from")
        v = sample.video
        frames = clip_frames(
            sample.episode, self.reader, v.frame_times, v.roles, blank=v.blank, cell_px=v.cell_px
        )
        return np.stack(frames)

    def make_batch(self, sample: Sample) -> dict[str, Any]:
        """Run the processor on the sample's chat and decoded clip; mask all but the answer."""
        msgs = list(sample.messages)
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = prompt + sample.target + "<|im_end|>\n"
        video = self.video_loader(sample)
        kw: dict[str, Any] = {"videos": [video], "return_tensors": "pt"}
        if sample.episode is not None:
            meta = video_metadata(
                sample.video.frame_indices, sample.video.fps, source_frame_count(sample.episode)
            )
            kw["videos_kwargs"] = {"video_metadata": [meta], "do_sample_frames": False}
        enc = self.processor(text=[full], **kw)
        prefix = self.processor(text=[prompt], **kw)["input_ids"]
        n_prompt = int(prefix.size(1))
        if enc["input_ids"].size(1) <= n_prompt or not _torch().equal(
            enc["input_ids"][:, :n_prompt], prefix
        ):
            raise TrainRefused("multimodal prompt+answer tokens must extend the prompt exactly")
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = IGNORE_INDEX
        enc["labels"] = labels
        return dict(enc)

    def save_adapter(self, path: Path) -> None:
        """Gather full adapter tensors collectively, then serialize on rank zero."""
        if self.world_size > 1:
            fsdp = importlib.import_module("torch.distributed.fsdp")
            state: dict[str, Any] = {}
            # LoRA leaves have their own FSDP units. Gather only those tensors: gathering
            # the entire frozen 27B base would consume tens of GB of unnecessary host RAM.
            for name, unit in self.model.named_modules():
                if not isinstance(unit, fsdp.FullyShardedDataParallel):
                    continue
                direct = dict(unit.module.named_parameters(recurse=False))
                if not any(p.requires_grad for p in direct.values()):
                    continue
                with fsdp.FullyShardedDataParallel.summon_full_params(
                    unit, recurse=False, writeback=False, rank0_only=True, offload_to_cpu=True
                ):
                    if self.rank == 0:
                        prefix = name.replace("_fsdp_wrapped_module.", "")
                        for key, parameter in unit.module.named_parameters(recurse=False):
                            export_key = f"{prefix}.{key}".replace(".default.", ".")
                            if export_key in self._adapter_saved_keys:
                                state[export_key] = parameter.detach().cpu().contiguous().clone()
            if self.rank == 0:
                if not state or set(state) != self._adapter_saved_keys:
                    raise RuntimeError("FSDP export did not gather every expected adapter tensor")
                from safetensors.torch import save_file

                path.mkdir(parents=True, exist_ok=True)
                config = copy.deepcopy(self.model.module.peft_config["default"])
                config.inference_mode = True
                config.save_pretrained(path)
                save_file(state, path / "adapter_model.safetensors", metadata={"format": "pt"})
        elif self.rank == 0:
            self.model.save_pretrained(path)
