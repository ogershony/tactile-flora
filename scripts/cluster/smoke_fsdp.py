"""Exercise the production HF/FSDP backend on a tiny supplied checkpoint and random tokens.

Launch with torchrun on at least two GPUs. This checks language-model forward/backward,
mixed precision, sharding and adapter gathering/reload. ``--multimodal`` uses synthetic video
with the real processor and answer mask. Neither mode constructs an accepted training manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from scripts.cluster.smoke_train import _artifact_record, minimum_two_steps


def prepare_checkpoint(checkpoint: Path, processor_path: Path) -> None:
    """Create a deterministic tiny random multimodal model with the released tokenizer."""
    import torch
    from transformers import (
        AutoProcessor,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    text = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=36,
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    model.save_pretrained(checkpoint)
    AutoProcessor.from_pretrained(processor_path, local_files_only=True).save_pretrained(checkpoint)


def synthetic_video_batch(backend: Any) -> dict[str, Any]:
    """Process synthetic video through production chat, vision and answer masking."""
    import numpy as np

    from tactile_flora.prompts.flora import build_prompt, to_messages
    from tactile_flora.schema import Episode, ForceStream, VideoRef, Window
    from tactile_flora.train.dataset import Sample

    episode = Episode(
        source="synthetic",
        episode_id="fsdp-video",
        session="smoke",
        task="Synthetic video test",
        duration_s=1,
        cameras={"scene": VideoRef(path="synthetic-video", fps=4, n_frames=4)},
        force=ForceStream(channel="pad", arms={}, rate_hz=1, t=[]),
        license="synthetic",
    )
    window = Window(
        episode_ref="synthetic/fsdp-video",
        prompt_kind="outcome",
        anchor_s=0,
        frame_times=(0, 0.25, 0.5, 0.75),
        force_lines=(),
        target="Score: 5",
        split="train",
    )
    bundle = build_prompt(window, None, with_rationale=False, task=episode.task)
    sample = Sample(
        window_id="synthetic-video",
        messages=to_messages(bundle),
        video=bundle.video,
        episode=episode,
        force_block=None,
        target=window.target,
        input_ids=[],
        labels=[],
        n_tokens=0,
    )
    backend.video_loader = lambda _: np.zeros((4, 96, 96, 3), dtype=np.uint8)
    return backend.make_batch(sample)


def run(checkpoint: Path, output: Path, steps: int, *, multimodal: bool = False) -> None:
    """Train repeated random tokens and compare a reloaded adapter with the live model."""
    rank = int(os.environ.get("RANK", "0"))
    kind = (
        "checkpoint-backed-fsdp-multimodal-smoke"
        if multimodal
        else "checkpoint-backed-fsdp-language-smoke"
    )
    scope = (
        "synthetic video; real processor/vision/answer-mask path; no accepted data; "
        "not production training"
        if multimodal
        else "random repeated tokens; no accepted data; not production training"
    )
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "running",
        "kind": kind,
        "scope": scope,
        "checkpoint": str(checkpoint.resolve()),
        "host": platform.node(),
        "multimodal": multimodal,
    }
    path = output / "result.json"
    if rank == 0:
        path.write_text(json.dumps(result, indent=2) + "\n")
    torch: Any = None
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        from tactile_flora.config import TrainConfig, load_config
        from tactile_flora.train.sft import HFBackend, make_scheduler, train_steps

        torch.manual_seed(0)
        cfg = load_config("configs/train/force.yaml", TrainConfig)
        cfg.base_model = str(checkpoint.resolve())
        cfg.lora.rank, cfg.lora.alpha = 4, 8
        backend = HFBackend(cfg)
        if backend.world_size < 2:
            raise RuntimeError("launch under torchrun with at least two GPUs")
        result.update(world_size=backend.world_size, torch=torch.__version__, hip=torch.version.hip)
        if multimodal:
            batch = synthetic_video_batch(backend)
        else:
            ids = torch.randint(1, 128, (1, 32), device=backend.device)
            labels = ids.clone()
            labels[:, :24] = -100
            batch = {"input_ids": ids, "labels": labels}
        inputs = {k: v.to(backend.device) for k, v in batch.items() if k != "labels"}
        params = backend.trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=0.02)
        scheduler = make_scheduler(optimizer, steps, 0.03)
        losses = train_steps(
            backend.model,
            [batch] * steps,
            optimizer,
            scheduler,
            accum=1,
            grad_clip=1.0,
            device=backend.device,
            keep_logits=True,
            params=params,
            run_dir=output if rank == 0 else None,
            autocast_bf16=True,
            log_every=1,
        )
        if not all(torch.isfinite(torch.tensor(losses))) or losses[-1] >= losses[0]:
            raise RuntimeError(f"smoke loss failed to decrease: {losses}")
        backend.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = backend.model(**inputs, logits_to_keep=1).logits.detach().float()
        backend.save_adapter(output / "adapter")
        torch.distributed.barrier()
        if rank == 0:
            base = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16)
            for name, parameter in base.named_parameters():
                if "linear_attn" in name:
                    parameter.data = parameter.data.float()
            base.config.use_cache = False
            reloaded = PeftModel.from_pretrained(base, output / "adapter").to(backend.device).eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                actual = reloaded(**inputs, logits_to_keep=1).logits.float()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
            result.update(
                status="passed",
                steps=steps,
                losses=losses,
                initial_loss=losses[0],
                final_loss=losses[-1],
                reload_max_abs_error=float((actual - expected).abs().max()),
                artifact=_artifact_record(output / "adapter"),
            )
        torch.distributed.barrier()
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if rank == 0:
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/acceptance/fsdp-smoke/run"))
    parser.add_argument("--steps", type=minimum_two_steps, default=8)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--processor", type=Path)
    args = parser.parse_args()
    if args.prepare_only:
        if args.processor is None:
            parser.error("--prepare-only requires --processor pointing at local processor assets")
        prepare_checkpoint(args.checkpoint, args.processor)
    else:
        run(args.checkpoint, args.output, args.steps, multimodal=args.multimodal)
