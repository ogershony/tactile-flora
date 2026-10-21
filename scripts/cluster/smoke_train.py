"""Run answer-only LoRA SFT on a tiny, randomly initialized Qwen3.5 model.

This hardware smoke checks the training path and adapter serialization. It does not load released
weights or measure model quality. Training dependencies are imported only after CLI parsing so
``--help`` works in environments without the optional train extra.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("data/cluster/smoke")
DEFAULT_STEPS = 20


def minimum_two_steps(value: str) -> int:
    """Parse a step count large enough to compare initial and final loss."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least two")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing the optional GPU training stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=minimum_two_steps, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device (default: cuda; use cpu for a functional check without a GPU)",
    )
    return parser


def _artifact_record(adapter_dir: Path) -> dict[str, Any]:
    """Check the saved PEFT config and safetensors file, and return artifact metadata."""
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"adapter is missing expected files under {adapter_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or not config.get("target_modules"):
        raise RuntimeError("saved adapter config is not a LoRA adapter with target modules")

    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        tensor_keys = list(handle.keys())
    if not tensor_keys:
        raise RuntimeError("saved adapter weights contain no tensors")

    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    return {
        "valid": True,
        "checks": [
            "adapter_config_is_lora",
            "weights_are_readable_safetensors",
            "nonempty_weights",
        ],
        "files": {
            config_path.name: {"bytes": config_path.stat().st_size},
            weights_path.name: {
                "bytes": weights_path.stat().st_size,
                "sha256": digest,
                "tensor_count": len(tensor_keys),
            },
        },
    }


def _version(distribution: str) -> str:
    """Return an installed package version or an explicit unavailable marker."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run(*, steps: int, output: Path, device_name: str) -> dict[str, Any]:
    """Train the synthetic task and write losses, logs, adapter, and result metadata."""
    if steps < 2:
        raise ValueError("smoke training requires at least two steps to compare loss")

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from tactile_flora.train.sft import make_scheduler, train_steps

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seed = 0
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=128,
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
    model = get_peft_model(
        Qwen3_5ForCausalLM(config),
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[
                "q_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "in_proj_qkv",
                "out_proj",
            ],
        ),
    ).to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    ids = torch.randint(0, 128, (1, 32), device=device)
    labels = ids.clone()
    labels[:, :24] = -100
    batch = {"input_ids": ids, "labels": labels}
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.02)
    scheduler = make_scheduler(optimizer, steps, 0.03)

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a prior result marker beside partial new artifacts.
    (output / "result.json").unlink(missing_ok=True)
    losses = train_steps(
        model,
        [batch] * steps,
        optimizer,
        scheduler,
        accum=1,
        grad_clip=1.0,
        device=device,
        run_dir=output,
        keep_logits=True,
    )
    if len(losses) != steps or not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError(f"invalid smoke losses: {losses}")
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"smoke loss did not decrease: {losses[0]} -> {losses[-1]}")

    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    artifact = _artifact_record(adapter_dir)
    on_cuda = device.type == "cuda"
    result = {
        "kind": "random-small-qwen3.5-lora-smoke",
        "scope": "synthetic repeated tokens; randomly initialized base model; no released weights",
        "model": {
            "architecture": "Qwen3_5ForCausalLM",
            "initialization": "random",
            "vocab_size": 128,
            "hidden_size": 64,
            "layers": 4,
            "sequence_length": 32,
            "answer_tokens": 8,
            "lora_rank": 4,
        },
        "seed": seed,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "precision": "float32",
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dependencies": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": _version("peft"),
            "safetensors": _version("safetensors"),
        },
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(device) if on_cuda else None,
        "host": platform.node(),
        "max_memory_bytes": torch.cuda.max_memory_allocated(device) if on_cuda else None,
        "artifact": artifact,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    """Parse arguments and start the smoke run."""
    args = make_parser().parse_args()
    run(steps=args.steps, output=args.output, device_name=args.device)


if __name__ == "__main__":
    main()
