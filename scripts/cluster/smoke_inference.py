"""Prepare real PHANTOM inputs or run the released FLORA dense/outcome acceptance gate.

Preparation downloads only processor assets at a resolved checkpoint revision. It records
real tokenizer counts and decodes both video clips without loading model weights. Full mode
requires enough free GPU memory for the checkpoint plus inference headroom, then records raw
answers and parser verdicts. Neither mode creates an accepted data manifest.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.eval.inference import (
    DENSE_MAX_NEW_TOKENS,
    MODEL_ID,
    RATIONALE_MAX_NEW_TOKENS,
    HFBackend,
    clip_array,
)
from tactile_flora.force import normalize, render
from tactile_flora.labels.windows import dense_frame_times, outcome_frame_indices
from tactile_flora.prompts.flora import build_prompt, to_messages
from tactile_flora.prompts.parse import (
    dense_regex,
    outcome_regex,
    parse_dense,
    parse_outcome,
)
from tactile_flora.schema import Episode, Window
from tactile_flora.sources.phantom import PhantomAdapter
from tactile_flora.verify.windowing import resample_shifted
from tactile_flora.video.mosaic import frame_indices, source_frame_count, video_metadata
from tactile_flora.video.readers import AutoReader

PROCESSOR_FILES = ["*.json", "*.txt", "*.jinja", "*.model"]


def make_parser() -> argparse.ArgumentParser:
    """Build a CLI usable without importing the training dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=Path("configs/sources/phantom.yaml"))
    parser.add_argument("--force-config", type=Path, default=Path("configs/force.yaml"))
    parser.add_argument(
        "--norm", type=Path, default=Path("data/acceptance/phantom/verified/phantom/norm.json")
    )
    parser.add_argument("--episode", help="Episode id (defaults to the first downloaded episode)")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/acceptance/released-inference"))
    return parser


def windows_for(episode: Episode) -> list[Window]:
    """Build unlabeled input windows; acceptance must not depend on training labels."""
    cam = episode.cameras["scene"]
    common: dict[str, Any] = dict(
        episode_ref=f"{episode.source}/{episode.episode_id}",
        anchor_s=0.0,
        force_lines=(),
        target="",
        split="train",
    )
    return [
        Window(prompt_kind="dense", frame_times=tuple(dense_frame_times(0.0)), **common),
        Window(
            prompt_kind="outcome",
            frame_times=tuple(i / cam.fps for i in outcome_frame_indices(cam.n_frames, cam.fps)),
            **common,
        ),
    ]


def run(args: argparse.Namespace) -> int:
    """Write one audit record; return nonzero unless the requested stage succeeds."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(output / "xet"))
    result: dict[str, Any] = {
        "status": "running",
        "gate_passed": False,
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "host": platform.node(),
        "cases": [],
        "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
    }
    path = output / "result.json"
    # Clear a prior success before any download, decode or model load can fail.
    path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoProcessor

        result["versions"] = {
            name: importlib.metadata.version(name)
            if importlib.util.find_spec(module) is not None
            else "unavailable"
            for name, module in (
                ("torch", "torch"),
                ("torchvision", "torchvision"),
                ("transformers", "transformers"),
                ("flash-linear-attention", "fla"),
                ("triton", "triton"),
            )
        }
        result.update(cuda=torch.version.cuda, hip=torch.version.hip)
        devices = [
            {"name": torch.cuda.get_device_name(i), "free_bytes": torch.cuda.mem_get_info(i)[0]}
            for i in range(torch.cuda.device_count())
        ]
        result["gpus"] = devices
        info = HfApi().model_info(MODEL_ID, revision=args.revision, files_metadata=True)
        result["revision"] = info.sha
        weight_bytes = sum(
            f.size or 0 for f in info.siblings or [] if f.rfilename.endswith(".safetensors")
        )
        if weight_bytes <= 0:
            raise RuntimeError("checkpoint metadata does not declare safetensors weight sizes")
        result["weight_bytes"] = weight_bytes
        if not args.prepare_only and sum(g["free_bytes"] for g in devices) < weight_bytes * 1.2:
            result["status"] = "blocked"
            result["reason"] = "Insufficient free GPU memory for released weights + 20% headroom"
            return 2
        snapshot = Path(
            snapshot_download(
                MODEL_ID,
                revision=info.sha,
                cache_dir=output / "hub",
                max_workers=1,
                allow_patterns=PROCESSOR_FILES
                if args.prepare_only
                else [*PROCESSOR_FILES, "*.safetensors"],
            )
        )
        result["snapshot"] = str(snapshot)
        result["tokenizer_json"] = str(snapshot / "tokenizer.json")
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        source_cfg = load_config(args.source_config, SourceConfig)
        force_cfg = load_config(args.force_config, ForceConfig)
        force_cfg.render.tokenizer_json = str(snapshot / "tokenizer.json")
        tokenizer_config = output / "force.real-tokenizer.yaml"
        tokenizer_config.write_text(yaml.safe_dump(force_cfg.model_dump(mode="json")))
        result["force_config"] = str(tokenizer_config)
        adapter = PhantomAdapter(source_cfg)
        ids = adapter.discover(Path(source_cfg.root))
        if not ids:
            raise RuntimeError(f"no downloaded episodes under {source_cfg.root}")
        episode = adapter.load(args.episode or ids[0])
        result["episode"] = f"{episode.source}/{episode.episode_id}"
        consts = normalize.load(args.norm)
        result["norm_version"] = consts.version()
        reader = AutoReader()
        backend = HFBackend(model_id=str(snapshot), reader=reader)
        for window in windows_for(episode):
            feats = normalize.apply(
                resample_shifted(episode, window.frame_times, force_cfg.resample),
                consts,
                force_cfg.render,
            )
            block = render.render_block(
                render.render_lines(feats, window.frame_times, feats.channel), force_cfg.render
            )
            # Record omitted and real-force counts so the matched control is inspectable.
            for mode, force_block in (("omitted", None), ("real", block)):
                bundle = build_prompt(
                    window,
                    force_block,
                    task=episode.task,
                    camera_roles=list(episode.cameras),
                    fps=episode.cameras["scene"].fps,
                    frame_indices=frame_indices(episode, "scene", window.frame_times),
                )
                messages = to_messages(bundle)
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                frames = clip_array(episode, bundle.video, reader)
                inputs = processor(
                    text=[prompt],
                    videos=[frames],
                    return_tensors="pt",
                    videos_kwargs={
                        "video_metadata": [
                            video_metadata(
                                bundle.video.frame_indices,
                                bundle.video.fps,
                                source_frame_count(episode),
                            )
                        ],
                        "do_sample_frames": False,
                    },
                )
                count = len(processor.tokenizer.encode(block, add_special_tokens=False))
                if mode == "real" and count >= force_cfg.render.max_tokens:
                    raise RuntimeError(f"force block exceeds configured budget: {count}")
                case: dict[str, Any] = {
                    "prompt_kind": window.prompt_kind,
                    "force_mode": mode,
                    "input_tokens": inputs["input_ids"].shape[1],
                    "force_tokens": count if mode == "real" else 0,
                    "video_shape": list(frames.shape),
                    "video_grid_thw": inputs["video_grid_thw"].tolist(),
                }
                result["cases"].append(case)
                if not args.prepare_only:
                    dense = window.prompt_kind == "dense"
                    generated = backend.generate(
                        messages,
                        bundle.video,
                        episode=episode,
                        regex=dense_regex() if dense else outcome_regex(True),
                        max_new_tokens=DENSE_MAX_NEW_TOKENS if dense else RATIONALE_MAX_NEW_TOKENS,
                        top_logprobs=0,
                    )
                    parsed = parse_dense(generated.text) if dense else parse_outcome(generated.text)
                    case.update(generation=generated.model_dump(), parsed=parsed.model_dump())
                    if parsed.parse_failed:
                        raise RuntimeError(f"{window.prompt_kind}/{mode} output failed parsing")
        result["status"] = "prepared" if args.prepare_only else "passed"
        result["gate_passed"] = not args.prepare_only
        return 0
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    raise SystemExit(run(make_parser().parse_args()))
