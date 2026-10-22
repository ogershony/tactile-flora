"""Evaluate the cleaned pilot's released, force and matched control checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tactile_flora.config import EvalConfig, ForceConfig, load_config
from tactile_flora.eval.harness import checkpoint_tag, run_eval
from tactile_flora.eval.inference import HFBackend
from tactile_flora.verify import manifest
from tactile_flora.video.readers import SourceRootReader

BASE = Path("data/acceptance/hrdexdb-clean-v1")


def main() -> None:
    """Score every planned condition without tuning or choosing a best validation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--benchmark", default="hrdexdb_clean_outcome")
    parser.add_argument("--control-run", default="clean-v1-control")
    parser.add_argument("--force-run", default="clean-v1-force")
    parser.add_argument("--worker", type=int, choices=range(3), help=argparse.SUPPRESS)
    parser.add_argument(
        "--reuse-released", action="store_true", help="Reuse the completed released-model cells."
    )
    args = parser.parse_args()
    eval_path = args.eval_config or args.base / "config/eval.yaml"
    cfg = load_config(eval_path, EvalConfig)
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    training = json.loads((args.base / "training-plan.json").read_text())
    model_id = training["base_model"]
    conditions = [
        ("released", ["omitted", "real"]),
        (str(args.base / "runs" / args.control_run / "adapter"), ["omitted"]),
        (str(args.base / "runs" / args.force_run / "adapter"), ["real", "shuffled"]),
    ]
    for checkpoint, _ in conditions[1:]:
        if not (Path(checkpoint) / "adapter_model.safetensors").is_file():
            raise ValueError(f"missing trained adapter: {checkpoint}")
    plan = {
        "scope": (
            "post hoc curated development validation; 13 clips; not a new test result"
            if args.benchmark == "hrdexdb_clean_outcome"
            else f"fixed final checkpoints; {args.benchmark}; see frozen dataset split provenance"
        ),
        "conditions": conditions,
        "val_manifest_hash": manifest.manifest_hash(cfg.benchmarks[args.benchmark].manifest),
        "train_manifest_hash": manifest.manifest_hash(args.base / "manifest/hrdexdb-sft.parquet"),
        "training_plan_sha256": hashlib.sha256(
            (args.base / "training-plan.json").read_bytes()
        ).hexdigest(),
        "selection": "final epoch for both adapters; fixed five evaluation conditions; no tuning",
        "seed": 0,
    }
    stem = (
        "evaluation"
        if args.benchmark == "hrdexdb_clean_outcome"
        else f"evaluation-{args.benchmark}"
    )
    (args.base / f"{stem}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.worker is not None:
        checkpoint, modes = conditions[args.worker]
        backend = HFBackend(
            checkpoint, reader=SourceRootReader(args.base / "config/sources"), model_id=model_id
        )
        for mode in modes:
            print(f"evaluating {checkpoint} {mode}", flush=True)
            output = run_eval(
                checkpoint,
                args.benchmark,
                mode,
                cfg,
                force_cfg,
                args.base,
                backend,
                seed=0,
                sources_dir=args.base / "config/sources",
            )
            print((output / "metrics.json").read_text(), flush=True)
        return
    # A fresh interpreter per checkpoint releases all model and accelerator state.
    for index in range(len(conditions)):
        if index == 0 and args.reuse_released:
            continue
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--base",
                str(args.base),
                "--worker",
                str(index),
                "--eval-config",
                str(eval_path),
                "--benchmark",
                args.benchmark,
                "--control-run",
                args.control_run,
                "--force-run",
                args.force_run,
            ],
            check=True,
        )
    results = []
    for checkpoint, modes in conditions:
        for mode in modes:
            path = args.base / "eval" / checkpoint_tag(checkpoint) / args.benchmark / mode
            result = json.loads((path / "metrics.json").read_text())
            saved_config = json.loads((path / "config.json").read_text())
            if (
                result["manifest_hash"] != plan["val_manifest_hash"]
                or result["checkpoint"] != checkpoint
                or result["force_mode"] != mode
                or result["seed"] != plan["seed"]
                or saved_config["eval"] != cfg.model_dump(mode="json")
                or saved_config["force"] != force_cfg.model_dump(mode="json")
            ):
                raise ValueError(f"evaluation provenance mismatch: {path}")
            results.append(result)
    (args.base / f"{stem}-result.json").write_text(
        json.dumps({"plan": plan, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
