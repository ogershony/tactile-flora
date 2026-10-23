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


def training_plan_path(base: Path, force_run: str, explicit: Path | None = None) -> Path:
    """Resolve an arm's frozen plan, retaining the legacy shared-plan layout."""
    if explicit is not None:
        return explicit
    arm_plan = base / f"training-plan-{force_run}.json"
    return arm_plan if arm_plan.is_file() else base / "training-plan.json"


def main() -> None:
    """Score every planned condition without tuning or choosing a best validation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--benchmark", default="hrdexdb_clean_outcome")
    parser.add_argument("--control-run", default="clean-v1-control")
    parser.add_argument("--force-run", default="clean-v1-force")
    parser.add_argument(
        "--plan", type=Path, help="Frozen training plan (defaults to the arm's plan)."
    )
    parser.add_argument(
        "--force-config",
        type=Path,
        default=Path("data/acceptance/released-inference/force.real-tokenizer.yaml"),
    )
    parser.add_argument(
        "--force-modes",
        default="real,shuffled",
        help="Comma-separated force modes evaluated on the force adapter.",
    )
    parser.add_argument("--worker", type=int, choices=range(3), help=argparse.SUPPRESS)
    parser.add_argument(
        "--reuse-released", action="store_true", help="Reuse the completed released-model cells."
    )
    args = parser.parse_args()
    eval_path = args.eval_config or args.base / "config/eval.yaml"
    cfg = load_config(eval_path, EvalConfig)
    force_cfg = load_config(args.force_config, ForceConfig)
    plan_path = training_plan_path(args.base, args.force_run, args.plan)
    training = json.loads(plan_path.read_text())
    model_id = training["base_model"]
    force_modes = [m for m in args.force_modes.split(",") if m]
    conditions = [
        ("released", ["omitted", "real"]),
        (str(args.base / "runs" / args.control_run / "adapter"), ["omitted"]),
        (str(args.base / "runs" / args.force_run / "adapter"), force_modes),
    ]
    selected = conditions if args.worker is None else [conditions[args.worker]]
    for checkpoint, _ in selected:
        if checkpoint == "released":
            continue
        if not (Path(checkpoint) / "adapter_model.safetensors").is_file():
            raise ValueError(f"missing trained adapter: {checkpoint}")
    plan = {
        "scope": (
            "post hoc curated development validation; 13 clips; not a new test result"
            if args.benchmark == "hrdexdb_clean_outcome"
            else f"fixed final checkpoints; {args.benchmark}; see frozen dataset split provenance"
        ),
        "conditions": conditions,
        "val_manifest_hash": manifest.manifest_hash(args.base / "manifest/val.parquet"),
        "evaluation_manifest_hash": manifest.manifest_hash(cfg.benchmarks[args.benchmark].manifest),
        "train_manifest_hash": manifest.manifest_hash(args.base / "manifest/hrdexdb-sft.parquet"),
        "training_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "training_plan_path": str(plan_path),
        "selection": (
            f"final epoch for both adapters; fixed {2 + 1 + len(force_modes)} evaluation "
            "conditions; no tuning"
        ),
        "force_config": str(args.force_config),
        "seed": 0,
    }
    stem = (
        "evaluation"
        if args.benchmark == "hrdexdb_clean_outcome"
        else f"evaluation-{args.benchmark}"
    )
    if plan_path.name != "training-plan.json":
        # Different arms share a base directory; their provenance must not overwrite each other.
        stem = f"{stem}-{args.force_run}"
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
                "--plan",
                str(plan_path),
                "--force-config",
                str(args.force_config),
                "--force-modes",
                args.force_modes,
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
                result["manifest_hash"] != plan["evaluation_manifest_hash"]
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
