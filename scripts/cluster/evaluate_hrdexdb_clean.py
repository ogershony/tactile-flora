"""Evaluate the cleaned pilot's released, force and matched control checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from tactile_flora.config import EvalConfig, ForceConfig, load_config
from tactile_flora.eval.harness import run_eval
from tactile_flora.eval.inference import HFBackend
from tactile_flora.verify import manifest
from tactile_flora.video.readers import SourceRootReader

BASE = Path("data/acceptance/hrdexdb-clean-v1")


def main() -> None:
    """Score every planned condition without tuning or choosing a best validation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    cfg = load_config(args.base / "config/eval.yaml", EvalConfig)
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    training = json.loads((args.base / "training-plan.json").read_text())
    model_id = training["base_model"]
    conditions = [
        ("released", ["omitted", "real"]),
        (str(args.base / "runs/clean-v1-control/adapter"), ["omitted"]),
        (str(args.base / "runs/clean-v1-force/adapter"), ["real", "shuffled"]),
    ]
    for checkpoint, _ in conditions[1:]:
        if not (Path(checkpoint) / "adapter_model.safetensors").is_file():
            raise ValueError(f"missing trained adapter: {checkpoint}")
    plan = {
        "scope": "post hoc curated development validation; 13 clips; not a new test result",
        "conditions": conditions,
        "val_manifest_hash": manifest.manifest_hash(args.base / "manifest/val.parquet"),
        "train_manifest_hash": manifest.manifest_hash(args.base / "manifest/hrdexdb-sft.parquet"),
        "training_plan_sha256": hashlib.sha256(
            (args.base / "training-plan.json").read_bytes()
        ).hexdigest(),
        "selection": "final epoch for both adapters; fixed five evaluation conditions; no tuning",
        "seed": 0,
    }
    (args.base / "evaluation-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for checkpoint, modes in conditions:
        backend = HFBackend(
            checkpoint, reader=SourceRootReader(args.base / "config/sources"), model_id=model_id
        )
        for mode in modes:
            print(f"evaluating {checkpoint} {mode}", flush=True)
            output = run_eval(
                checkpoint,
                "hrdexdb_clean_outcome",
                mode,
                cfg,
                force_cfg,
                args.base,
                backend,
                seed=0,
                sources_dir=args.base / "config/sources",
            )
            result = json.loads((output / "metrics.json").read_text())
            results.append(result)
            print(json.dumps(result), flush=True)
        del backend
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    (args.base / "evaluation-result.json").write_text(
        json.dumps({"plan": plan, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
