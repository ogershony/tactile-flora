"""Validate completed expanded runs and export a portable comparison without model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

from tactile_flora.eval.harness import checkpoint_tag
from tactile_flora.eval.metrics import SUCCESS_THRESHOLD
from tactile_flora.verify import manifest

BASE = Path("data/acceptance/hrdexdb-expanded-v2")


def main(argv: list[str] | None = None) -> None:
    """Check artifacts, identities and truth targets before summarizing fixed conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--force-run", default="expanded-v2-force")
    parser.add_argument("--control-run", default="expanded-v2-control")
    args = parser.parse_args(argv)
    base = args.base if args.base is not None else BASE
    plan = json.loads((base / "training-plan.json").read_text())
    result = {
        "scope": "Object-disjoint source-label evaluation; no independent human gold labels",
        "training_plan": plan,
        "training_artifacts": [],
        "evaluation": {},
    }
    for name in [args.force_run, args.control_run]:
        directory = base / "runs" / name
        with (directory / "loss.csv").open() as handle:
            losses = list(csv.DictReader(handle))
        if len(losses) != plan["optimizer_steps_per_run"]:
            raise ValueError(f"unexpected optimizer update count: {name}")
        if (directory / "manifest_hash.txt").read_text().strip() != plan["manifest_sha256"]:
            raise ValueError(f"training manifest mismatch: {name}")
        weights = directory / "adapter/adapter_model.safetensors"
        tensors = 0
        with safe_open(weights, framework="np") as adapter:
            keys = adapter.keys()
            for key in keys:
                if not np.isfinite(adapter.get_tensor(key)).all():
                    raise ValueError(f"nonfinite trained adapter: {name} {key}")
                tensors += 1
        artifact = {
            "run": name,
            "optimizer_updates": len(losses),
            "tensor_count": tensors,
            "all_tensors_finite": True,
            "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "first_loss": float(losses[0]["loss"]),
            "last_loss": float(losses[-1]["loss"]),
        }
        result["training_artifacts"].append(artifact)
    lines = [
        "# Expanded HRDexDB results",
        "",
        result["scope"],
        "",
        "Final adapters from fixed four-epoch runs; no validation-selected checkpoint.",
        "",
    ]
    for split, hash_key in [
        ("val", "validation_manifest_sha256"),
        ("test", "test_manifest_sha256"),
    ]:
        expected_hash = manifest.manifest_hash(base / f"manifest/{split}.parquet")
        if expected_hash != plan[hash_key]:
            raise ValueError(f"frozen {split} manifest changed")
        expected_ids = {
            r.episode_id for r in manifest.read_manifest(base / f"manifest/{split}.parquet")
        }
        evaluation = json.loads(
            (base / f"evaluation-hrdexdb_aligned_{split}-result.json").read_text()
        )
        expected_cells = {
            (checkpoint, mode)
            for checkpoint, modes in plan["evaluation_conditions"].items()
            for mode in modes
        }
        actual_cells = [
            (checkpoint_tag(m["checkpoint"]), m["force_mode"]) for m in evaluation["results"]
        ]
        if set(actual_cells) != expected_cells or len(actual_cells) != len(expected_cells):
            raise ValueError(f"missing or duplicate planned evaluation conditions: {split}")
        cells, identity = [], None
        lines.extend(
            [
                f"## {split} ({len(expected_ids)} episodes)",
                "",
                "| Checkpoint | Force | Accuracy | AUROC | Failure recall | Parse failures |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for metrics in evaluation["results"]:
            tag, mode = checkpoint_tag(metrics["checkpoint"]), metrics["force_mode"]
            path = base / "eval" / tag / f"hrdexdb_aligned_{split}" / mode
            pred = pd.read_parquet(path / "predictions.parquet").sort_values("episode_id")
            if (
                set(pred.episode_id) != expected_ids
                or len(pred) != len(expected_ids)
                or metrics["manifest_hash"] != expected_hash
            ):
                raise ValueError(f"evaluation episode/manifest mismatch: {path}")
            keys = pred[["episode_id", "window_id", "truth_target"]].to_dict("records")
            if identity is not None and keys != identity:
                raise ValueError(f"different truth targets across evaluation conditions: {path}")
            identity = keys
            truth = pred.truth_target.str.extract(r"Score:\s*([1-5])\s*$")[0].astype(int)
            success = truth >= SUCCESS_THRESHOLD
            guessed = pred.expected_score >= SUCCESS_THRESHOLD
            failure_recall = float((~guessed[~success]).mean())
            cell = {
                "metrics": metrics,
                "failure_recall": failure_recall,
                "predictions": json.loads(pred.to_json(orient="records")),
                "config": json.loads((path / "config.json").read_text()),
            }
            cells.append(cell)
            score = metrics["metrics"]
            lines.append(
                f"| {tag} | {mode} | {score['accuracy']:.1%} | {score['auroc']:.3f} | "
                f"{failure_recall:.1%} | {metrics['parse_failed']} |"
            )
        result["evaluation"][split] = {"plan": evaluation["plan"], "cells": cells}
        lines.append("")
    lines.extend(
        [
            "Source outcomes have not all been independently relabeled. Recorded-clock",
            "alignment is verified against the released files, not independent hardware latency.",
            "Retained clips passed mechanical checks; this does not certify every outcome label.",
            "",
        ]
    )
    (base / "final-result.json").write_text(json.dumps(result, indent=2) + "\n")
    (base / "final-result.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
