"""Score released FLORA and two fine-tuned adapters on a prepared Baxter rollout benchmark.

Conditions (each checkpoint runs in a fresh interpreter so model memory is released):

1. released FLORA, force omitted and real (the untrained baseline, with and without the block);
2. the fine-tuned video-only control adapter, force omitted (as trained);
3. the fine-tuned tactile adapter with the force modes requested (``real`` is the headline;
   ``shuffled`` and ``omitted`` ask whether the block matters on this data).

Every cell lands under ``<base>/eval/<tag>/<benchmark>/<mode>/`` through the shared harness,
and the collected metrics plus per-episode predictions are written to
``<base>/evaluation-result.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from tactile_flora.config import EvalConfig, ForceConfig, load_config
from tactile_flora.eval.harness import checkpoint_tag, run_eval
from tactile_flora.eval.inference import HFBackend, StubBackend
from tactile_flora.verify import manifest
from tactile_flora.video.readers import SourceRootReader

BASE = Path("data/acceptance/baxter-rollouts-v1")
V3 = Path("data/acceptance/hrdexdb-expanded-v3")
BENCHMARK = "baxter_rollouts"


def _conditions(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    if args.released_modes:
        out.append(("released", [m for m in args.released_modes.split(",") if m]))
    if args.control_adapter:
        out.append((str(args.control_adapter), ["omitted"]))
    if args.force_adapter:
        out.append((str(args.force_adapter), [m for m in args.force_modes.split(",") if m]))
    return out


def main() -> None:
    """Run every condition (or one worker) and collect the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--released-modes", default="omitted,real")
    parser.add_argument(
        "--control-adapter",
        type=Path,
        default=V3 / "runs/expanded-v2-control/adapter",
        help="Fine-tuned video-only adapter directory ('' to skip).",
    )
    parser.add_argument(
        "--force-adapter",
        type=Path,
        default=V3 / "runs/expanded-v3-force/adapter",
        help="Fine-tuned tactile adapter directory ('' to skip).",
    )
    parser.add_argument("--force-modes", default="real,shuffled,omitted")
    parser.add_argument("--model-id", default="", help="Base model path; default: v3 plan.")
    parser.add_argument("--backend", choices=["hf", "stub"], default="hf")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.control_adapter is not None and str(args.control_adapter) in ("", "."):
        args.control_adapter = None
    if args.force_adapter is not None and str(args.force_adapter) in ("", "."):
        args.force_adapter = None
    base: Path = args.base
    eval_path = base / "config/eval.yaml"
    cfg = load_config(eval_path, EvalConfig)
    force_cfg = load_config(base / "config/force.yaml", ForceConfig)
    model_id = args.model_id or json.loads((V3 / "training-plan.json").read_text())["base_model"]
    conditions = _conditions(args)
    for checkpoint, _ in conditions:
        if (
            checkpoint != "released"
            and not (Path(checkpoint) / "adapter_model.safetensors").is_file()
        ):
            raise ValueError(f"missing trained adapter: {checkpoint}")

    if args.worker is not None:
        checkpoint, modes = conditions[args.worker]
        reader = SourceRootReader(base / "config/sources")
        backend: Any = (
            StubBackend()
            if args.backend == "stub"
            else HFBackend(checkpoint, reader=reader, model_id=model_id)
        )
        for mode in modes:
            print(f"evaluating {checkpoint} {mode}", flush=True)
            out = run_eval(
                checkpoint,
                args.benchmark,
                mode,  # type: ignore[arg-type]
                cfg,
                force_cfg,
                base,
                backend,
                seed=args.seed,
                sources_dir=base / "config/sources",
            )
            print((out / "metrics.json").read_text(), flush=True)
        return

    for index in range(len(conditions)):
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--base",
                str(base),
                "--benchmark",
                args.benchmark,
                "--released-modes",
                args.released_modes,
                "--control-adapter",
                "" if args.control_adapter is None else str(args.control_adapter),
                "--force-adapter",
                "" if args.force_adapter is None else str(args.force_adapter),
                "--force-modes",
                args.force_modes,
                "--model-id",
                model_id,
                "--backend",
                args.backend,
                "--seed",
                str(args.seed),
                "--worker",
                str(index),
            ],
            check=True,
        )

    spec = cfg.benchmarks[args.benchmark]
    results: list[dict[str, Any]] = []
    for checkpoint, modes in conditions:
        for mode in modes:
            path = base / "eval" / checkpoint_tag(checkpoint) / args.benchmark / mode
            result = json.loads((path / "metrics.json").read_text())
            if result["manifest_hash"] != manifest.manifest_hash(spec.manifest):
                raise ValueError(f"evaluation provenance mismatch: {path}")
            preds = pd.read_parquet(path / "predictions.parquet")
            result["predictions"] = preds.to_dict("records")
            results.append(result)
    (base / "evaluation-result.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "manifest_hash": manifest.manifest_hash(spec.manifest),
                "model_id": model_id,
                "backend": args.backend,
                "conditions": conditions,
                "results": results,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    for r in results:
        print(checkpoint_tag(r["checkpoint"]), r["force_mode"], r["metrics"])


if __name__ == "__main__":
    main()
