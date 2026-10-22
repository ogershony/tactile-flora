"""Compare generic and explicit grasp instructions without changing pilot labels or splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tactile_flora.config import EvalConfig, ForceConfig, SourceConfig, load_config
from tactile_flora.eval.harness import EpisodeBatch, make_provider, run_eval
from tactile_flora.eval.inference import HFBackend, StubBackend

BASE = Path("data/acceptance/hrdexdb-pilot")
CRITERION = (
    " Establish and maintain a stable grasp while lifting the object clear of the table."
    " Merely touching, tipping, dragging, or briefly lifting the object before losing the"
    " grasp does not complete the task. Distinguish an accidental slip or drop from a"
    " deliberate placement after a stable lift."
)


def main() -> None:
    """Run both fixed prompts through the same released backend and evaluation harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--output", type=Path, default=Path("data/acceptance/hrdexdb-resume/criterion")
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["hf", "stub"], default="hf")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").unlink(missing_ok=True)
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    source = load_config(args.base / "config/sources/hrdexdb.yaml", SourceConfig)
    manifest = args.base / "manifest/val.parquet"
    provider = make_provider(
        manifest,
        {"outcome"},
        force_cfg,
        args.base / "config/sources",
        {"hrdexdb": source},
        args.base,
    )
    batches = list(provider())
    plan = {
        "scope": "validation development diagnostic; no training or relabeling",
        "checkpoint": "released",
        "backend": args.backend,
        "model_id": args.model_id,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "conditions": {"generic": "original task", "stability": CRITERION},
        "force_mode": "omitted",
        "selection": "two fixed prompts; report both; no prompt sweep",
        "success_threshold": 4.0,
        "episodes": [b.episode.episode_id for b in batches],
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    backend = HFBackend(model_id=args.model_id) if args.backend == "hf" else StubBackend()
    reports = {}
    for condition, suffix in [("generic", ""), ("stability", CRITERION)]:
        benchmark = f"hrdexdb_criterion_{condition}"
        cfg = EvalConfig.model_validate(
            {
                "output_dir": str(args.output / "eval"),
                "benchmarks": {
                    benchmark: {
                        "manifest": str(manifest),
                        "prompts": ["outcome"],
                        "metrics": ["mae", "accuracy", "auroc"],
                    }
                },
            }
        )
        modified = [
            EpisodeBatch(
                b.episode.model_copy(update={"task": b.episode.task + suffix}),
                b.windows,
                b.window_ids,
                b.shift_s,
            )
            for b in batches
        ]
        output = run_eval(
            "released",
            benchmark,
            "omitted",
            cfg,
            force_cfg,
            args.output,
            backend,
            seed=0,
            provider=lambda batches=modified: iter(batches),
        )
        reports[condition] = json.loads((output / "metrics.json").read_text())
        print(condition, json.dumps(reports[condition]), flush=True)
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "plan": plan,
                "conditions": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
