"""Fit the stacked head on the dev split and score the test split; write stack-result.{json,md}.

The stack is a logistic regression over three numbers per clip: the tactile VLM's expected
score, the video-only VLM's expected score and the force-only classifier's logit. It is fit on
the dev cells only; the dev number reported is object-grouped out-of-fold, the test number is
the dev-fit model applied once. Example (v3 tactile arm against the v2 control):

    .venv/bin/python scripts/cluster/stack_hrdexdb.py \\
        --base data/acceptance/hrdexdb-expanded-v3 \\
        --tactile-cells data/acceptance/hrdexdb-expanded-v3/eval/expanded-v3-force \\
        --control-cells data/acceptance/hrdexdb-expanded-v2/eval/expanded-v2-control \\
        --tactile-mode real --label "v3 tactile + stack"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.eval import stack


def main() -> None:
    """Fit on dev, apply to test, write the result next to the tactile cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="experiment base (norm, classifier)"
    )
    parser.add_argument("--tactile-cells", type=Path, required=True)
    parser.add_argument("--control-cells", type=Path, required=True)
    parser.add_argument("--tactile-mode", default="real")
    parser.add_argument("--control-mode", default="omitted")
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--label", default="tactile + stack")
    parser.add_argument(
        "--out", type=Path, help="output stem (default <tactile-cells>/stack-result)"
    )
    args = parser.parse_args()
    force_cfg = load_config(args.base / "config/force.yaml", ForceConfig)
    source_cfg = stack.load_source_cfg(args.base)
    frames: dict[str, pd.DataFrame] = {}
    for split in (args.fit_split, args.apply_split):
        bench = f"hrdexdb_aligned_{split}"
        tactile = pd.read_parquet(
            args.tactile_cells / bench / args.tactile_mode / "predictions.parquet"
        )
        control = pd.read_parquet(
            args.control_cells / bench / args.control_mode / "predictions.parquet"
        )
        p = None
        if "force_classifier_p" not in tactile.columns:
            p = stack.classifier_probabilities(
                args.base, args.base / f"manifest/{split}.parquet", force_cfg, source_cfg
            )
        frames[split] = stack.stack_frame(tactile, control, p)
    fit = frames[args.fit_split]
    model = stack.fit_stack(fit, l2=args.l2, fit_split=args.fit_split)
    result = {
        "label": args.label,
        "model": model.model_dump(),
        "inputs": {
            "tactile_cells": str(args.tactile_cells),
            "control_cells": str(args.control_cells),
            "tactile_mode": args.tactile_mode,
            "control_mode": args.control_mode,
        },
        "splits": {},
    }
    for split, frame in frames.items():
        p = (
            stack.oof_predict(frame, l2=args.l2)
            if split == args.fit_split
            else stack.predict(model, frame)
        )
        kind = (
            "object-grouped out-of-fold" if split == args.fit_split else f"fit on {args.fit_split}"
        )
        result["splits"][split] = {
            "kind": kind,
            "stack": stack.stack_metrics(p, frame["label"].to_numpy()),
            "tactile_vlm": stack.stack_metrics(
                (frame["tactile_E"] >= 4.0).astype(float).to_numpy(), frame["label"].to_numpy()
            )
            | {
                "auroc": stack.stack_metrics(
                    frame["tactile_E"].to_numpy() / 5.0, frame["label"].to_numpy()
                )["auroc"]
            },
            "control_vlm": stack.stack_metrics(
                (frame["control_E"] >= 4.0).astype(float).to_numpy(), frame["label"].to_numpy()
            )
            | {
                "auroc": stack.stack_metrics(
                    frame["control_E"].to_numpy() / 5.0, frame["label"].to_numpy()
                )["auroc"]
            },
            "classifier_only": stack.stack_metrics(
                1 / (1 + np.exp(-frame["classifier_logit"].to_numpy())), frame["label"].to_numpy()
            ),
            "dropped_rows": int(frame.attrs.get("dropped", 0)),
        }
    out = args.out or args.tactile_cells / "stack-result"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# {args.label}",
        "",
        "| split | kind | model | accuracy | AUROC | failure recall | n |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for split, r in result["splits"].items():
        for name in ("control_vlm", "tactile_vlm", "classifier_only", "stack"):
            m = r[name]
            lines.append(
                f"| {split} | {r['kind']} | {name} | {m['accuracy']:.3f} | {m['auroc']:.3f} | "
                f"{m['failure_recall']:.3f} | {int(m['n'])} |"
            )
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
