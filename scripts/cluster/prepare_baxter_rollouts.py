"""Prepare a Baxter rollout dataset (LeRobot v3, FSR fingertips) for the outcome evaluation.

Copies the dataset under ``<base>/raw``, writes the operator's outcome labels as
``raw/labels.json``, fits force normalization constants on the rollouts themselves (there is no
training split: every episode is a benchmark episode), builds the outcome manifest and the
source, force and eval configs, and records a representation audit with the force block each
episode will show. The force config (``configs/eval/baxter_rollouts/force.yaml``, the v3
HRDexDB one with its force-only classifier) keeps rendering what the fine-tuned adapters were
trained with; paths are made absolute in the copy written under ``<base>/config`` so the
evaluation cells are self-describing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import normalize, render
from tactile_flora.force.classifier import extract_features, load_scorer
from tactile_flora.force.resample import resample
from tactile_flora.labels.windows import outcome_frame_indices
from tactile_flora.schema import ResampledForce
from tactile_flora.sources.baxter_rollouts import BaxterRolloutsAdapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

BASE = Path("data/acceptance/baxter-rollouts-v1")
FORCE_CONFIG = Path("configs/eval/baxter_rollouts/force.yaml")
BENCHMARK = "baxter_rollouts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Copy, label, normalize, window and record one rollout dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Downloaded dataset dir.")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--force-config",
        type=Path,
        default=FORCE_CONFIG,
        help="Force config (resampling, rendering, classifier) the adapters were trained with.",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="tokenizer.json of the base model for exact token budgets (default: estimate).",
    )
    parser.add_argument(
        "--success-episodes",
        default="",
        help="Comma-separated episode indices the operator calls successes; every other "
        "episode is a failure. Empty keeps the recorder's labels.",
    )
    parser.add_argument("--label-note", default="operator statement")
    parser.add_argument("--arm", choices=["L", "R"], default="R")
    parser.add_argument("--cameras", default="head", help="Comma-separated video keys.")
    parser.add_argument("--dataset-id", default="")
    args = parser.parse_args()
    base: Path = args.base
    if base.exists():
        raise FileExistsError(f"{base} exists; use a new base directory")
    raw = base / "raw"
    shutil.copytree(args.dataset, raw)
    # Downloaded files are data: keep the copy read-only apart from our label override.
    labels: dict[str, Any] = {}
    if args.success_episodes:
        successes = {int(v) for v in args.success_episodes.split(",") if v.strip()}
        adapter_probe = BaxterRolloutsAdapter(
            SourceConfig(
                name="baxter_rollouts",
                root=str(raw.resolve()),
                license_allowed=["apache-2.0"],
                max_windows=0,
                max_episodes=0,
                eval_only=True,
                expected_label_rate=1.0,
                seed=0,
            )
        )
        for eid in adapter_probe.discover(raw):
            idx = int(eid.removeprefix("ep"))
            labels[eid] = {
                "outcome": 5 if idx in successes else 1,
                "note": args.label_note,
            }
        (raw / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")

    source_cfg = SourceConfig(
        name="baxter_rollouts",
        root=str(raw.resolve()),
        license_allowed=["apache-2.0"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=1.0,
        split_rules={"by": "session", "note": "eval only; every rollout is a benchmark episode"},
        baxter_rollouts_arm=args.arm,
        baxter_rollouts_cameras=[c for c in args.cameras.split(",") if c],
        seed=0,
    )
    config_dir = base / "config" / "sources"
    config_dir.mkdir(parents=True)
    (config_dir / "baxter_rollouts.yaml").write_text(
        yaml.safe_dump(source_cfg.model_dump(mode="json"), sort_keys=False)
    )
    force_cfg = load_config(args.force_config, ForceConfig)
    classifier = force_cfg.render.classifier_json
    force_cfg = force_cfg.model_copy(
        update={
            "render": force_cfg.render.model_copy(
                update={
                    "tokenizer_json": (
                        str(args.tokenizer_json.resolve()) if args.tokenizer_json else None
                    ),
                    "classifier_json": str(Path(classifier).resolve()) if classifier else None,
                }
            )
        }
    )
    (base / "config/force.yaml").write_text(
        yaml.safe_dump(force_cfg.model_dump(mode="json"), sort_keys=False)
    )

    adapter = BaxterRolloutsAdapter(source_cfg)
    ids = adapter.discover(raw)
    episodes = {eid: adapter.load(eid) for eid in ids}
    grid_features = []
    for ep in episodes.values():
        times = np.arange(0, ep.duration_s, force_cfg.resample.frame_period_s).tolist()
        grid_features.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    consts = normalize.fit(grid_features, force_cfg.resample, source="baxter_rollouts")
    normalize.save(consts, base / "verified/baxter_rollouts/norm.json")

    scorer = (
        load_scorer(force_cfg.render.classifier_json) if force_cfg.render.classifier_json else None
    )
    tokenizer = render.load_tokenizer(force_cfg.render)
    version = full_git_hash()
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for eid, ep in episodes.items():
        cam = ep.cameras["scene"]
        times = [i / cam.fps for i in outcome_frame_indices(cam.n_frames, cam.fps)]
        norm: ResampledForce = normalize.apply(
            resample(ep.force, ep.gripper, times, force_cfg.resample), consts, force_cfg.render
        )
        lines = render.render_lines(norm, times, ep.force.channel)
        block = render.render_block(lines, force_cfg.render, norm, ep.episode_id)
        tokens = render.budget_check(block, force_cfg.render.max_tokens, tokenizer)
        arm = norm.arms[args.arm]
        forces = np.nan_to_num(np.asarray(arm.F, dtype=np.float64))
        label = next((lab for lab in ep.labels if lab.kind == "outcome"), None)
        records.append(
            {
                "episode_id": eid,
                "outcome": None if label is None else label.outcome,
                "label_origin": None if label is None else label.origin,
                "recorder_label": ep.meta["recorder_label"],
                "stop_reason": ep.meta["stop_reason"],
                "note": ep.meta["note"],
                "n_frames": cam.n_frames,
                "duration_s": ep.duration_s,
                "sampled_frames": len(times),
                "outcome_block_tokens": tokens,
                "summary": render.summary_lines(norm, scorer, eid),
                "classifier_p_success": None if scorer is None else scorer.score(norm),
                "classifier_features": dict(
                    zip(
                        scorer.model.feature_names,
                        extract_features(norm, args.arm).tolist(),
                        strict=True,
                    )
                )
                if scorer is not None
                else None,
                "F_max": float(forces.max()),
                "F_mean": float(forces.mean()),
                "contact_fraction": float(np.mean(arm.c or [0])),
                "g_max": float(max(arm.g or [0.0])),
                "block": block,
            }
        )
        for window in episode_windows(ep, force_cfg, consts, "benchmark", with_force=False):
            rows.append(
                manifest.ManifestRow(
                    window_id=manifest.window_id(
                        ep.source, ep.episode_id, window.prompt_kind, window.anchor_s
                    ),
                    source=ep.source,
                    episode_id=ep.episode_id,
                    session=ep.session,
                    split="benchmark",
                    prompt_kind=window.prompt_kind,
                    anchor_s=window.anchor_s,
                    frame_hashes=[],
                    norm_version=consts.version(),
                    label_origin=window.label_origin,
                    license=ep.license,
                    report_version=version,
                    shift_s=0.0,
                ).model_dump()
            )
    directory = base / "manifest"
    directory.mkdir()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=manifest._SCHEMA), directory / "benchmark.parquet"
    )
    eval_cfg = {
        "output_dir": str(base / "eval"),
        "benchmarks": {
            BENCHMARK: {
                "manifest": str(directory / "benchmark.parquet"),
                "prompts": ["outcome"],
                "metrics": ["mae", "accuracy", "auroc"],
            }
        },
    }
    (base / "config/eval.yaml").write_text(yaml.safe_dump(eval_cfg))
    audit = {
        "scope": (
            "Baxter policy rollouts, FSR fingertips, outcome evaluation only; normalization "
            "fitted on the benchmark episodes themselves (no training split exists)"
        ),
        "dataset": args.dataset_id,
        "labels": labels,
        "source_config": source_cfg.model_dump(mode="json"),
        "force_config": force_cfg.model_dump(mode="json"),
        "force_config_reference": str(args.force_config),
        "normalization": {**consts.model_dump(mode="json"), "norm_version": consts.version()},
        "classifier_json": force_cfg.render.classifier_json,
        "classifier_sha256": (
            _sha256(Path(force_cfg.render.classifier_json))
            if force_cfg.render.classifier_json
            else None
        ),
        "manifest_hash": manifest.manifest_hash(directory / "benchmark.parquet"),
        "git_hash": version,
        "records": records,
    }
    (base / "representation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(records),
                "normalization": audit["normalization"],
                "manifest_hash": audit["manifest_hash"],
                "labels": {r["episode_id"]: r["outcome"] for r in records},
                "classifier_p_success": {
                    r["episode_id"]: r["classifier_p_success"] for r in records
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
