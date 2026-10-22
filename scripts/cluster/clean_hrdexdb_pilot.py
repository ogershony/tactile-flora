"""Build a versioned, reviewed pilot subset without changing raw data or frozen splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.sources.base import make_adapter
from tactile_flora.verify import full_git_hash, manifest

BASE = Path("data/acceptance/hrdexdb-pilot")
DECISIONS = Path("docs/experiments/hrdexdb-pilot/cleaning-review.json")


def select_episodes(documents: dict, review: dict) -> dict[str, list[dict]]:
    """Require one explicit, source-consistent keep/exclude decision for every episode."""
    original = {}
    for split, document in documents.items():
        for row in document["episodes"]:
            if row["episode_id"] in original:
                raise ValueError("duplicate episode or overlapping original splits")
            original[row["episode_id"]] = (split, row)
    decisions = review["records"]
    if len({r["episode_id"] for r in decisions}) != len(decisions):
        raise ValueError("duplicate review decisions")
    if {r["episode_id"] for r in decisions} != set(original):
        raise ValueError("review must cover exactly the original episodes")
    kept: dict[str, list[dict]] = {split: [] for split in documents}
    for r in decisions:
        split, row = original[r["episode_id"]]
        if r["split"] != split or r["source_label"] != row["success"]:
            raise ValueError("review changes a frozen split or original label")
        if r["decision"] not in {"keep", "exclude"} or not r["reason"].strip():
            raise ValueError("each decision needs keep/exclude and a reason")
        if r["decision"] == "keep":
            if r["verdict"] not in {"stable_lift", "failed_or_unstable"}:
                raise ValueError("uncertain or unusable clips cannot enter the clean subset")
            if (r["verdict"] == "stable_lift") != row["success"]:
                raise ValueError("kept visual verdict conflicts with the source label")
            kept[split].append(row)
    objects = [{r["object_id"] for r in rows} for rows in kept.values()]
    if len(objects) != 2 or objects[0] & objects[1]:
        raise ValueError("expected two object-disjoint splits")
    if not all(kept.values()):
        raise ValueError("clean train and validation splits must both be nonempty")
    return kept


def main() -> None:
    """Validate review provenance, refit train-only scaling, and export runnable manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--review", type=Path, default=DECISIONS)
    parser.add_argument("--output", type=Path, default=Path("data/acceptance/hrdexdb-clean-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; choose a new version directory")
    review = json.loads(args.review.read_text())
    paths = {
        "train": args.base / "manifest/train-sample.json",
        "val": args.base / "manifest/split.json",
    }
    documents = {s: json.loads(p.read_text()) for s, p in paths.items()}
    for split, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != review["input_split_sha256"][split]:
            raise ValueError(f"{split} manifest changed since visual review")
    kept = select_episodes(documents, review)
    source_cfg = load_config(args.base / "config/sources/hrdexdb.yaml", SourceConfig)
    raw = Path(source_cfg.root).resolve()
    for r in review["records"]:
        row = next(
            x for x in documents[r["split"]]["episodes"] if x["episode_id"] == r["episode_id"]
        )
        trial = raw / r["episode_id"]
        if (
            json.loads((trial / "grasp_result.json").read_text())["grasp_success"]
            != r["source_label"]
        ):
            raise ValueError("raw label changed since review")
        video = trial / "vid" / f"{row['camera_id']}.mp4"
        if hashlib.sha256(video.read_bytes()).hexdigest() != r["video_sha256"]:
            raise ValueError("video changed since visual review")
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    adapter = make_adapter("hrdexdb", source_cfg)
    features = []
    for row in kept["train"]:
        ep = adapter.load(row["episode_id"])
        period = force_cfg.resample.frame_period_s
        times = [i * period for i in range(int(ep.duration_s / period) + 1)]
        features.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    consts = normalize.fit(features, force_cfg.resample, source="hrdexdb")
    manifest_dir = args.output / "manifest"
    manifest_dir.mkdir(parents=True)
    for split, path in paths.items():
        name = path.name
        doc = dict(
            documents[split],
            episodes=kept[split],
            purpose="AI-reviewed clean pilot v1; original split assignments retained",
        )
        (manifest_dir / name).write_text(json.dumps(doc, indent=2) + "\n")
    original = pq.read_table(args.base / "manifest/hrdexdb-sft.parquet")
    keep_ids = {r["episode_id"] for rows in kept.values() for r in rows}
    version = full_git_hash()
    rows = [
        dict(r, norm_version=consts.version(), report_version=version)
        for r in original.to_pylist()
        if r["episode_id"] in keep_ids
    ]
    if {r["episode_id"] for r in rows} != keep_ids:
        raise ValueError("source parquet does not cover the selected episodes")
    for split, name in [(None, "hrdexdb-sft"), ("train", "train"), ("val", "val")]:
        selected = [r for r in rows if split is None or r["split"] == split]
        pq.write_table(
            pa.Table.from_pylist(selected, schema=original.schema), manifest_dir / f"{name}.parquet"
        )
    # Excluded validation IDs remain held out, preventing their later reuse in training.
    shutil.copy2(args.base / "manifest/holdout_ids.json", manifest_dir / "holdout_ids.json")
    norm_path = args.output / "verified/hrdexdb/norm.json"
    normalize.save(consts, norm_path)
    args.output.joinpath("raw").symlink_to(raw, target_is_directory=True)
    source_dir = args.output / "config/sources"
    source_dir.mkdir(parents=True)
    source_doc = source_cfg.model_dump()
    source_doc["root"] = str(raw)
    (source_dir / "hrdexdb.yaml").write_text(yaml.safe_dump(source_doc, sort_keys=False))
    train_cfg = yaml.safe_load((args.base / "config/train-hrdexdb.yaml").read_text())
    train_cfg.update(
        manifest=str(manifest_dir / "hrdexdb-sft.parquet"),
        output_dir=str(args.output / "runs"),
    )
    (args.output / "config/train-hrdexdb.yaml").write_text(
        yaml.safe_dump(train_cfg, sort_keys=False)
    )
    (args.output / "config/train-control.yaml").write_text(
        yaml.safe_dump(dict(train_cfg, force_mode="omitted"), sort_keys=False)
    )
    eval_cfg = {
        "output_dir": str(args.output / "eval"),
        "benchmarks": {
            "hrdexdb_clean_outcome": {
                "manifest": str(manifest_dir / "val.parquet"),
                "prompts": ["outcome"],
                "metrics": ["mae", "accuracy", "auroc"],
            }
        },
    }
    (args.output / "config/eval.yaml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False))
    shutil.copy2(args.review, args.output / "cleaning-review.json")
    summary = {
        "scope": "all 48 locally downloaded HRDexDB pilot episodes, not the full release",
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "original_counts": {s: len(d["episodes"]) for s, d in documents.items()},
        "kept_counts": {s: len(r) for s, r in kept.items()},
        "class_counts": {
            s: {
                "success": sum(r["success"] for r in rows),
                "failure": sum(not r["success"] for r in rows),
            }
            for s, rows in kept.items()
        },
        "excluded": [r for r in review["records"] if r["decision"] == "exclude"],
        "normalization_fit_ids": [r["episode_id"] for r in kept["train"]],
        "norm_version": consts.version(),
        "manifest_hashes": {
            name: manifest.manifest_hash(manifest_dir / f"{name}.parquet")
            for name in ["hrdexdb-sft", "train", "val"]
        },
        "force_sync_status": "unverified; visual cleaning does not repair sensor/video timing",
        "evaluation_status": "post hoc curated development subset; not an untouched benchmark",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
