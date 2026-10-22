"""Audit a frozen HRDexDB download and prepare aligned, object-disjoint SFT manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.schema import HoldoutFile
from tactile_flora.sources.hrdexdb import HrdexdbAdapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

BASE = Path("data/acceptance/hrdexdb-expanded-v2")
REVIEW = Path("docs/experiments/hrdexdb-pilot/cleaning-review.json")


def inspect(row: dict, raw: Path, adapter: HrdexdbAdapter, known_bad: set[str]) -> dict:
    """Decode the complete video and validate clocks/arrays without model predictions."""
    eid = row["episode_id"]
    result = {**row, "excluded_reasons": []}
    reasons = result["excluded_reasons"]
    if eid in known_bad:
        reasons.append("previous visual review excluded this outcome/capture")
    try:
        trial = raw / eid
        labels = json.loads((trial / "grasp_result.json").read_text())
        if labels.get("grasp_success") is not row["success"]:
            raise ValueError("raw and frozen catalog labels disagree")
        lengths = set()
        for name in ["force", "position", "time"]:
            array = np.load(trial / f"raw/hand/{name}.npy", allow_pickle=False)
            if len(array) < 2 or not np.isfinite(array).all():
                raise ValueError(f"invalid {name} sensor array")
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError("hand force, position and timestamp lengths disagree")
        sensor_time = np.load(trial / "raw/hand/time.npy").reshape(-1)
        if not (np.diff(sensor_time) > 0).all():
            raise ValueError("non-increasing sensor timestamps")
        ep = adapter.load(eid)
        if len(ep.force.t) < 2 or not np.isfinite(ep.force.t).all():
            raise ValueError("invalid mapped force times")
        alignment = ep.meta["clock_alignment"]
        if alignment["max_capture_gap_s"] > 0.25:
            reasons.append("recorded camera timestamp gap exceeds 0.25 seconds")
        video = trial / "vid/22641005.mp4"
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            count = sum(1 for _ in container.decode(stream))
        if count != ep.cameras["scene"].n_frames:
            raise ValueError("decoded video length differs from frame metadata")
        result.update(
            clock_alignment=alignment,
            duration_s=ep.duration_s,
            decoded_frames=count,
            video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            sensor_samples=len(sensor_time),
            aligned_samples=len(ep.force.t),
        )
    except (OSError, ValueError, IndexError, KeyError) as exc:
        reasons.append(f"mechanical audit failed: {exc}")
    return result


def main() -> None:
    """Apply fixed quality exclusions and fit force normalization on training data only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    base, raw = args.base, args.base / "raw"
    if (base / "manifest").exists():
        raise FileExistsError("prepared manifest already exists; use a new dataset root")
    plan = json.loads((base / "download-plan.json").read_text())
    entries = plan["episodes"]
    review = json.loads(REVIEW.read_text())
    known_bad = {r["episode_id"] for r in review["records"] if r["decision"] == "exclude"}
    source_cfg = load_config("configs/sources/hrdexdb.yaml", SourceConfig).model_copy(
        update={"root": str(raw.resolve()), "hrdexdb_use_video_timestamps": True}
    )
    adapter = HrdexdbAdapter(source_cfg)
    adapter.discover(raw)
    audits = []
    cache_dir = base / "audit-cache"
    cache_dir.mkdir(exist_ok=True)

    def cached_inspect(row):
        # The root is revision-pinned, and downloaded files have SHA256 receipts.
        key = hashlib.sha256(row["episode_id"].encode()).hexdigest()
        path = cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        result = inspect(row, raw, adapter, known_bad)
        path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(cached_inspect, entries), 1):
            audits.append(result)
            if index % 20 == 0:
                print(f"audited {index}/{len(entries)} complete videos", flush=True)
    duplicates = defaultdict(list)
    for row in audits:
        if row.get("video_sha256"):
            duplicates[row["video_sha256"]].append(row)
    for rows in duplicates.values():
        if len(rows) > 1:
            for row in rows:
                row["excluded_reasons"].append("exact duplicate video; all copies excluded")
    summary = {
        "scope": "mechanical quality checks; new source labels have no independent human signoff",
        "download_plan_sha256": hashlib.sha256(
            (base / "download-plan.json").read_bytes()
        ).hexdigest(),
        "rules": (
            "previous ten exclusions; invalid arrays/clocks/labels/decodes; "
            "gaps >0.25s; exact duplicates"
        ),
        "records": audits,
    }
    (base / "alignment-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    kept = [r for r in audits if not r["excluded_reasons"]]
    objects = {
        s: {r["object_id"] for r in entries if r["split"] == s} for s in ["train", "val", "test"]
    }
    if any(
        objects[a] & objects[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("frozen object splits overlap")
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    features = []
    for row in kept:
        if row["split"] != "train":
            continue
        ep = adapter.load(row["episode_id"])
        times = np.arange(0, ep.duration_s, force_cfg.resample.frame_period_s).tolist()
        features.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    if not features:
        raise ValueError("no training episodes passed the fixed audit rules")
    consts = normalize.fit(features, force_cfg.resample, source="hrdexdb")
    version = full_git_hash()
    rows = []
    for entry in kept:
        ep = adapter.load(entry["episode_id"])
        for window in episode_windows(ep, force_cfg, consts, entry["split"], with_force=False):
            rows.append(
                manifest.ManifestRow(
                    window_id=manifest.window_id(
                        ep.source, ep.episode_id, window.prompt_kind, window.anchor_s
                    ),
                    source=ep.source,
                    episode_id=ep.episode_id,
                    session=ep.session,
                    split=entry["split"],
                    prompt_kind=window.prompt_kind,
                    anchor_s=window.anchor_s,
                    frame_hashes=[],
                    norm_version=consts.version(),
                    label_origin="source",
                    license=ep.license,
                    report_version=version,
                    shift_s=0.0,
                ).model_dump()
            )
    directory = base / "manifest"
    directory.mkdir()
    for name in ["train", "val", "test", "hrdexdb-sft"]:
        subset = [
            r
            for r in rows
            if (r["split"] in {"train", "val"} if name == "hrdexdb-sft" else r["split"] == name)
        ]
        pq.write_table(
            pa.Table.from_pylist(subset, schema=manifest._SCHEMA), directory / f"{name}.parquet"
        )
    # Freeze excluded holdout episodes too. They can never silently become training examples.
    holdout = HoldoutFile.model_validate(
        {
            "git_hash": version,
            "splits": {f"hrdexdb/{r['episode_id']}": r["split"] for r in entries},
            "manifests": {
                name: manifest.manifest_hash(directory / f"{name}.parquet")
                for name in ["train", "val", "test", "hrdexdb-sft"]
            },
        }
    )
    (directory / "holdout_ids.json").write_text(holdout.model_dump_json(indent=2) + "\n")
    normalize.save(consts, base / "verified/hrdexdb/norm.json")
    config_dir = base / "config/sources"
    config_dir.mkdir(parents=True)
    (config_dir / "hrdexdb.yaml").write_text(
        yaml.safe_dump(source_cfg.model_dump(), sort_keys=False)
    )
    template = yaml.safe_load(
        Path("data/acceptance/hrdexdb-clean-v1/config/train-hrdexdb.yaml").read_text()
    )
    template.update(manifest=str(directory / "hrdexdb-sft.parquet"), output_dir=str(base / "runs"))
    for name, mode in [("train-hrdexdb", "real"), ("train-control", "omitted")]:
        (base / f"config/{name}.yaml").write_text(yaml.safe_dump(dict(template, force_mode=mode)))
    for split in ["val", "test"]:
        config = {
            "output_dir": str(base / "eval"),
            "benchmarks": {
                f"hrdexdb_aligned_{split}": {
                    "manifest": str(directory / f"{split}.parquet"),
                    "prompts": ["outcome"],
                    "metrics": ["mae", "accuracy", "auroc"],
                }
            },
        }
        (base / f"config/eval-{split}.yaml").write_text(yaml.safe_dump(config))
    summary["kept_counts"] = dict(Counter(r["split"] for r in kept))
    summary["excluded_count"] = len(audits) - len(kept)
    summary["normalization_fit_ids"] = [r["episode_id"] for r in kept if r["split"] == "train"]
    summary["manifest_hashes"] = holdout.manifests
    (base / "alignment-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ["kept_counts", "excluded_count", "manifest_hashes"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
