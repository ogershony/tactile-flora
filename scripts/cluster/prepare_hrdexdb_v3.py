"""Prepare the v3 HRDexDB experiment: corrected force representation on the frozen v2 splits.

The v2 run rendered an almost uninformative force block: the adapter ranked channels by mean
(resting offsets), its hand-closure proxy was inverted for inspire_dftp, and the fitted grasp
scale (37.5 raw units) saturated every contact at F=3.00. This script keeps the v2 download,
mechanical audit, exclusions and object-disjoint splits untouched and rebuilds manifests,
train-only normalization and configs with the ``tared`` representation plus a contact-threshold
floor. Window ids are identical to v2; only ``norm_version`` and ``report_version`` change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.labels.windows import outcome_frame_indices
from tactile_flora.schema import HoldoutFile
from tactile_flora.sources.hrdexdb import HrdexdbAdapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

PREVIOUS = Path("data/acceptance/hrdexdb-expanded-v2")
BASE = Path("data/acceptance/hrdexdb-expanded-v3")
FORCE_TEMPLATE = Path("data/acceptance/released-inference/force.real-tokenizer.yaml")


def main() -> None:
    """Rebuild manifests and normalization with the corrected representation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--contact-min-frac", type=float, default=0.05)
    parser.add_argument("--tare-s", type=float, default=1.0)
    parser.add_argument("--summary", action="store_true", help="Enable render.summary lines.")
    args = parser.parse_args()
    base, previous = args.base, args.previous
    if (base / "manifest").exists():
        raise FileExistsError("prepared manifest already exists; use a new dataset root")
    base.mkdir(parents=True, exist_ok=True)
    raw = base / "raw"
    if not raw.exists():
        os.symlink((previous / "raw").resolve(), raw, target_is_directory=True)
    audit = json.loads((previous / "alignment-audit.json").read_text())
    kept = [r for r in audit["records"] if not r["excluded_reasons"]]
    previous_holdout = HoldoutFile.model_validate_json(
        (previous / "manifest/holdout_ids.json").read_text()
    )
    for r in audit["records"]:
        if previous_holdout.splits[f"hrdexdb/{r['episode_id']}"] != r["split"]:
            raise ValueError(f"frozen split disagrees with the audit: {r['episode_id']}")
    source_cfg = load_config(previous / "config/sources/hrdexdb.yaml", SourceConfig).model_copy(
        update={
            "root": str(raw.resolve()),
            "hrdexdb_force_representation": "tared",
            "hrdexdb_tare_s": args.tare_s,
        }
    )
    force_template = load_config(FORCE_TEMPLATE, ForceConfig)
    force_cfg = force_template.model_copy(
        update={
            "resample": force_template.resample.model_copy(
                update={"contact_min_frac": args.contact_min_frac}
            ),
            "render": force_template.render.model_copy(update={"summary": args.summary}),
        }
    )
    (base / "config").mkdir(exist_ok=True)
    (base / "config/force.yaml").write_text(
        yaml.safe_dump(force_cfg.model_dump(mode="json"), sort_keys=False)
    )
    adapter = HrdexdbAdapter(source_cfg)
    adapter.discover(raw)
    episodes = {r["episode_id"]: adapter.load(r["episode_id"]) for r in kept}
    features = []
    for row in kept:
        if row["split"] != "train":
            continue
        ep = episodes[row["episode_id"]]
        times = np.arange(0, ep.duration_s, force_cfg.resample.frame_period_s).tolist()
        features.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    consts = normalize.fit(features, force_cfg.resample, source="hrdexdb")
    tokenizer = render.load_tokenizer(force_cfg.render)
    version = full_git_hash()
    rows, records = [], []
    for entry in kept:
        ep = episodes[entry["episode_id"]]
        cam = ep.cameras["scene"]
        idx = outcome_frame_indices(cam.n_frames, cam.fps)
        times = [i / cam.fps for i in idx]
        norm = normalize.apply(
            resample(ep.force, ep.gripper, times, force_cfg.resample), consts, force_cfg.render
        )
        lines = render.render_lines(norm, times, ep.force.channel)
        tokens = render.budget_check(
            render.render_block(lines, force_cfg.render, norm),
            force_cfg.render.max_tokens,
            tokenizer,
        )
        arm = norm.arms["R"]
        forces = np.nan_to_num(np.asarray(arm.F, dtype=np.float64))
        records.append(
            {
                "episode_id": entry["episode_id"],
                "split": entry["split"],
                "success": entry["success"],
                "tactile_channels": ep.meta["tactile_channels"],
                "force_offsets": ep.meta["force_offsets"],
                "gripper_convention": ep.meta["gripper_convention"],
                "outcome_block_tokens": tokens,
                "summary": render.summary_lines(norm) if force_cfg.render.summary else [],
                "outcome_lines": len(lines),
                "F_max": float(forces.max()),
                "F_mean": float(forces.mean()),
                "saturated_fraction": float(np.mean(forces > force_cfg.render.clip_max - 0.01)),
                "contact_fraction": float(np.mean(arm.c or [0])),
            }
        )
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
    previous_ids = {
        name: {r.window_id for r in manifest.read_manifest(previous / f"manifest/{name}.parquet")}
        for name in ["train", "val", "test", "hrdexdb-sft"]
    }
    for name, ids in previous_ids.items():
        current = {r.window_id for r in manifest.read_manifest(directory / f"{name}.parquet")}
        if current != ids:
            raise ValueError(f"window ids differ from the frozen v2 manifest: {name}")
    holdout = HoldoutFile.model_validate(
        {
            "git_hash": version,
            "splits": dict(previous_holdout.splits),
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
    template = yaml.safe_load((previous / "config/train-hrdexdb.yaml").read_text())
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
    summary = {
        "scope": (
            "v3 representation on the frozen v2 audit and splits; same window ids; "
            "train-only normalization refit"
        ),
        "previous_alignment_audit_sha256": hashlib.sha256(
            (previous / "alignment-audit.json").read_bytes()
        ).hexdigest(),
        "representation": source_cfg.model_dump(),
        "force_config": force_cfg.model_dump(mode="json"),
        "normalization": {**consts.model_dump(mode="json"), "norm_version": consts.version()},
        "kept_counts": {s: sum(r["split"] == s for r in kept) for s in ["train", "val", "test"]},
        "manifest_hashes": holdout.manifests,
        "git_hash": version,
        "records": records,
    }
    (base / "representation-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ["normalization", "kept_counts", "manifest_hashes"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
