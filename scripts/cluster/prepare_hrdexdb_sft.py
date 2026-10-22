"""Freeze the object-disjoint HRDexDB outcome-SFT split and fit train-only force scaling."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.force.resample import resample
from tactile_flora.sources.base import make_adapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

BASE = Path("data/acceptance/hrdexdb-pilot")
DATA_ROOT = BASE


def main() -> None:
    """Build the fixed pilot manifest and normalization from training episodes only."""
    source_cfg = load_config(BASE / "config/sources/hrdexdb.yaml", SourceConfig)
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    adapter = make_adapter("hrdexdb", source_cfg)
    ids = sorted(adapter.discover(Path(source_cfg.root)))
    split_doc = json.loads((BASE / "manifest/split.json").read_text())
    train_doc = json.loads((BASE / "manifest/train-sample.json").read_text())
    val_ids = {r["episode_id"] for r in split_doc["episodes"]}
    train_ids = {r["episode_id"] for r in train_doc["episodes"]}
    if len(train_ids) != 32 or len(val_ids) != 16 or train_ids & val_ids:
        raise RuntimeError("expected 32 train episodes and 16 disjoint validation episodes")
    if set(ids) != train_ids | val_ids:
        raise RuntimeError("downloaded HRDexDB episodes do not match the frozen split manifests")

    features = []
    episodes = {}
    for eid in sorted(train_ids):
        ep = adapter.load(eid)
        episodes[eid] = ep
        times = [
            i * force_cfg.resample.frame_period_s
            for i in range(int(ep.duration_s / force_cfg.resample.frame_period_s) + 1)
        ]
        features.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    consts = normalize.fit(features, force_cfg.resample, source="hrdexdb")
    vdir = DATA_ROOT / "verified/hrdexdb"
    vdir.mkdir(parents=True, exist_ok=True)
    normalize.save(consts, vdir / "norm.json")

    rows = []
    git_hash = full_git_hash()
    for eid in sorted(train_ids | val_ids):
        split = "train" if eid in train_ids else "val"
        ep = episodes.get(eid) or adapter.load(eid)
        for w in episode_windows(ep, force_cfg, consts, split, with_force=False):
            rows.append(
                manifest.ManifestRow(
                    window_id=manifest.window_id(ep.source, eid, w.prompt_kind, w.anchor_s),
                    source=ep.source,
                    episode_id=eid,
                    session=ep.session,
                    split=split,
                    prompt_kind=w.prompt_kind,
                    anchor_s=w.anchor_s,
                    frame_hashes=[],
                    norm_version=consts.version(),
                    label_origin="source",
                    license=ep.license,
                    report_version=git_hash,
                    shift_s=0.0,
                )
            )
    cols = {
        field: [getattr(row, field) for row in rows] for field in manifest.ManifestRow.model_fields
    }
    table = pa.table(cols, schema=manifest._SCHEMA)
    combined = BASE / "manifest/hrdexdb-sft.parquet"
    pq.write_table(table, combined)
    val_rows = [row for row in rows if row.split == "val"]
    pq.write_table(
        pa.table(
            {
                field: [getattr(row, field) for row in val_rows]
                for field in manifest.ManifestRow.model_fields
            },
            schema=manifest._SCHEMA,
        ),
        BASE / "manifest/val.parquet",
    )
    hp = BASE / "manifest/holdout_ids.json"
    if not hp.exists():
        manifest.freeze_holdout([combined], hp, git_hash=git_hash)
    print(
        f"train={len(train_ids)} val={len(val_ids)} windows={len(rows)} "
        f"norm={consts.version()} git={git_hash} manifest={manifest.manifest_hash(combined)}"
    )
    print(consts.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
