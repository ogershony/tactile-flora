"""Run the held-out-safe, single-hand HRDexDB outcome SFT experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tactile_flora.config import ForceConfig, SourceConfig, TrainConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.schema import ManifestRow
from tactile_flora.sources.base import make_adapter
from tactile_flora.train.dataset import WindowItem
from tactile_flora.train.sft import train
from tactile_flora.verify import manifest, report
from tactile_flora.verify.windowing import (
    build_source_inputs,
    episode_windows,
    prepare_episode,
    resample_shifted,
)

BASE = Path("data/acceptance/hrdexdb-clean-v1")


def planned_report_version(base: Path, cfg: TrainConfig) -> str | None:
    """Keep an active Slurm experiment pinned when unrelated commits advance HEAD."""
    path = base / "training-plan.json"
    if not path.exists():
        return None
    plan = json.loads(path.read_text())
    job = os.environ.get("SLURM_JOB_ID")
    if job is None or str(plan.get("slurm_training_job")) != job:
        return None
    if cfg.model_dump() not in plan["configs"].values():
        raise ValueError("configuration differs from the active training plan")
    if manifest.manifest_hash(cfg.manifest) != plan["manifest_sha256"]:
        raise ValueError("manifest differs from the active training plan")
    hashes = plan.get("core_code_sha256")
    if not hashes:
        raise ValueError("active training plan has no pinned core code hashes")
    for name, expected in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"core code changed during the experiment: {name}")
    for name, expected in plan.get("input_sha256", {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"learning input changed during the experiment: {name}")
    return plan["git_hash"]


def main() -> None:
    """Run the explicit HRDexDB training pilot with the frozen object split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-name", default="hrdexdb-outcome-sft")
    args = parser.parse_args()
    cfg = load_config(args.config or args.base / "config/train-hrdexdb.yaml", TrainConfig)
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    source_cfg = load_config(args.base / "config/sources/hrdexdb.yaml", SourceConfig)
    adapter = make_adapter("hrdexdb", source_cfg)
    adapter.discover(Path(source_cfg.root))
    ids = sorted({r.episode_id for r in manifest.read_manifest(cfg.manifest)})
    vdir = args.base / "verified/hrdexdb"
    consts = normalize.load(vdir / "norm.json")
    checks = report.read_checks(vdir) if (vdir / "checks.parquet").exists() else {}
    inputs = build_source_inputs(adapter, force_cfg, consts, vdir, ids, checks)
    cache: dict[str, WindowItem] = {}

    def windows_source(row: ManifestRow) -> WindowItem | None:
        if row.window_id in cache:
            return cache[row.window_id]
        ep = adapter.load(row.episode_id)
        prepared = prepare_episode(ep, inputs)
        if prepared is None:
            return None
        windows = episode_windows(ep, force_cfg, consts, row.split, inputs=inputs)
        by_id = {
            manifest.window_id(ep.source, ep.episode_id, w.prompt_kind, w.anchor_s): w
            for w in windows
        }
        w = by_id.get(row.window_id)
        if w is None:
            return None
        features = normalize.apply(
            resample_shifted(
                prepared,
                w.frame_times,
                force_cfg.resample,
                inputs.shift_of(ep.episode_id),
            ),
            consts,
            force_cfg.render,
        )
        item = WindowItem(
            window=w,
            task=ep.task,
            camera_roles=tuple(ep.cameras) or ("scene",),
            fps=ep.cameras.get("scene").fps if ep.cameras.get("scene") else None,
            features=features,
            episode=ep,
        )
        cache[row.window_id] = item
        return item

    run_dir = train(
        cfg,
        force_cfg,
        args.base,
        dry_run=args.dry_run,
        run_name=args.run_name,
        windows_source=windows_source,
        eval_only_sources=set(),  # the manifest has a fixed train split, object-disjoint from val
        report_version=planned_report_version(args.base, cfg),
    )
    print(f"training complete: {run_dir}")


if __name__ == "__main__":
    main()
