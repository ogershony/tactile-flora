"""Run the held-out-safe, single-hand HRDexDB outcome SFT experiment."""

from __future__ import annotations

import argparse
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

BASE = Path("data/acceptance/hrdexdb-pilot")
DATA_ROOT = BASE


def main() -> None:
    """Run the explicit HRDexDB training pilot with the frozen object split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=BASE / "config/train-hrdexdb.yaml")
    parser.add_argument("--run-name", default="hrdexdb-outcome-sft")
    args = parser.parse_args()
    cfg = load_config(args.config, TrainConfig)
    force_cfg = load_config(
        "data/acceptance/released-inference/force.real-tokenizer.yaml", ForceConfig
    )
    source_cfg = load_config(BASE / "config/sources/hrdexdb.yaml", SourceConfig)
    adapter = make_adapter("hrdexdb", source_cfg)
    ids = sorted(adapter.discover(Path(source_cfg.root)))
    vdir = DATA_ROOT / "verified/hrdexdb"
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
        DATA_ROOT,
        dry_run=args.dry_run,
        run_name=args.run_name,
        windows_source=windows_source,
        eval_only_sources=set(),  # the manifest has a fixed train split, object-disjoint from val
    )
    print(f"training complete: {run_dir}")


if __name__ == "__main__":
    main()
