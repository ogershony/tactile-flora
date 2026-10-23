"""Prepare a rotated object fold of the HRDexDB Inspire DFTP experiment (v4).

The v3 result rests on 251 object-disjoint held-out clips, too few to certify a 3-point accuracy
gain. The inspire_f1 hand was audited as a data source and rejected (its force-only AUROC is no
better than DFTP's and 23% of its successful grasps register no fingertip force at all), so v4
keeps DFTP's known signal and rotates which objects are held out:

- the 601 kept trials, exclusions and tared representation are exactly v3's;
- the 44 objects that v3 TRAINED on are hash-ordered (salt ``hrdexdb-v4-folds:20261022``) and
  split in two halves; fold B holds out the first half, fold C the second, so every v3
  training object is evaluated out-of-object exactly once across the two folds;
- each fold trains on everything else (v3's val and test objects plus the other half), with
  normalization and the force-only classifier refitted on that fold's training trials only;
- the held-out objects alternate test/val in hash order so the eval pipeline keeps its shape;
  the fold's dev and test sets are both new held-out clips and are pooled in the analysis.

Run from the repository root (CPU, about 10 minutes per fold):

    .venv/bin/python scripts/cluster/prepare_hrdexdb_folds.py --fold B
    .venv/bin/python scripts/cluster/prepare_hrdexdb_folds.py --fold C
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import classifier, normalize, render
from tactile_flora.force.resample import resample
from tactile_flora.labels.windows import outcome_frame_indices
from tactile_flora.schema import HoldoutFile, ResampledForce
from tactile_flora.sources.hrdexdb import HrdexdbAdapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

V2 = Path("data/acceptance/hrdexdb-expanded-v2")
V3 = Path("data/acceptance/hrdexdb-expanded-v3")
SALT = "hrdexdb-v4-folds:20261022"


def fold_splits(v2: Path, v3: Path, fold: str) -> list[dict]:
    """Kept DFTP trials with this fold's split; v3's training objects are the held-out pool."""
    audit = json.loads((v2 / "alignment-audit.json").read_text())
    holdout = HoldoutFile.model_validate_json((v3 / "manifest/holdout_ids.json").read_text())
    kept = [r for r in audit["records"] if not r["excluded_reasons"]]
    v3_split = {r["object_id"]: holdout.splits[f"hrdexdb/{r['episode_id']}"] for r in kept}
    pool = sorted(
        (o for o, s in v3_split.items() if s == "train"),
        key=lambda o: hashlib.sha256(f"{SALT}:{o}".encode()).hexdigest(),
    )
    half = len(pool) // 2
    held = pool[:half] if fold == "B" else pool[half:]
    assignment = {o: "train" for o in v3_split}
    for i, o in enumerate(held):
        assignment[o] = "test" if i % 2 == 0 else "val"
    return [
        {
            "episode_id": r["episode_id"],
            "object_id": r["object_id"],
            "success": bool(r["success"]),
            "split": assignment[r["object_id"]],
            "v3_split": v3_split[r["object_id"]],
        }
        for r in kept
    ]


def main() -> None:
    """Build one fold: splits, train-only normalization and classifier, manifests, configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["B", "C"], required=True)
    parser.add_argument("--v2", type=Path, default=V2)
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--early-stop-score-loss", type=float, default=0.03)
    parser.add_argument("--early-stop-min-epochs", type=int, default=2)
    args = parser.parse_args()
    base = args.base or Path(f"data/acceptance/hrdexdb-expanded-v4-fold{args.fold}")
    v3 = args.v3
    if (base / "manifest").exists():
        raise FileExistsError("prepared manifest already exists; use a new dataset root")
    base.mkdir(parents=True, exist_ok=True)
    raw = base / "raw"
    if not raw.exists():
        os.symlink((args.v2 / "raw").resolve(), raw, target_is_directory=True)
    kept = fold_splits(args.v2, v3, args.fold)
    train = [r for r in kept if r["split"] == "train"]
    held = [r for r in kept if r["split"] != "train"]
    if {r["v3_split"] for r in held} != {"train"}:
        raise ValueError("held-out clips must all be v3 training clips")

    force_cfg = load_config(v3 / "config/force.yaml", ForceConfig)
    source_cfg = load_config(v3 / "config/sources/hrdexdb.yaml", SourceConfig).model_copy(
        update={"root": str(raw.resolve())}
    )
    adapter = HrdexdbAdapter(source_cfg)
    adapter.discover(raw)
    episodes = {r["episode_id"]: adapter.load(r["episode_id"]) for r in kept}
    feats = []
    for r in train:
        ep = episodes[r["episode_id"]]
        times = np.arange(0, ep.duration_s, force_cfg.resample.frame_period_s).tolist()
        feats.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    consts = normalize.fit(feats, force_cfg.resample, source="hrdexdb")
    outcome_norm: dict[str, ResampledForce] = {}
    for r in kept:
        ep = episodes[r["episode_id"]]
        cam = ep.cameras["scene"]
        times = [i / cam.fps for i in outcome_frame_indices(cam.n_frames, cam.fps)]
        outcome_norm[r["episode_id"]] = normalize.apply(
            resample(ep.force, ep.gripper, times, force_cfg.resample), consts, force_cfg.render
        )
    scorer = classifier.fit_cv(
        np.stack([classifier.extract_features(outcome_norm[r["episode_id"]]) for r in train]),
        np.asarray([int(r["success"]) for r in train]),
        [r["object_id"] for r in train],
        [r["episode_id"] for r in train],
        source="hrdexdb",
    )
    (base / "config").mkdir(exist_ok=True)
    scorer_path = base / "config/force-classifier.json"
    scorer.save(scorer_path)
    force_cfg = force_cfg.model_copy(
        update={
            "render": force_cfg.render.model_copy(
                update={"classifier_json": str(scorer_path.resolve())}
            )
        }
    )
    (base / "config/force.yaml").write_text(
        yaml.safe_dump(force_cfg.model_dump(mode="json"), sort_keys=False)
    )
    tokenizer = render.load_tokenizer(force_cfg.render)
    version = full_git_hash()
    rows: list[dict] = []
    records: list[dict[str, Any]] = []
    for r in kept:
        ep = episodes[r["episode_id"]]
        norm = outcome_norm[r["episode_id"]]
        lines = render.render_lines(norm, list(norm.frame_times), ep.force.channel)
        tokens = render.budget_check(
            render.render_block(lines, force_cfg.render, norm, ep.episode_id),
            force_cfg.render.max_tokens,
            tokenizer,
        )
        arm = norm.arms["R"]
        forces = np.nan_to_num(np.asarray(arm.F, dtype=np.float64))
        records.append(
            {
                **r,
                "outcome_block_tokens": tokens,
                "F_max": float(forces.max()),
                "contact_fraction": float(np.mean(arm.c or [0])),
            }
        )
        for window in episode_windows(ep, force_cfg, consts, r["split"], with_force=False):
            rows.append(
                manifest.ManifestRow(
                    window_id=manifest.window_id(
                        ep.source, ep.episode_id, window.prompt_kind, window.anchor_s
                    ),
                    source=ep.source,
                    episode_id=ep.episode_id,
                    session=ep.session,
                    split=r["split"],
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
    names = ["train", "val", "test", "hrdexdb-sft"]
    for name in names:
        subset = [
            x
            for x in rows
            if (x["split"] in {"train", "val"} if name == "hrdexdb-sft" else x["split"] == name)
        ]
        pq.write_table(
            pa.Table.from_pylist(subset, schema=manifest._SCHEMA), directory / f"{name}.parquet"
        )
    v3_held = {
        x.window_id
        for n in ("val", "test")
        for x in manifest.read_manifest(v3 / f"manifest/{n}.parquet")
    }
    fold_held = {
        x.window_id
        for n in ("val", "test")
        for x in manifest.read_manifest(directory / f"{n}.parquet")
    }
    if v3_held & fold_held:
        raise ValueError("fold held-out windows overlap v3's held-out windows")
    holdout = HoldoutFile.model_validate(
        {
            "git_hash": version,
            "splits": {f"hrdexdb/{r['episode_id']}": r["split"] for r in kept},
            "manifests": {n: manifest.manifest_hash(directory / f"{n}.parquet") for n in names},
        }
    )
    (directory / "holdout_ids.json").write_text(holdout.model_dump_json(indent=2) + "\n")
    normalize.save(consts, base / "verified/hrdexdb/norm.json")
    config_dir = base / "config/sources"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "hrdexdb.yaml").write_text(
        yaml.safe_dump(source_cfg.model_dump(), sort_keys=False)
    )
    template = yaml.safe_load((v3 / "config/train-hrdexdb.yaml").read_text())
    template.update(
        manifest=str(directory / "hrdexdb-sft.parquet"),
        output_dir=str(base / "runs"),
        epochs=args.epochs,
        early_stop_score_loss=args.early_stop_score_loss,
        early_stop_window=10,
        early_stop_min_epochs=args.early_stop_min_epochs,
    )
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

    def counts(sel: list[dict]) -> dict:
        return {
            s: {
                "clips": sum(r["split"] == s for r in sel),
                "failures": sum(r["split"] == s and not r["success"] for r in sel),
                "objects": len({r["object_id"] for r in sel if r["split"] == s}),
            }
            for s in ["train", "val", "test"]
        }

    summary = {
        "scope": f"v4 fold {args.fold}: v3's DFTP trials and representation, rotated object split",
        "split_rule": (
            f"v3 training objects hash-ordered with salt {SALT}; fold B holds out the first "
            "half, fold C the second; held-out objects alternate test/val"
        ),
        "held_out_objects": sorted({r["object_id"] for r in held}),
        "normalization": {**consts.model_dump(mode="json"), "norm_version": consts.version()},
        "force_classifier": {
            "path": str(scorer_path),
            "sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
            "cv_metrics": scorer.cv_metrics,
            "fit_episodes": len(train),
        },
        "training": {
            k: template[k] for k in ("epochs", "early_stop_score_loss", "early_stop_min_epochs")
        },
        "kept_counts": counts(kept),
        "manifest_hashes": holdout.manifests,
        "git_hash": version,
        "records": records,
    }
    (base / "representation-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ["kept_counts", "held_out_objects", "force_classifier"]},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
