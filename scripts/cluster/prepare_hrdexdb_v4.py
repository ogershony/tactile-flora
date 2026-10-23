"""Prepare the v4 HRDexDB experiment: both Inspire hands, frozen DFTP splits, cross-hand objects.

v4 differs from v3 in data only. The Inspire DFTP half is exactly v3's: the same 601 kept trials,
the same object-disjoint splits, the same window ids. The Inspire F1 half (downloaded by
``download_hrdexdb_v4.py``) is added with these rules:

- an F1 trial of an object that DFTP also has goes to that object's frozen split, so no object
  appears in two splits through the other hand;
- F1-only objects are hash-ordered (``hrdexdb-v4:20261022:<object>``) and the first
  ``--f1-test-objects`` go to test, the next ``--f1-val-objects`` to val, the rest to train;
- an F1 trial that fails the mechanical audit (adapter load with recorded video timestamps) is
  excluded with the reason recorded;
- the two hands report force in different units, so a per-hand scale is fitted on TRAINING
  trials only: the DFTP median grasp divided by the F1 median grasp (each from
  ``normalize.fit`` on that hand's training trials). The scale goes into the source config
  (``hrdexdb_hand_scale``) and one joint normalization is then fitted on all training trials.
- the force-only classifier is refitted on all training trials, grouped by object.

Run from the repository root after the download has finished:

    .venv/bin/python scripts/cluster/prepare_hrdexdb_v4.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from tactile_flora.schema import Episode, HoldoutFile, ResampledForce
from tactile_flora.sources.hrdexdb import HrdexdbAdapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

V2 = Path("data/acceptance/hrdexdb-expanded-v2")
V3 = Path("data/acceptance/hrdexdb-expanded-v3")
BASE = Path("data/acceptance/hrdexdb-expanded-v4")
SPLIT_SALT = "hrdexdb-v4:20261022"


def dftp_records(v2: Path, v3: Path) -> list[dict]:
    """Return the 601 kept DFTP trials with their frozen splits (v3 == v2)."""
    audit = json.loads((v2 / "alignment-audit.json").read_text())
    holdout = HoldoutFile.model_validate_json((v3 / "manifest/holdout_ids.json").read_text())
    kept = []
    for r in audit["records"]:
        if r["excluded_reasons"]:
            continue
        split = holdout.splits[f"hrdexdb/{r['episode_id']}"]
        if split != r["split"]:
            raise ValueError(f"frozen split disagrees with the audit: {r['episode_id']}")
        kept.append(
            {
                "episode_id": r["episode_id"],
                "object_id": r["object_id"],
                "success": bool(r["success"]),
                "split": split,
                "hand": "inspire_dftp",
                "excluded_reasons": [],
            }
        )
    return kept


def f1_records(base: Path, object_split: dict[str, str], n_test: int, n_val: int) -> list[dict]:
    """Every downloaded F1 trial with its split; objects shared with DFTP keep DFTP's split."""
    plan = json.loads((base / "download-plan.json").read_text())
    receipts = json.loads((base / "download-receipts.json").read_text())
    failed = {f["path"].split("/raw/")[0].split("/vid/")[0] for f in receipts["failures"]}
    objects = sorted({e["object_id"] for e in plan["episodes"]})
    new = sorted(
        (o for o in objects if o not in object_split),
        key=lambda o: hashlib.sha256(f"{SPLIT_SALT}:{o}".encode()).hexdigest(),
    )
    assignment = dict(object_split)
    for i, o in enumerate(new):
        assignment[o] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    rows = []
    for e in plan["episodes"]:
        reasons = []
        if e["episode_id"] in failed:
            reasons.append("download failed for at least one file")
        rows.append(
            {
                "episode_id": e["episode_id"],
                "object_id": e["object_id"],
                "success": bool(e["success"]),
                "split": assignment[e["object_id"]],
                "hand": "inspire_f1",
                "excluded_reasons": reasons,
                "shared_object": e["object_id"] in object_split,
            }
        )
    return rows


def fit_consts(
    episodes: dict[str, Episode], rows: list[dict], force_cfg: ForceConfig
) -> normalize.NormConstants:
    """Fit normalization constants on the given trials' full-rate force."""
    feats = []
    for r in rows:
        ep = episodes[r["episode_id"]]
        times = np.arange(0, ep.duration_s, force_cfg.resample.frame_period_s).tolist()
        feats.append(resample(ep.force, ep.gripper, times, force_cfg.resample))
    return normalize.fit(feats, force_cfg.resample, source="hrdexdb")


def main() -> None:
    """Build the two-hand experiment (kept for reference; v4 pivoted to folds)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--v2", type=Path, default=V2)
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--f1-test-objects", type=int, default=4)
    parser.add_argument("--f1-val-objects", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--early-stop-score-loss", type=float, default=0.03)
    parser.add_argument("--early-stop-min-epochs", type=int, default=2)
    args = parser.parse_args()
    base, v3 = args.base, args.v3
    if (base / "manifest").exists():
        raise FileExistsError("prepared manifest already exists; use a new dataset root")
    raw = base / "raw"
    for name in ("inspire_dftp", "inspire_f1", "metadata"):
        if not (raw / name).exists():
            raise FileNotFoundError(f"{raw / name} is missing; run download_hrdexdb_v4.py first")

    dftp = dftp_records(args.v2, v3)
    object_split: dict[str, str] = {}
    for r in dftp:
        if object_split.setdefault(r["object_id"], r["split"]) != r["split"]:
            raise ValueError(f"DFTP object in two splits: {r['object_id']}")
    f1 = f1_records(base, object_split, args.f1_test_objects, args.f1_val_objects)

    force_cfg = load_config(v3 / "config/force.yaml", ForceConfig)
    source_cfg = load_config(v3 / "config/sources/hrdexdb.yaml", SourceConfig).model_copy(
        update={"root": str(raw.resolve()), "hrdexdb_hand_scale": {}}
    )
    adapter = HrdexdbAdapter(source_cfg)
    adapter.discover(raw)

    # Mechanical audit of the F1 half: it must load on the recorded video clock.
    episodes: dict[str, Episode] = {}
    for r in f1:
        if r["excluded_reasons"]:
            continue
        try:
            episodes[r["episode_id"]] = adapter.load(r["episode_id"])
        except Exception as exc:  # noqa: BLE001 - recorded as an exclusion
            r["excluded_reasons"].append(
                f"mechanical audit failed: {type(exc).__name__}: {exc}"[:200]
            )
    for r in dftp:
        episodes[r["episode_id"]] = adapter.load(r["episode_id"])
    kept = [r for r in dftp + f1 if not r["excluded_reasons"]]
    train = [r for r in kept if r["split"] == "train"]

    # Per-hand unit alignment, training trials only.
    per_hand = {
        hand: fit_consts(episodes, [r for r in train if r["hand"] == hand], force_cfg)
        for hand in ("inspire_dftp", "inspire_f1")
    }
    grasp = {h: c.arms["R"].grasp for h, c in per_hand.items()}
    scale = {"inspire_f1": round(grasp["inspire_dftp"] / grasp["inspire_f1"], 4)}
    source_cfg = source_cfg.model_copy(update={"hrdexdb_hand_scale": scale})
    adapter = HrdexdbAdapter(source_cfg)
    adapter.discover(raw)
    for r in kept:
        if r["hand"] == "inspire_f1":
            episodes[r["episode_id"]] = adapter.load(r["episode_id"])
    consts = fit_consts(episodes, train, force_cfg)

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
                **{k: r[k] for k in ("episode_id", "object_id", "success", "split", "hand")},
                "shared_object": r.get("shared_object"),
                "tactile_channels": ep.meta["tactile_channels"],
                "hand_scale": ep.meta["hand_scale"],
                "clock_alignment": ep.meta["clock_alignment"],
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
    # The DFTP windows must be exactly v3's so results stay comparable on that subset.
    for name in ["val", "test"]:
        v3_ids = {x.window_id for x in manifest.read_manifest(v3 / f"manifest/{name}.parquet")}
        v4_dftp = {
            x.window_id
            for x in manifest.read_manifest(directory / f"{name}.parquet")
            if x.session == "inspire_dftp"
        }
        if v3_ids != v4_dftp:
            raise ValueError(f"DFTP window ids differ from v3: {name}")
    holdout = HoldoutFile.model_validate(
        {
            "git_hash": version,
            "splits": {f"hrdexdb/{r['episode_id']}": r["split"] for r in dftp + f1},
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
        "scope": "v4: both Inspire hands; DFTP half identical to v3; F1 half added",
        "split_rule": (
            f"shared objects follow DFTP's frozen split; F1-only objects hash-ordered with salt "
            f"{SPLIT_SALT}: first {args.f1_test_objects} test, next {args.f1_val_objects} val"
        ),
        "per_hand_grasp_raw": grasp,
        "hand_scale": scale,
        "per_hand_normalization_before_scale": {
            h: c.model_dump(mode="json") for h, c in per_hand.items()
        },
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
        "kept_counts": {
            "all": counts(kept),
            "inspire_dftp": counts([r for r in kept if r["hand"] == "inspire_dftp"]),
            "inspire_f1": counts([r for r in kept if r["hand"] == "inspire_f1"]),
        },
        "excluded": [r for r in f1 if r["excluded_reasons"]],
        "manifest_hashes": holdout.manifests,
        "git_hash": version,
        "records": records,
    }
    (base / "representation-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ["hand_scale", "per_hand_grasp_raw", "kept_counts", "force_classifier"]
            },
            indent=2,
            default=str,
        )
    )
    print(f"excluded F1 trials: {len(summary['excluded'])}")


if __name__ == "__main__":
    main()
