"""Prepare HRDexDB v4: both Inspire hands, the frozen v4 object split, per-hand normalization,
the force-only classifier and outcome manifests. Labels are read only from grasp_result.json.

Steps (none of them looks at a TEST label):
1. Load every labelled inspire_dftp / inspire_f1 trial of configs/v4/hrdexdb_split.json with the
   tared representation and recorded video timestamps (sources/hrdexdb.py). A trial whose sensor
   or video clock cannot be established raises and is excluded; the rule never reads the label,
   and the excluded ids and reasons are written to the audit.
2. Fit normalization constants on TRAIN trials only, one group per hand (normalize.norm_group).
3. Fit the force-only classifier (force/classifier.py fit_cv, its default hyperparameters) on
   TRAIN trials, folds grouped by OBJECT, so every training trial carries an out-of-fold score.
   Report out-of-fold metrics on TRAIN and full-model metrics on DEV; TEST is not scored.
4. Write manifests train / val (= dev) / test of outcome windows, the normalization constants,
   the scorer and an audit json.

Usage: python scripts/v4/prepare_hrdexdb_v4.py --raw data/raw/hrdexdb --frames data/frames/hrdexdb
       --out data/v4/hrdexdb [--cameras 22641005]
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

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import classifier, normalize
from tactile_flora.force.resample import resample
from tactile_flora.labels.windows import outcome_frame_indices
from tactile_flora.sources.hrdexdb import HrdexdbAdapter
from tactile_flora.verify import full_git_hash, manifest
from tactile_flora.verify.windowing import episode_windows

SPLIT_NAME = {"train": "train", "dev": "val", "test": "test"}
TASK = "Grasp the {object}, lift it off the table and hold it in the air for 10 seconds."


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

    return float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/hrdexdb")
    ap.add_argument("--frames", default="data/frames/hrdexdb")
    ap.add_argument("--split", default="configs/v4/hrdexdb_split.json")
    ap.add_argument("--out", default="data/v4/hrdexdb")
    ap.add_argument("--cameras", default="22641005")
    ap.add_argument("--force-config", default="configs/force.code.yaml")
    ap.add_argument("--contact-min-frac", type=float, default=0.05)
    ap.add_argument("--tare-s", type=float, default=1.0)
    ap.add_argument("--report-version", default=None, help="manifest report_version (default HEAD)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(a.split).read_text())
    src_cfg = SourceConfig(
        name="hrdexdb",
        root=str(Path(a.raw).resolve()),
        license_allowed=["CC-BY-NC-4.0"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=0.3,
        split_rules={"by": "object", "note": "configs/v4/hrdexdb_split.json (frozen 2026-10-23)"},
        seed=0,
        hrdexdb_use_video_timestamps=True,
        hrdexdb_force_representation="tared",
        hrdexdb_tare_s=a.tare_s,
        hrdexdb_cameras=a.cameras.split(","),
        hrdexdb_frames_root=str(Path(a.frames).resolve()),
        hrdexdb_task_template=TASK,
    )
    base = load_config(a.force_config, ForceConfig)
    fcfg = base.model_copy(
        update={
            "resample": base.resample.model_copy(update={"contact_min_frac": a.contact_min_frac}),
            "render": base.render.model_copy(update={"summary": True, "frame_lines": False}),
        }
    )
    adapter = HrdexdbAdapter(src_cfg)
    adapter.discover(Path(a.raw))
    eps: dict[str, Any] = {}
    excluded: dict[str, str] = {}
    for ep_id, info in sorted(split["episodes"].items()):
        try:
            ep = adapter.load(ep_id)
            if ep.labels and ep.labels[0].outcome != (5 if info["success"] else 1):
                raise ValueError("grasp_result.json disagrees with the catalog label")
            eps[ep_id] = ep
        except Exception as e:  # noqa: BLE001 - recorded, never silently dropped
            excluded[ep_id] = f"{type(e).__name__}: {str(e)[:200]}"
    info_of = split["episodes"]
    kept = {s: [e for e in eps if info_of[e]["split"] == s] for s in ("train", "dev", "test")}
    print({s: len(v) for s, v in kept.items()}, "excluded", len(excluded), flush=True)
    # 2) per-hand normalization on TRAIN only
    fit_items = []
    for e in kept["train"]:
        ep = eps[e]
        times = np.arange(0, ep.duration_s, fcfg.resample.frame_period_s).tolist()
        fit_items.append(
            (normalize.norm_group(ep), resample(ep.force, ep.gripper, times, fcfg.resample))
        )
    consts = normalize.fit(fit_items, fcfg.resample, source="hrdexdb")
    (out / "verified/hrdexdb").mkdir(parents=True, exist_ok=True)
    normalize.save(consts, out / "verified/hrdexdb/norm.json")
    # 3) classifier on TRAIN (object-grouped folds), outcome-window features
    feats: dict[str, np.ndarray] = {}
    for e, ep in eps.items():
        cam = ep.cameras["scene"]
        times = [i / cam.fps for i in outcome_frame_indices(cam.n_frames, cam.fps)]
        f = normalize.apply(
            resample(ep.force, ep.gripper, times, fcfg.resample),
            consts,
            fcfg.render,
            group=normalize.norm_group(ep),
        )
        feats[e] = classifier.extract_features(f)
    tr = kept["train"]
    X = np.stack([feats[e] for e in tr])
    y = np.asarray([int(info_of[e]["success"]) for e in tr])
    scorer = classifier.fit_cv(X, y, [info_of[e]["object_id"] for e in tr], tr, source="hrdexdb")
    scorer_path = out / "verified/hrdexdb/force_classifier.json"
    scorer.save(scorer_path)
    dev = kept["dev"]
    report: dict[str, Any] = {"oof_train": scorer.cv_metrics, "dev": {"n": len(dev)}}
    y_dev = np.asarray([int(info_of[e]["success"]) for e in dev])
    # The first Colab prep runs before DEV/TEST are rendered (their trials do not load yet); the
    # DEV report then comes from the second prep, whose TRAIN fits are identical.
    if dev and 0 < y_dev.sum() < len(dev):
        p_dev = np.asarray([scorer.model.predict_proba(feats[e]) for e in dev])
        report["dev"].update(
            {
                "auroc_failure": auroc(1 - y_dev, 1 - p_dev),
                "accuracy@0.5": float(np.mean((p_dev >= 0.5) == y_dev)),
            }
        )
        for hand in ("inspire_dftp", "inspire_f1"):
            idx = [i for i, e in enumerate(dev) if info_of[e]["hand"] == hand]
            if idx and 0 < y_dev[idx].sum() < len(idx):
                report["dev"][f"auroc_failure_{hand}"] = auroc(1 - y_dev[idx], 1 - p_dev[idx])
    print(json.dumps(report, indent=1), flush=True)
    # 4) manifests
    version = a.report_version or full_git_hash()
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s, ids in kept.items():
        for e in ids:
            ep = eps[e]
            for w in episode_windows(ep, fcfg, consts, SPLIT_NAME[s], with_force=False):
                if w.prompt_kind != "outcome":
                    continue
                rows[SPLIT_NAME[s]].append(
                    manifest.ManifestRow(
                        window_id=manifest.window_id(
                            ep.source, ep.episode_id, w.prompt_kind, w.anchor_s
                        ),
                        source=ep.source,
                        episode_id=ep.episode_id,
                        session=ep.session,
                        split=SPLIT_NAME[s],
                        prompt_kind=w.prompt_kind,
                        anchor_s=w.anchor_s,
                        frame_hashes=[],
                        norm_version=consts.version(),
                        label_origin="source",
                        license=ep.license,
                        report_version=version,
                        shift_s=0.0,
                        norm_group=normalize.norm_group(ep),
                    ).model_dump()
                )
    (out / "manifest").mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, r in rows.items():
        path = out / "manifest" / f"hrdexdb_v4_{name}.parquet"
        pq.write_table(pa.Table.from_pylist(r, schema=manifest._SCHEMA), path)
        hashes[name] = manifest.manifest_hash(path)
    print({k: len(v) for k, v in rows.items()}, flush=True)
    audit = {
        "split_sha256": hashlib.sha256(Path(a.split).read_bytes()).hexdigest(),
        "source_config": src_cfg.model_dump(),
        "force_config": fcfg.model_dump(mode="json"),
        "normalization": {**consts.model_dump(mode="json"), "norm_version": consts.version()},
        "classifier": {
            "path": str(scorer_path),
            "sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
            "params": scorer.params,
            **report,
        },
        "kept": {s: len(v) for s, v in kept.items()},
        "excluded": excluded,
        # per-trial covariates for the pre-registered coverage sensitivity analysis (docs/v4)
        "trials": {
            e: {
                "split": info_of[e]["split"],
                "hand": ep.session,
                "duration_s": float(ep.duration_s),
                "frame_ids_beyond_video": int(
                    ep.meta.get("clock_alignment", {}).get("frame_ids_beyond_video", 0) or 0
                ),
            }
            for e, ep in eps.items()
        },
        "manifests": hashes,
        "report_version": version,
    }
    (out / "hrdexdb_v4_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
