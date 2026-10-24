"""Freeze the v4 HRDexDB object-level split (both Inspire hands) before any model is run on it.

Rule (fixed 2026-10-23 before any v4 evaluation): every object that has a labelled inspire_dftp
or inspire_f1 trial is ordered by sha256("tactile-flora-v4:20261023:" + object_id); objects are
taken in that order into TEST until it holds at least 35% of the labelled Inspire trials, then
into DEV until it holds at least 12%, and the rest are TRAIN. An object's trials of both hands
always share a split, so no test object is seen in training through the other hand. allegro_v5
(every trial a success) and human (no robot, no label) are not used. TEST is locked: it is scored
once per final model; DEV is used only to set operating thresholds and stacking weights.
Also writes the camera-study subset: 300 TRAIN trials stratified by hand x outcome (seed 0).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "tactile-flora-v4:20261023:"
HANDS = ("inspire_dftp", "inspire_f1")


def main(episodes: str, out: str) -> None:
    e = pd.read_parquet(episodes)
    e = e[e.embodiment.isin(HANDS) & e.success.notna()].copy()
    objs = sorted(set(e.object_id), key=lambda o: hashlib.sha256((SALT + o).encode()).hexdigest())
    per_obj = e.groupby("object_id").size()
    n = len(e)
    split: dict[str, str] = {}
    acc = {"test": 0, "dev": 0}
    for o in objs:
        if acc["test"] < 0.35 * n:
            s = "test"
        elif acc["dev"] < 0.12 * n:
            s = "dev"
        else:
            s = "train"
        if s in acc:
            acc[s] += int(per_obj[o])
        split[o] = s
    e["split"] = e.object_id.map(split)
    summary = (
        e.groupby(["split", "embodiment"])
        .agg(
            trials=("episode_id", "size"),
            objects=("object_id", "nunique"),
            success=("success", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())
    print(
        e.groupby("split")
        .agg(
            trials=("episode_id", "size"),
            objects=("object_id", "nunique"),
            failures=("success", lambda s: int((~s.astype(bool)).sum())),
        )
        .to_string()
    )
    rng = np.random.default_rng(0)
    study = []
    tr = e[e.split == "train"]
    for (hand, ok), g in tr.groupby(["embodiment", "success"]):
        k = int(round(300 * len(g) / len(tr)))
        study += sorted(
            rng.choice(g.episode_id.to_numpy(), size=min(k, len(g)), replace=False).tolist()
        )
    doc = {
        "rule": __doc__.strip().splitlines()[2:9],
        "salt": SALT,
        "objects": split,
        "episodes": {
            r.episode_id: {
                "split": r.split,
                "object_id": r.object_id,
                "hand": r.embodiment,
                "success": bool(r.success),
            }
            for r in e.itertuples()
        },
        "camera_study_train_subset": sorted(study),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        "camera-study subset",
        len(study),
        "->",
        out,
        hashlib.sha256(Path(out).read_bytes()).hexdigest()[:16],
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
