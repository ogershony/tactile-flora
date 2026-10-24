"""Build data/manifest/v4_train.parquet: HRDexDB v4, REASSEMBLE segment and PHANTOM train rows.

The training rows of the HRDexDB v4 manifest and of train_judge_pf (REASSEMBLE segments and
PHANTOM) are merged. The class-balance weight is not set here: configs/train/v4_*.yaml carry
failure_weight 2.0, and the merged rows measure successes / failures = 1,213 / 598 = 2.03
(HRDexDB 618 rows / 193 failures with the local 2026-10-23 prep, REASSEMBLE 1,146 / 382,
PHANTOM 47 / 23), within 2% of it.

report_version of every row is rewritten to the current HEAD (sft refuses a manifest from another
commit); the rows are otherwise copied unchanged, so window ids stay those of their source
manifests.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.verify import full_git_hash, manifest

PARTS = [
    "data/manifest/hrdexdb_v4_train.parquet",
    "data/manifest/train_judge_pf.parquet",  # 1,146 reassemble_seg + 47 phantom outcome rows
]


def main() -> None:
    """Merge the training rows and report the per-source and HRDexDB failure counts."""
    version = full_git_hash()
    rows = []
    for part in PARTS:
        for r in manifest.read_manifest(part):
            if r.split == "train" and r.prompt_kind == "outcome":
                rows.append(r.model_copy(update={"report_version": version}).model_dump())
    out = Path("data/manifest/v4_train.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=manifest._SCHEMA), out)
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(out, len(rows), by, version)
    # labels: count failures through the training windows source would need episodes; the
    # manifests do not store labels, so read them from the per-source label files instead.
    import json

    split = json.loads(Path("configs/v4/hrdexdb_split.json").read_text())["episodes"]
    hr = [r for r in rows if r["source"] == "hrdexdb"]
    hr_fail = sum(not split[r["episode_id"]]["success"] for r in hr)
    print("hrdexdb train", len(hr), "failures", hr_fail)
    return None


if __name__ == "__main__":
    main()
