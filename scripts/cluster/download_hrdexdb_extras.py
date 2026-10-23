"""Download HRDexDB extras for the Inspire DFTP trials: arm joint torque and object 6D pose.

Both are small (about 170 KB of torque + time per trial, about 200 KB of pose per trial) and
land under ``data/acceptance/hrdexdb-extras/`` so the frozen v2 download stays untouched:

    hrdexdb-extras/inspire_dftp/<object>/<id>/raw/arm/{torque,time}.npy
    hrdexdb-extras/object_6d_pose_v2/inspire_dftp/<object>_<id>.npz

Run from the repository root: ``.venv/bin/python scripts/cluster/download_hrdexdb_extras.py``
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from scripts.cluster.download_hrdexdb_v4 import _token, fetch

OUT = Path("data/acceptance/hrdexdb-extras")
CATALOG = Path("data/acceptance/hrdexdb-expanded-v2/raw/metadata/episodes.parquet")


def main() -> None:
    """Fetch torque, arm time and object pose for every labeled DFTP trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-torque", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    token = _token()
    catalog = pd.read_parquet(CATALOG)
    rows = catalog[(catalog.embodiment == "inspire_dftp") & catalog.success.notna()]
    names: list[str] = []
    for r in rows.itertuples():
        if not args.skip_torque:
            names += [f"{r.episode_id}/raw/arm/torque.npy", f"{r.episode_id}/raw/arm/time.npy"]
        if not args.skip_pose and r.has_object_pose_v2:
            obj, seq = r.episode_id.split("/")[1:]
            names.append(f"object_6d_pose_v2/inspire_dftp/{obj}_{seq}.npz")

    def one(name: str) -> dict:
        try:
            return fetch(args.out, name, token)
        except Exception as exc:  # noqa: BLE001 - recorded per file
            return {"path": name, "error": f"{type(exc).__name__}: {exc}"[:200]}

    receipts: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, receipt in enumerate(pool.map(one, names), 1):
            receipts.append(receipt)
            if index % 200 == 0 or index == len(names):
                print(f"{index}/{len(names)} files", flush=True)
    failures = [r for r in receipts if "error" in r]
    (args.out / "download-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"done: {len(names)} files, {len(failures)} failed", flush=True)


if __name__ == "__main__":
    main()
