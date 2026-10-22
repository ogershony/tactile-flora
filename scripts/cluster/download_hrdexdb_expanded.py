"""Download one camera and sensor clocks for the complete Inspire DFTP embodiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from scripts.download_phantom import contained_target

BASE = Path("data/acceptance/hrdexdb-expanded-v2")
REPO = "https://huggingface.co/datasets/HRDexDB/HRDexDB"
REVISION = "98b8f8ea2e361a3c7df38b5d8f3f8395913773f2"
FILES = (
    "grasp_result.json",
    "raw/hand/force.npy",
    "raw/hand/position.npy",
    "raw/hand/time.npy",
    "raw/timestamps/timestamp.npy",
    "raw/timestamps/frame_id.npy",
    "vid/22641005.mp4",
)


def freeze_splits(catalog: pd.DataFrame, pilot: Path) -> list[dict]:
    """Reserve unseen test objects and retain every historical train/validation object."""
    previous = {
        split: {
            row["object_id"]
            for row in json.loads((pilot / "manifest" / name).read_text())["episodes"]
        }
        for split, name in [("train", "train-sample.json"), ("val", "split.json")]
    }
    if previous["train"] & previous["val"]:
        raise ValueError("historical object splits overlap")
    data = catalog[catalog.embodiment == "inspire_dftp"].copy()
    if data.episode_id.duplicated().any() or data.success.isna().any():
        raise ValueError("duplicate or unlabeled Inspire DFTP catalog episodes")
    objects = set(data.object_id)
    if not (previous["train"] | previous["val"]) <= objects:
        raise ValueError("catalog no longer covers the historical objects")
    unseen = sorted(
        objects - previous["train"] - previous["val"],
        key=lambda obj: hashlib.sha256(f"hrdexdb-v2:20261021:{obj}".encode()).hexdigest(),
    )
    if len(unseen) < 32:
        raise ValueError("too few unseen objects to reserve independent development and test sets")
    test_objects = set(unseen[:16])
    val_objects = previous["val"] | set(unseen[16:24])
    rows = []
    for row in data.sort_values("episode_id").to_dict("records"):
        obj = row["object_id"]
        split = "test" if obj in test_objects else "val" if obj in val_objects else "train"
        rows.append(
            {
                "episode_id": row["episode_id"],
                "object_id": obj,
                "success": bool(row["success"]),
                "split": split,
            }
        )
    return rows


def fetch(root: Path, name: str) -> dict:
    """Stream a pinned file atomically; verify hashes before resuming completed downloads."""
    _, target = contained_target(root, name)
    receipt = target.with_name(target.name + ".sha256")
    if target.is_file() and receipt.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if receipt.read_text().strip() == digest:
            return {"path": name, "sha256": digest, "bytes": target.stat().st_size}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(7):
        try:
            with (
                urlopen(f"{REPO}/resolve/{REVISION}/{name}", timeout=120) as response,
                part.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            digest = hashlib.sha256(part.read_bytes()).hexdigest()
            part.replace(target)
            receipt.write_text(digest + "\n")
            return {"path": name, "sha256": digest, "bytes": target.stat().st_size}
        except OSError:
            if attempt == 6:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise AssertionError("unreachable")


def main() -> None:
    """Freeze object assignments before downloading or scoring new videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--pilot", type=Path, default=Path("data/acceptance/hrdexdb-pilot"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    raw = args.base / "raw"
    catalog_receipt = fetch(raw, "metadata/episodes.parquet")
    episodes = freeze_splits(pd.read_parquet(raw / "metadata/episodes.parquet"), args.pilot)
    plan = {
        "dataset": "HRDexDB/HRDexDB",
        "revision": REVISION,
        "scope": "all Inspire DFTP episodes; one fixed scene camera; source labels",
        "selection": "all episodes; hash-ordered unseen objects; 16 new test / 8 new val objects",
        "test_policy": (
            "no test inference until training settings are fixed; historical val remains val"
        ),
        "catalog": catalog_receipt,
        "files_per_episode": FILES,
        "episodes": episodes,
    }
    plan_path = args.base / "download-plan.json"
    canonical = json.dumps(plan, indent=2) + "\n"
    if plan_path.exists() and plan_path.read_text() != canonical:
        raise ValueError("existing download plan differs; refusing to change frozen splits")
    plan_path.write_text(canonical)
    names = [f"{row['episode_id']}/{name}" for row in episodes for name in FILES]
    receipts = [catalog_receipt]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, receipt in enumerate(pool.map(lambda name: fetch(raw, name), names), 1):
            receipts.append(receipt)
            if index % 70 == 0:
                print(f"downloaded {index}/{len(names)} files", flush=True)
    (args.base / "download-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"download complete: {len(episodes)} episodes; {sum(r['bytes'] for r in receipts)} bytes")


if __name__ == "__main__":
    main()
