"""Download the Inspire F1 embodiment of HRDexDB next to the frozen Inspire DFTP download.

v4 trains on both Inspire hands. The DFTP half is the frozen v2 download (pinned revision,
receipts, splits); this script adds the F1 half under ``<base>/raw/inspire_f1`` and links the
DFTP trials and the catalog into ``<base>/raw`` so one source root holds both hands.

Run from the repository root (login node, CPU only):

    .venv/bin/python scripts/cluster/download_hrdexdb_v4.py --workers 8

Sensor files (about 1 MB per trial) are fetched before the videos (15-30 MB per trial) so the
force-signal audit can start while the videos are still arriving. A Hugging Face token from
``huggingface_hub.get_token()`` is sent when present; the pinned revision and the per-file
sha256 receipts make the download reproducible either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

REPO = "https://huggingface.co/datasets/HRDexDB/HRDexDB"
REVISION = "98b8f8ea2e361a3c7df38b5d8f3f8395913773f2"  # same pin as the v2 download
BASE = Path("data/acceptance/hrdexdb-expanded-v4")
V2_RAW = Path("data/acceptance/hrdexdb-expanded-v2/raw")
EMBODIMENT = "inspire_f1"
SENSOR_FILES = (
    "grasp_result.json",
    "raw/hand/right_tactile.npy",
    "raw/hand/right_tactile_time.npy",
    "raw/hand/right_joint_states.npy",
    "raw/hand/right_joint_states_time.npy",
    "raw/timestamps/timestamp.npy",
    "raw/timestamps/frame_id.npy",
)
VIDEO_FILES = ("vid/22641005.mp4",)  # the scene camera used for every DFTP trial


def _token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 - the token is optional
        return os.environ.get("HF_TOKEN")


def fetch(root: Path, name: str, token: str | None) -> dict:
    """Stream one pinned file atomically; reuse a completed download whose receipt matches."""
    target = (root / name).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"path escapes the download root: {name}")
    receipt = target.with_name(target.name + ".sha256")
    if target.is_file() and receipt.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if receipt.read_text().strip() == digest:
            return {"path": name, "sha256": digest, "bytes": target.stat().st_size}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(7):
        try:
            request = Request(f"{REPO}/resolve/{REVISION}/{name}", headers=headers)
            with urlopen(request, timeout=120) as response, part.open("wb") as handle:
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


def link_dftp(raw: Path) -> None:
    """Expose the frozen DFTP trials and the catalog inside the v4 source root."""
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("inspire_dftp", "metadata"):
        link = raw / name
        source = (V2_RAW / name).resolve()
        if not source.exists():
            raise FileNotFoundError(f"v2 download is missing {source}")
        if link.is_symlink() or link.exists():
            if link.resolve() != source:
                raise FileExistsError(f"{link} exists and does not point at {source}")
            continue
        link.symlink_to(source)


def main() -> None:
    """Link the DFTP half, then fetch every labeled F1 trial (sensors first, videos second)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--sensors-only", action="store_true")
    args = parser.parse_args()
    raw = args.base / "raw"
    link_dftp(raw)
    token = _token()
    catalog = pd.read_parquet(raw / "metadata/episodes.parquet")
    rows = catalog[(catalog.embodiment == EMBODIMENT) & catalog.success.notna()]
    rows = rows[rows.has_grasp_result & rows.has_tactile & rows.has_videos]
    episodes = [
        {"episode_id": r.episode_id, "object_id": r.object_id, "success": bool(r.success)}
        for r in rows.sort_values("episode_id").itertuples()
    ]
    plan = {
        "dataset": "HRDexDB/HRDexDB",
        "revision": REVISION,
        "scope": f"all labeled {EMBODIMENT} episodes; one fixed scene camera; source labels",
        "dftp_half": str(V2_RAW),
        "sensor_files": SENSOR_FILES,
        "video_files": VIDEO_FILES,
        "authenticated": bool(token),
        "episodes": episodes,
    }
    plan_path = args.base / "download-plan.json"
    canonical = json.dumps(plan, indent=2) + "\n"
    if plan_path.exists() and plan_path.read_text() != canonical:
        raise ValueError("existing v4 download plan differs; refusing to change it")
    plan_path.write_text(canonical)
    groups = [SENSOR_FILES] if args.sensors_only else [SENSOR_FILES, VIDEO_FILES]
    receipts: list[dict] = []
    failures: list[dict] = []
    for files in groups:
        names = [f"{row['episode_id']}/{name}" for row in episodes for name in files]

        def one(name: str) -> dict:
            try:
                return fetch(raw, name, token)
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal per file
                return {"path": name, "error": f"{type(exc).__name__}: {exc}"[:200]}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, receipt in enumerate(pool.map(one, names), 1):
                (failures if "error" in receipt else receipts).append(receipt)
                if index % 100 == 0 or index == len(names):
                    print(f"{files[0].split('/')[0]}: {index}/{len(names)} files", flush=True)
    (args.base / "download-receipts.json").write_text(
        json.dumps({"receipts": receipts, "failures": failures}, indent=2) + "\n"
    )
    print(
        f"download complete: {len(episodes)} episodes; {sum(r['bytes'] for r in receipts)} bytes;"
        f" {len(failures)} failed files",
        flush=True,
    )


if __name__ == "__main__":
    main()
