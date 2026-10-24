"""Fetch RoboRewardBench (teetone/RoboReward test split) metadata, then only the 500 videos that
eval/roboreward.py's stratified sampler picks, and build the roboreward_500 manifest + sidecar.

The sampler draws from metadata alone; the probe below downloads each picked video just before
probing it, so nothing outside the sample is fetched. Revision pinned.
Usage: python scripts/v4/fetch_roboreward_500.py [--out data/raw/roboreward]
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from tactile_flora.eval import roboreward
from tactile_flora.verify import full_git_hash

REV = "469b9af76e8539e8d2ac553b081307738fd92ca5"
BASE = f"https://huggingface.co/datasets/teetone/RoboReward/resolve/{REV}/test"


def fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (resumable at file level, waits out 429 windows)."""
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(14):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.with_suffix(".part").write_bytes(r.read())
            dest.with_suffix(".part").replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            if k == 13:
                raise
            time.sleep(60 + 30 * min(k, 8) if "429" in str(e) else min(5 * 2**k, 120))


def main() -> None:
    """Build data/manifest/roboreward_500.parquet from the pinned test split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/roboreward")
    ap.add_argument("--manifest", default="data/manifest/roboreward_500.parquet")
    a = ap.parse_args()
    root = Path(a.out) / "test"
    meta = root / "metadata.jsonl"
    fetch(f"{BASE}/metadata.jsonl", meta)

    def probe(video: Path) -> tuple[float, int]:
        rel = video.resolve().relative_to(root.resolve())
        fetch(f"{BASE}/{rel.as_posix()}", video)
        return roboreward.probe_video(video)

    out = roboreward.make_roboreward(
        meta, Path(a.manifest), n=500, seed=0, report_version=full_git_hash(), probe=probe
    )
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
