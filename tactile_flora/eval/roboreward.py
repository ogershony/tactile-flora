"""Build the roboreward_500 benchmark: a stratified, force-free outcome-only subsample.

Owns: make_roboreward and probe_video. Input is the ``test/metadata.jsonl`` of the
RoboReward HF dataset (teetone/RoboReward, test split = RoboRewardBench; rows have
``file_name`` (mp4 path relative to the metadata file's directory), ``task`` and ``reward``
1-5). Output is a manifest-like parquet ``data/manifest/roboreward_500.parquet`` (ManifestRow
fields; source "roboreward", split "benchmark", no force, no hashes) and the windows sidecar
``roboreward_500.windows.parquet`` that the harness reads instead of an adapter.
Must never: touch force, train on these videos, or use any column but file_name/task/reward.

Stratification: allocate ``n`` videos over the reward values 1-5 in proportion to their count
in the metadata (largest remainder, at least one per present value), then sample without
replacement within each stratum with ``random.Random(seed)`` over file-name-sorted rows. The
same inputs and seed give byte-identical outputs. Video frame count and fps come from the
container header via PyAV (``probe_video``; injectable for tests).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.eval.cache import CachedWindow, cache_path, write_windows_cache
from tactile_flora.labels.windows import outcome_frame_indices
from tactile_flora.schema import ManifestRow
from tactile_flora.sources.base import strip_outcome_words
from tactile_flora.verify.manifest import window_id

SOURCE = "roboreward"
LICENSE = "CC-BY-4.0"
Probe = Callable[[Path], tuple[float, int]]


def probe_video(path: Path) -> tuple[float, int]:
    """Return (fps, n_frames) of a video from its container header (counts packets if needed)."""
    import av

    c = av.open(str(path))
    try:
        s = c.streams.video[0]
        fps = float(s.average_rate or 0)
        n = int(s.frames or 0)
        if n == 0:
            n = sum(1 for _ in c.demux(s))
    finally:
        c.close()
    if fps <= 0 or n < 1:
        raise ValueError(f"cannot probe {path}")
    return fps, n


def allocate(counts: dict[int, int], n: int) -> dict[int, int]:
    """Largest-remainder allocation of ``n`` over strata, at least one per non-empty stratum."""
    total = sum(counts.values())
    if n >= total:
        return dict(counts)
    quota = {k: n * v / total for k, v in counts.items()}
    alloc = {k: max(1, min(counts[k], int(q))) for k, q in quota.items()}
    order = sorted(quota, key=lambda k: (quota[k] - int(quota[k]), -k), reverse=True)
    i = 0
    while sum(alloc.values()) < n:
        k = order[i % len(order)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
        i += 1
    while sum(alloc.values()) > n:  # the min-1 rule can overshoot on tiny n
        k = max(alloc, key=lambda j: alloc[j])
        alloc[k] -= 1
    return alloc


def make_roboreward(
    metadata: Path,
    out_manifest: Path,
    *,
    n: int = 500,
    seed: int = 0,
    report_version: str = "unknown",
    probe: Probe = probe_video,
) -> Path:
    """Write the stratified manifest and its windows sidecar; return the manifest path."""
    root = metadata.parent
    rows = [json.loads(line) for line in metadata.read_text("utf-8").splitlines() if line.strip()]
    by_reward: dict[int, list[dict[str, object]]] = {}
    for r in sorted(rows, key=lambda r: str(r["file_name"])):
        by_reward.setdefault(int(str(r["reward"])), []).append(r)
    alloc = allocate({k: len(v) for k, v in by_reward.items()}, n)
    rng = random.Random(seed)
    picked = [r for k in sorted(by_reward) for r in rng.sample(by_reward[k], alloc[k])]
    picked.sort(key=lambda r: str(r["file_name"]))

    mrows: list[ManifestRow] = []
    wins: list[CachedWindow] = []
    for r in picked:
        fname = str(r["file_name"])
        video = (root / fname).resolve()
        fps, nf = probe(video)
        idx = outcome_frame_indices(nf, fps)
        wid = window_id(SOURCE, fname, "outcome", 0.0)
        score = int(str(r["reward"]))
        mrows.append(
            ManifestRow(
                window_id=wid, source=SOURCE, episode_id=fname, session=fname, split="benchmark",
                prompt_kind="outcome", anchor_s=0.0, frame_hashes=[], norm_version="none",
                label_origin="source", license=LICENSE, report_version=report_version,
            )
        )  # fmt: skip
        wins.append(
            CachedWindow(
                window_id=wid, source=SOURCE, episode_id=fname, session=fname,
                task=strip_outcome_words(str(r["task"])), prompt_kind="outcome", anchor_s=0.0,
                frame_times=[i / fps for i in idx], target=f"Score: {score}",
                video_path=str(video), fps=fps, n_frames=nf, license=LICENSE,
            )
        )  # fmt: skip
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            (f, pa.list_(pa.string()) if f == "frame_hashes" else
             pa.float64() if f in ("anchor_s", "shift_s") else pa.string())
            for f in ManifestRow.model_fields
        ]
    )  # fmt: skip
    cols = {f: [getattr(m, f) for m in mrows] for f in ManifestRow.model_fields}
    pq.write_table(pa.table(cols, schema=schema), out_manifest)
    write_windows_cache(cache_path(out_manifest), wins)
    return out_manifest
