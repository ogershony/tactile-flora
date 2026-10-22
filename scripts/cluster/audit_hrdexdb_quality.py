"""Audit pilot label plumbing and create a blind full-video human review packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path("data/acceptance/hrdexdb-pilot")


def inspect_video(path: Path, strip: Path) -> dict:
    """Read real presentation times for eight overview frames; retain the full clip link."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
        sheet = Image.new("RGB", (4 * 320, 2 * 264), "white")
        draw = ImageDraw.Draw(sheet)
        times = []
        for i, seconds in enumerate(np.linspace(0, max(0, duration - 0.15), 8)):
            target = int(seconds / stream.time_base)
            container.seek(target, stream=stream)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target:
                    times.append(float(frame.pts * stream.time_base))
                    x, y = (i % 4) * 320, (i // 4) * 264
                    sheet.paste(frame.to_image().resize((320, 240)), (x, y + 24))
                    draw.text((x + 4, y + 4), f"{times[-1]:.3f} s", fill="black")
                    break
            else:
                raise ValueError(f"Cannot decode {path} at {seconds:.3f}s")
        sheet.save(strip)
        return {"duration_s": duration, "frames": stream.frames, "sample_times_s": times}


def main() -> None:
    """Write reproducible mechanical checks and unfilled human annotations separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--output", type=Path, default=Path("data/acceptance/hrdexdb-resume/quality")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(args.base / "raw/metadata/episodes.parquet").set_index("episode_id")
    entries, hashes = [], {}
    for split, name in [("train", "train-sample.json"), ("val", "split.json")]:
        path = args.base / "manifest" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in json.loads(path.read_text())["episodes"]:
            entries.append(dict(row, split=split))
    # Stable pseudo-random order breaks the original failure/success pairing.
    entries.sort(key=lambda r: hashlib.sha256(r["episode_id"].encode()).hexdigest())
    reviews, sections = [], []
    for index, row in enumerate(entries):
        eid = row["episode_id"]
        trial = args.base / "raw" / eid
        result = json.loads((trial / "grasp_result.json").read_text())
        path = trial / "vid" / f"{row['camera_id']}.mp4"
        strip = args.output / f"clip-{index:02d}.jpg"
        video = inspect_video(path, strip)
        force = np.load(trial / "raw/hand/force.npy")
        position = np.load(trial / "raw/hand/position.npy")
        time = np.load(trial / "raw/hand/time.npy").reshape(-1)
        dt = np.diff(time)
        raw_label = result.get("grasp_success")
        catalog_label = catalog.loc[eid, "success"]
        review = {
            "review_id": f"clip-{index:02d}",
            "episode_id": eid,
            "split": row["split"],
            "manifest_label": row["success"],
            "raw_label": raw_label,
            "catalog_label": bool(catalog_label) if pd.notna(catalog_label) else None,
            "labels_agree": bool(raw_label == row["success"] == catalog_label),
            "video": str(path),
            "overview": strip.name,
            **video,
            "force_shape": list(force.shape),
            "position_shape": list(position.shape),
            "sensor_finite": bool(np.isfinite(force).all() and np.isfinite(position).all()),
            "sensor_lengths_match": len(force) == len(position) == len(time),
            "timestamps_increasing": bool((dt > 0).all()),
            "sensor_span_s": float(time[-1] - time[0]),
            "median_sample_interval_s": float(np.median(dt)),
            "max_sample_gap_s": float(dt.max()),
            "video_minus_sensor_span_s": float(video["duration_s"] - (time[-1] - time[0])),
            "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        reviews.append(review)
        e = html.escape
        relative = e(os.path.relpath(path.resolve(), args.output.resolve()))
        sections.append(
            f'<article id="clip-{index:02d}"><h2>Clip {index:02d}</h2>'
            f"<p>{video['duration_s']:.2f} seconds. Watch the complete video before judging.</p>"
            f'<video controls preload="none" width="960" src="{relative}"></video>'
            f'<p><img loading="lazy" width="960" src="{strip.name}" alt="Eight overview frames">'
            "</p><p>Record stable lift / failed or unstable / cannot determine; lift and loss "
            "times; occlusion; human intervention; and a short reason in annotations.csv.</p>"
            "<details><summary>Reveal source label after your independent judgment</summary>"
            f"<p>{e(eid)} ({e(row['split'])}): {raw_label}</p></details></article>"
        )
        print(f"audited {eid}", flush=True)
    objects = {
        s: {r["episode_id"].split("/")[1] for r in reviews if r["split"] == s}
        for s in ["train", "val"]
    }
    digests = [r["video_sha256"] for r in reviews]
    report = {
        "scope": "mechanical checks plus eight-frame overview; no human signoff or relabeling",
        "manifest_sha256": hashes,
        "n_episodes": len(reviews),
        "label_disagreements": [r["episode_id"] for r in reviews if not r["labels_agree"]],
        "train_val_object_overlap": sorted(objects["train"] & objects["val"]),
        "duplicate_video_groups": [
            [r["episode_id"] for r in reviews if r["video_sha256"] == h]
            for h in sorted(set(digests))
            if digests.count(h) > 1
        ],
        "reviews": reviews,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    # Preserve any annotations a reviewer has already entered.
    annotations = args.output / "annotations.csv"
    if not annotations.exists():
        with annotations.open("w", newline="") as handle:
            fields = [
                "review_id",
                "verdict",
                "lift_start_s",
                "loss_or_placement_s",
                "occlusion",
                "human_intervention",
                "reason",
                "reviewer",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({"review_id": r["review_id"]} for r in reviews)
    (args.output / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>HRDexDB quality review</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;line-height:1.5}"
        "article{border-top:1px solid #aaa;margin-top:30px}img,video{max-width:100%}</style>"
        f"<h1>HRDexDB: {len(reviews)} clips for independent review</h1>"
        "<p>Judge whether the robot establishes and maintains a stable grasp during the lift. "
        "Record brief lift, tipping, dragging, slip and human assistance separately. "
        "Intentional placement after stable lifting can be valid. If the camera cannot show "
        "stability, choose cannot determine. The overview alone may miss the decisive event.</p>"
        "<p>Labels are hidden initially. Enter judgments in annotations.csv; this page does "
        "not save judgments. audit.json maps review IDs to source data. Nothing here changes "
        "source labels or signs off the dataset.</p>" + "".join(sections) + "</html>"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "reviews"}, indent=2))


if __name__ == "__main__":
    main()
