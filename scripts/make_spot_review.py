"""Create a local review packet with existing clips, complete targets and force lines.

This does not record a human verdict or change the verification gates. It reconstructs
exactly the unshifted windows used for spot checks and verifies every indexed file matches.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.force import normalize
from tactile_flora.sources.base import make_adapter
from tactile_flora.verify import manifest, report
from tactile_flora.verify.windowing import build_source_inputs, episode_windows


def run(config: Path, force_config: Path, data_root: Path) -> Path:
    """Write review.html and review.json beside the indexed spot-check media."""
    cfg = load_config(config, SourceConfig)
    fcfg = load_config(force_config, ForceConfig)
    vdir = data_root / "verified" / cfg.name
    spot = vdir / "spot"
    index = json.loads((spot / "index.json").read_text())
    consts = normalize.load(vdir / "norm.json")
    results = report.read_checks(vdir)
    adapter = make_adapter(cfg.name, cfg)
    ids = sorted(adapter.discover(Path(cfg.root)))
    splits = adapter.splits()
    inputs = build_source_inputs(adapter, fcfg, consts, vdir, ids)
    indexed = {name.split("_", 1)[1]: name for name in index["files"]}
    reviews = []
    sections: list[str] = []
    for eid in ids:
        ep = adapter.load(eid)
        for window in episode_windows(
            ep, fcfg, consts, manifest.assign_split(ep, splits), inputs=inputs
        ):
            suffix = f"{eid.replace('/', '_')}_{window.prompt_kind}_{window.anchor_s:.2f}.png"
            if suffix not in indexed:
                continue
            name = indexed.pop(suffix)
            video = Path(name).with_suffix(".mp4").name
            row = {
                "file": name,
                "video": video if video in index.get("videos", []) else None,
                "episode": eid,
                "task": ep.task,
                "window": window.model_dump(mode="json"),
                "checks": {n: r.model_dump() for n, r in results[eid].items()},
                "human_verdict": None,
            }
            reviews.append(row)
            e = html.escape
            media = f'<video controls preload="metadata" width="640" src="{e(video)}"></video>'
            if row["video"] is None:
                media = "<p>Video missing: this window cannot be fully reviewed.</p>"
            failed = "; ".join(f"{n}: {r.detail}" for n, r in results[eid].items() if not r.passed)
            sections.append(
                f'<article id="{e(name)}"><h2>{e(name)}</h2><p>{e(ep.task)}</p>{media}'
                f'<p><img src="{e(name)}" width="768" alt="Sampled frames and label strip"></p>'
                f"<h3>Complete target</h3><pre>{e(window.target)}</pre>"
                f"<p>Current failed checks: {e(failed or 'none')}</p>"
                f"<details><summary>Original frame times (seconds)</summary>"
                f"<pre>{e(json.dumps(window.frame_times))}</pre></details>"
                f"<details><summary>Force lines supplied to training</summary>"
                f"<pre>{e(chr(10).join(window.force_lines))}</pre></details></article>"
            )
    if indexed:
        raise RuntimeError(f"indexed spot windows do not match current inputs: {sorted(indexed)}")
    if len(reviews) != index["n_windows"]:
        raise RuntimeError("spot review count does not match the index")
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Dataset spot review</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;"
        "line-height:1.5}article{border-top:1px solid #aaa;margin-top:32px;padding-top:16px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:12px}"
        "video,img{max-width:100%;height:auto}details{margin:12px 0}</style>"
        f"<h1>{html.escape(cfg.name)}: {len(reviews)} windows for human review</h1>"
        "<p>Review each clip against its complete target for label correctness and timing. "
        "Record pass/fail and a short reason per window. "
        "No verdict has been recorded by this tool.</p>"
        "<p>Clips replay the sampled frames at 4 fps; consult the original frame times below, "
        "especially for outcome windows. "
        "A readable clip alone does not establish label correctness.</p>"
        + "".join(sections)
        + "</html>"
    )
    (spot / "review.html").write_text(document)
    (spot / "review.json").write_text(
        json.dumps(
            {
                "source": cfg.name,
                "norm_version": consts.version(),
                "human_signoff": False,
                "windows": reviews,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(reviews)} windows: {spot / 'review.html'}")
    return spot / "review.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-config", type=Path, default=Path("configs/force.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    run(args.config, args.force_config, args.data_root)
