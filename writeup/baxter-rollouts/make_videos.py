"""Render side-by-side clips of the Baxter rollouts: fine-tuned FLORA (video only) vs tactile.

Same layout as ``writeup/hrdexdb/make_videos.py`` for HRDexDB: the head camera shown twice
(one panel per model, each with its expected score and its rank among the ten clips) above the
normalized fingertip force the tactile model read. No model crosses the 4.0 threshold on these clips, so a
panel is green when that model ranks the clip on the correct side of the other class (a success
above every failure, a failure below both successes) and red otherwise.

Rendered: every clip where the tactile adapter (real force) is further on the correct side than
the video-only control by at least ``--min-margin`` (0.1), sorted by that margin; ``--all``
renders all ten. Run from the repo root:

    .venv/bin/python writeup/baxter-rollouts/make_videos.py
    .venv/bin/python writeup/baxter-rollouts/make_videos.py --limit 1

Inputs are the finished cells under data/acceptance/baxter-rollouts-v1/eval and the dataset copy
under data/acceptance/baxter-rollouts-v1/raw; nothing is recomputed with the model.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tactile_flora.config import SourceConfig, load_config  # noqa: E402
from tactile_flora.eval.metrics import SUCCESS_THRESHOLD  # noqa: E402
from tactile_flora.labels.windows import outcome_frame_indices  # noqa: E402
from tactile_flora.sources.baxter_rollouts import BaxterRolloutsAdapter  # noqa: E402
from tactile_flora.video.readers import PyAVReader  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/acceptance/baxter-rollouts-v1"
OUT = ROOT / "writeup/baxter-rollouts"
FORCE_PRED = BASE / "eval/expanded-v3-force/baxter_rollouts/real/predictions.parquet"
CTRL_PRED = BASE / "eval/expanded-v2-control/baxter_rollouts/omitted/predictions.parquet"
OUT_FPS = 15
W, H = 1280, 720
ARM = "R"


def _predictions() -> pd.DataFrame:
    f = pd.read_parquet(FORCE_PRED).set_index("episode_id")
    c = pd.read_parquet(CTRL_PRED).set_index("episode_id").loc[f.index]
    audit = json.loads((BASE / "representation-audit.json").read_text())
    truth = {r["episode_id"]: int(r["outcome"] == 5) for r in audit["records"]}
    df = pd.DataFrame(
        {
            "truth": pd.Series(truth).loc[f.index],
            "force_E": f["expected_score"],
            "ctrl_E": c["expected_score"],
            "duration_s": pd.Series(
                {r["episode_id"]: r["duration_s"] for r in audit["records"]}
            ).loc[f.index],
        }
    )
    for col in ("force", "ctrl"):
        e = df[f"{col}_E"]
        df[f"{col}_pred"] = (e >= SUCCESS_THRESHOLD).astype(int)
        df[f"{col}_rank"] = e.rank(ascending=False, method="min").astype(int)
        succ, fail = e[df.truth == 1], e[df.truth == 0]
        df[f"{col}_ranked_ok"] = np.where(df.truth == 1, e > fail.max(), e < succ.min())
    sign = np.where(df["truth"] == 1, 1.0, -1.0)
    df["margin"] = (df["force_E"] - df["ctrl_E"]) * sign
    return df


def _label(pred: int) -> str:
    return "SUCCESS" if pred else "FAILURE"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _force_stats(total: np.ndarray, t: np.ndarray, tau_n: float) -> dict:
    contact = total > tau_n
    hold_s = float(np.trapezoid(contact.astype(float), t)) if len(t) > 1 else 0.0
    return {
        "peak_force_norm": round(float(total.max()), 2),
        "hold_s": round(hold_s, 2),
        "last_contact_s": round(float(t[contact][-1]), 2) if contact.any() else None,
        "clip_s": round(float(t[-1]), 1),
    }


def render(
    episode_id: str,
    row: pd.Series,
    adapter: BaxterRolloutsAdapter,
    reader: PyAVReader,
    norm: dict,
    out_dir: Path,
) -> dict:
    """Write the mp4 and poster for one rollout and return its row for clips.json."""
    ep = adapter.load(episode_id)
    cam = ep.cameras["scene"]
    arm = ep.force.arms[ARM]
    assert arm.kind == "pad"
    assert ep.gripper is not None
    t = np.asarray(ep.force.t)
    scale, baseline = float(norm["scale"]), float(norm["baseline"])
    tau_n = (float(norm["tau"]) - baseline) / scale
    # Each pad tared by its own share of the resting baseline, in units of the median grasp.
    fa = (np.asarray(arm.f_a) - baseline / 2) / scale
    fb = (np.asarray(arm.f_b) - baseline / 2) / scale
    total = np.clip(fa + fb, 0.0, None)
    gt_, grip = np.asarray(ep.gripper.t), np.asarray(ep.gripper.arms[ARM])
    seen = np.asarray(outcome_frame_indices(cam.n_frames, cam.fps)) / cam.fps
    stats = _force_stats(total, t, tau_n)

    truth_word = _label(int(row.truth))
    stem = f"{episode_id}_{truth_word.lower()}"
    mp4, png = out_dir / f"{stem}.mp4", out_dir / f"{stem}.png"
    red, green = "#c0392b", "#1e8449"

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(
        0.5,
        0.975,
        f"Ground truth: {truth_word}    Baxter cube pick {episode_id} ({row.duration_s:.1f} s)",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "No model scores any clip at or above the 4.0 threshold; the rank of its expected score "
        "among the 10 clips is the signal",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    panels = []
    for i, (title, col) in enumerate(
        [
            ("Fine-tuned FLORA (video only)", "ctrl"),
            ("Fine-tuned tactile FLORA (video + fingertip force)", "force"),
        ]
    ):
        ax = fig.add_axes([0.02 + 0.49 * i, 0.36, 0.47, 0.56])
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=4)
        ok = bool(row[f"{col}_ranked_ok"])
        rank = int(row[f"{col}_rank"])
        side = (
            ("above every failure" if ok else "below some failures")
            if row.truth == 1
            else ("below both successes" if ok else "above a success")
        )
        ax.text(
            0.5,
            -0.045,
            f"expected score {row[f'{col}_E']:.2f} / 5, {_ordinal(rank)} of 10 ({side})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color=green if ok else red,
            weight="bold",
        )
        panels.append(ax)
    axf = fig.add_axes([0.06, 0.07, 0.91, 0.2])
    ymax = max(0.3, float(total.max()) * 1.05)
    axf.plot(t, fa, color="#1f77b4", lw=1.4, label="pad A")
    axf.plot(t, fb, color="#ff7f0e", lw=1.4, label="pad B")
    axf.plot(gt_, grip * ymax * 0.9, color="grey", lw=0.9, ls=":", label="gripper closure (scaled)")
    axf.axhline(tau_n, color="k", lw=0.6, ls="--")
    axf.plot(
        seen,
        np.full_like(seen, -0.03 * ymax / 0.3),
        "|",
        color="k",
        ms=5,
        label="frames the models saw",
    )
    axf.set_xlim(0, float(max(t[-1], cam.n_frames / cam.fps)))
    axf.set_ylim(-0.06 * ymax / 0.3, ymax)
    axf.set_xlabel("clip time (s)", fontsize=10)
    axf.set_ylabel("force\n(1 = grasp in ep0000)", fontsize=9)
    contact = (
        f"contact {stats['hold_s']} s, last at {stats['last_contact_s']} s"
        if stats["last_contact_s"] is not None
        else "no contact"
    )
    axf.set_title(
        "Normalized fingertip force read by the tactile model   |   "
        f"peak {stats['peak_force_norm']}x grasp, {contact}   |   "
        f"closure at end {float(grip[-1]):.2f} (1.0 = closed on nothing)",
        fontsize=10,
        loc="left",
    )
    axf.legend(loc="upper right", fontsize=8, ncol=4, frameon=False)
    cursor = axf.axvline(0, color="red", lw=1.2)
    images = [ax.imshow(np.zeros((3, 4, 3), dtype=np.uint8), aspect="equal") for ax in panels]

    step = max(1, round(cam.fps / OUT_FPS))
    out_rate = Fraction(cam.fps).limit_denominator(1000) / step
    indices = list(range(0, cam.n_frames, step))
    frames = reader.read_many(ep, "scene", indices)
    poster_t = 0.6 * float(cam.n_frames / cam.fps)
    poster_done = False
    with av.open(str(mp4), "w") as writer:
        stream = writer.add_stream("libx264", rate=out_rate)
        stream.width, stream.height = W, H  # type: ignore[attr-defined]
        stream.pix_fmt = "yuv420p"  # type: ignore[attr-defined]
        stream.options = {"crf": "22", "preset": "veryfast"}  # type: ignore[attr-defined]
        for k, arr in zip(indices, frames, strict=True):
            tt = k / cam.fps
            img = Image.fromarray(arr)
            img.thumbnail((640, 480))
            arr = np.asarray(img)
            for im in images:
                im.set_data(arr)
                im.set_extent((0, arr.shape[1], arr.shape[0], 0))
            cursor.set_xdata([tt, tt])
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = np.ascontiguousarray(rgba[..., :3])
            if not poster_done and tt >= poster_t:
                Image.fromarray(rgb).save(png)
                poster_done = True
            for pkt in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                writer.mux(pkt)
        if not poster_done:
            Image.fromarray(rgb).save(png)
        for pkt in stream.encode():
            writer.mux(pkt)
    plt.close(fig)
    return {
        "video": mp4.name,
        "poster": png.name,
        "episode_id": episode_id,
        "truth": truth_word,
        "control_expected_score": round(float(row.ctrl_E), 2),
        "control_rank": int(row.ctrl_rank),
        "control_ranked_right": bool(row.ctrl_ranked_ok),
        "tactile_expected_score": round(float(row.force_E), 2),
        "tactile_rank": int(row.force_rank),
        "tactile_ranked_right": bool(row.force_ranked_ok),
        "margin": round(float(row.margin), 2),
        "closure_at_end": round(float(grip[-1]), 2),
        **stats,
    }


def main() -> None:
    """Render the chosen clips and write clips.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="render only the first N clips")
    parser.add_argument("--all", action="store_true", help="render all ten clips, not only wins")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.1,
        help="smallest tactile-over-control margin to render",
    )
    args = parser.parse_args()
    df = _predictions()
    norm = json.loads((BASE / "verified/baxter_rollouts/norm.json").read_text())["arms"][ARM]
    cfg = load_config(BASE / "config/sources/baxter_rollouts.yaml", SourceConfig)
    adapter = BaxterRolloutsAdapter(cfg)
    adapter.discover(adapter._root)  # noqa: SLF001
    reader = PyAVReader(adapter._root)  # noqa: SLF001
    out_dir = OUT / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = df if args.all else df[df.margin >= args.min_margin]
    chosen = chosen.sort_values("margin", ascending=False)
    if args.limit:
        chosen = chosen.head(args.limit)
    report: dict[str, object] = {"clips": []}
    for episode_id, row in chosen.iterrows():
        print("rendering", episode_id, flush=True)
        report["clips"].append(render(str(episode_id), row, adapter, reader, norm, out_dir))  # type: ignore[attr-defined]
    report["not_rendered"] = [
        {"episode_id": e, "truth": _label(int(r.truth)), "margin": round(float(r.margin), 2)}
        for e, r in df.drop(chosen.index).iterrows()
    ]
    (OUT / "clips.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rendered": len(chosen), "skipped": len(df) - len(chosen)}))


if __name__ == "__main__":
    main()
