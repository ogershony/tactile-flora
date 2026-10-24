"""Render side-by-side clips: fine-tuned FLORA (video only) vs fine-tuned tactile FLORA.

For every held-out HRDexDB clip (dev + test) where the tactile model (v3 force adapter, real
force) is right and the video-only control (v2 control adapter) is wrong, write an mp4 with
the scene video shown twice (one panel per model, each with its prediction) above the
fingertip-force trace the tactile model read, plus a poster PNG and a row in clips.json.

Run from the repository root with the login-node venv:

    .venv/bin/python writeup/hrdexdb/make_videos.py            # all wins
    .venv/bin/python writeup/hrdexdb/make_videos.py --limit 1  # smoke test
    .venv/bin/python writeup/hrdexdb/make_videos.py --losses   # also render clips the control wins

Inputs are the finished evaluation cells under data/acceptance/hrdexdb-expanded-{v2,v3}/eval
and the v2 raw download; nothing is recomputed with the model.
"""

from __future__ import annotations

import argparse
import json
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
from tactile_flora.sources.hrdexdb import HrdexdbAdapter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "data/acceptance/hrdexdb-expanded-v3"
V2 = ROOT / "data/acceptance/hrdexdb-expanded-v2"
OUT = ROOT / "writeup" / "hrdexdb"
FORCE_PRED = V3 / "eval/expanded-v3-force/hrdexdb_aligned_{split}/real/predictions.parquet"
CTRL_PRED = V2 / "eval/expanded-v2-control/hrdexdb_aligned_{split}/omitted/predictions.parquet"
SPLIT_NAME = {"val": "dev", "test": "test"}
OUT_FPS = 10
W, H = 1280, 720


def _predictions(split: str) -> pd.DataFrame:
    f = pd.read_parquet(str(FORCE_PRED).format(split=split)).set_index("window_id")
    c = pd.read_parquet(str(CTRL_PRED).format(split=split)).set_index("window_id").loc[f.index]
    truth = f["truth_target"].str.extract(r"Score:\s*(\d)")[0].astype(int) >= SUCCESS_THRESHOLD
    df = pd.DataFrame(
        {
            "split": SPLIT_NAME[split],
            "episode_id": f["episode_id"],
            "object": f["object"],
            "truth": truth.astype(int),
            "force_E": f["expected_score"],
            "ctrl_E": c["expected_score"],
        }
    )
    df["force_pred"] = (df["force_E"] >= SUCCESS_THRESHOLD).astype(int)
    df["ctrl_pred"] = (df["ctrl_E"] >= SUCCESS_THRESHOLD).astype(int)
    force_ok, ctrl_ok = df["force_pred"] == df["truth"], df["ctrl_pred"] == df["truth"]
    df["outcome"] = np.select(
        [force_ok & ~ctrl_ok, ~force_ok & ctrl_ok], ["force_wins", "ctrl_wins"], "same"
    )
    # Margin by which the tactile model is further on the correct side than the control.
    sign = np.where(df["truth"] == 1, 1.0, -1.0)
    df["margin"] = (df["force_E"] - df["ctrl_E"]) * sign
    return df


def _label(pred: int) -> str:
    return "SUCCESS" if pred else "FAILURE"


def _force_stats(fa: np.ndarray, fb: np.ndarray, t: np.ndarray, grasp: float, tau: float) -> dict:
    total = fa + fb
    contact = total > tau
    hold_s = float(np.trapezoid(contact.astype(float), t)) if len(t) > 1 else 0.0
    if contact.any():
        held = total[contact]
        n = len(held)
        early, late = held[: max(1, n // 3)].mean(), held[-max(1, n // 3) :].mean()
        ratio = float(late / early) if early > 0 else float("nan")
    else:
        ratio = float("nan")
    return {
        "peak_force_norm": round(float(total.max() / grasp), 2),
        "hold_s": round(hold_s, 1),
        "late_over_early_hold": None if np.isnan(ratio) else round(ratio, 2),
        "last_contact_s": round(float(t[contact][-1]), 1) if contact.any() else None,
        "clip_s": round(float(t[-1]), 1),
    }


def render(
    row: pd.Series, adapter: HrdexdbAdapter, grasp: float, tau: float, out_dir: Path
) -> dict:
    ep = adapter.load(row.episode_id)
    cam = ep.cameras["scene"]
    arm = ep.force.arms["R"]
    assert arm.kind == "pad"
    assert ep.gripper is not None
    t = np.asarray(ep.force.t)
    fa, fb = np.asarray(arm.f_a), np.asarray(arm.f_b)
    grip = np.asarray(ep.gripper.arms["R"])
    seen = np.asarray(outcome_frame_indices(cam.n_frames, cam.fps)) / cam.fps
    stats = _force_stats(fa, fb, t, grasp, tau)

    obj, seq = row.episode_id.split("/")[1:]
    stem = f"{row.split}_{obj}_{seq}_{_label(row.truth).lower()}"
    mp4, png = out_dir / f"{stem}.mp4", out_dir / f"{stem}.png"
    truth_word = _label(row.truth)
    ctrl_ok, force_ok = row.ctrl_pred == row.truth, row.force_pred == row.truth
    red, green = "#c0392b", "#1e8449"

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(
        0.5,
        0.965,
        f"Ground truth: {truth_word}    object: {obj.replace('_', ' ')}    "
        f"HRDexDB {row.episode_id}    split: {row.split}",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    panels = []
    for i, (title, pred, e, ok) in enumerate(
        [
            ("Fine-tuned FLORA (video only)", row.ctrl_pred, row.ctrl_E, ctrl_ok),
            (
                "Fine-tuned tactile FLORA (video + fingertip force)",
                row.force_pred,
                row.force_E,
                force_ok,
            ),
        ]
    ):
        ax = fig.add_axes([0.02 + 0.49 * i, 0.36, 0.47, 0.56])
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=4)
        colour = green if ok else red
        ax.text(
            0.5,
            -0.045,
            f"predicts {_label(pred)}   (expected score {e:.2f} / 5)   {'correct' if ok else 'wrong'}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color=colour,
            weight="bold",
        )
        panels.append(ax)
    axf = fig.add_axes([0.06, 0.07, 0.91, 0.2])
    axf.plot(t, fa / grasp, color="#1f77b4", lw=1.4, label="pad A (most loaded fingertip)")
    axf.plot(t, fb / grasp, color="#ff7f0e", lw=1.4, label="pad B")
    axf.plot(
        t,
        grip * max(0.3, float((fa + fb).max() / grasp)),
        color="grey",
        lw=0.9,
        ls=":",
        label="hand closure (scaled)",
    )
    axf.axhline(tau / grasp, color="k", lw=0.6, ls="--")
    axf.plot(seen, np.full_like(seen, -0.03), "|", color="k", ms=5, label="frames the models saw")
    axf.set_xlim(0, float(t[-1]))
    axf.set_ylim(-0.06, max(0.3, float((fa + fb).max() / grasp) * 1.05))
    axf.set_xlabel("clip time (s)", fontsize=10)
    axf.set_ylabel("force\n(1 = median grasp)", fontsize=9)
    axf.set_title(
        f"Tared fingertip force read by the tactile model   |   peak {stats['peak_force_norm']}x median grasp, "
        f"contact {stats['hold_s']} s"
        + (
            f", last contact at {stats['last_contact_s']} s"
            if stats["last_contact_s"] is not None
            else ", no contact"
        ),
        fontsize=10,
        loc="left",
    )
    axf.legend(loc="upper right", fontsize=8, ncol=4, frameon=False)
    cursor = axf.axvline(0, color="red", lw=1.2)
    images = []
    for ax in panels:
        images.append(ax.imshow(np.zeros((3, 4, 3), dtype=np.uint8), aspect="equal"))

    poster_t = 0.6 * float(t[-1])
    with av.open(cam.path) as container, av.open(str(mp4), "w") as writer:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"
        stream = writer.add_stream("libx264", rate=OUT_FPS)
        stream.width, stream.height = W, H  # type: ignore[attr-defined]
        stream.pix_fmt = "yuv420p"  # type: ignore[attr-defined]
        stream.options = {"crf": "22", "preset": "veryfast"}  # type: ignore[attr-defined]
        step = max(1, round(cam.fps / OUT_FPS))
        poster_done = False
        for k, frame in enumerate(container.decode(vs)):
            if k % step:
                continue
            tt = k / cam.fps
            if tt > t[-1] + 0.5:
                break
            img = frame.to_image()
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
        for pkt in stream.encode():
            writer.mux(pkt)
    plt.close(fig)
    return {
        "video": mp4.name,
        "poster": png.name,
        "split": row.split,
        "episode_id": row.episode_id,
        "object": obj,
        "truth": truth_word,
        "control_expected_score": round(float(row.ctrl_E), 2),
        "control_prediction": _label(row.ctrl_pred),
        "tactile_expected_score": round(float(row.force_E), 2),
        "tactile_prediction": _label(row.force_pred),
        "margin": round(float(row.margin), 2),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="render only the first N wins")
    parser.add_argument("--losses", action="store_true", help="also render clips the control wins")
    args = parser.parse_args()
    df = pd.concat([_predictions("val"), _predictions("test")])
    norm = json.loads((V3 / "verified/hrdexdb/norm.json").read_text())["arms"]["R"]
    grasp, tau = float(norm["grasp"]), float(norm["tau"])
    adapter = HrdexdbAdapter(load_config(V3 / "config/sources/hrdexdb.yaml", SourceConfig))
    adapter.discover(adapter._root)  # noqa: SLF001
    out_dir = OUT / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, list[dict]] = {"tactile_wins": [], "control_wins": []}
    wins = df[df.outcome == "force_wins"].sort_values("margin", ascending=False)
    if args.limit:
        wins = wins.head(args.limit)
    for _, row in wins.iterrows():
        print("rendering", row.episode_id, flush=True)
        report["tactile_wins"].append(render(row, adapter, grasp, tau, out_dir))
    losses = df[df.outcome == "ctrl_wins"].sort_values("margin")
    if args.losses:
        loss_dir = out_dir / "control_wins"
        loss_dir.mkdir(exist_ok=True)
        for _, row in losses.iterrows():
            print("rendering (control wins)", row.episode_id, flush=True)
            report["control_wins"].append(render(row, adapter, grasp, tau, loss_dir))
    else:  # list them without video so the write-up can show both sides
        for _, row in losses.iterrows():
            report["control_wins"].append(
                {
                    "split": row.split,
                    "episode_id": row.episode_id,
                    "object": row.episode_id.split("/")[1],
                    "truth": _label(row.truth),
                    "control_expected_score": round(float(row.ctrl_E), 2),
                    "control_prediction": _label(row.ctrl_pred),
                    "tactile_expected_score": round(float(row.force_E), 2),
                    "tactile_prediction": _label(row.force_pred),
                    "margin": round(float(row.margin), 2),
                }
            )
    counts = df.outcome.value_counts().to_dict()
    report["counts"] = {  # type: ignore[assignment]
        "clips": int(len(df)),
        "tactile_wins": int(counts.get("force_wins", 0)),
        "control_wins": int(counts.get("ctrl_wins", 0)),
        "same": int(counts.get("same", 0)),
    }
    (OUT / "clips.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
