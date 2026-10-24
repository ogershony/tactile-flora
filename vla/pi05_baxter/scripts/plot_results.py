"""Figures for the pi0.5 Baxter run: validation curve and held-out trajectories.

    python scripts/plot_results.py --run runs/pi05_baxter_r32_rel10 --best <step> --out <dir>

Reads the run's lerobot-train log (training loss), eval.jsonl (held-out metrics per checkpoint) and
heldout_pred.npz (per-frame chunks of the chosen checkpoint, written by eval_heldout.py --dump).
Writes val_curve.png and heldout_traj.png. Needs matplotlib and pandas.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BG, INK, MUTED, GRID = "#f5f3ee", "#1b1e23", "#4a515c", "#e2ded5"
ACCENT, REF = "#2a78d6", "#9aa1ab"
JOINTS = ["s0", "s1", "e0", "e1", "w0", "w1", "w2", "gripper"]


def style(fonts: Path | None) -> None:
    if fonts and fonts.is_dir():
        for f in fonts.glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Inter"
    plt.rcParams.update({"font.size": 18, "axes.edgecolor": REF, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})


def train_loss(log: Path) -> tuple[np.ndarray, np.ndarray]:
    # lerobot logs "step:5K" after 1,000 steps, so the step is read from the tqdm counter
    # ("4950/5000 [") that precedes each INFO line on the same carriage-return segment.
    steps, loss = [], []
    text = log.read_text(errors="ignore").replace("\r", "\n")
    for m in re.finditer(r"(\d+)/\d+ \[[^\]]*\]INFO[^\n]*?loss:([0-9.eE+-]+)", text):
        steps.append(int(m.group(1)))
        loss.append(float(m.group(2)))
    return np.array(steps), np.array(loss)


def val_curve(run: Path, log: Path, best: int, out: Path) -> None:
    rows = [json.loads(x) for x in (run / "eval.jsonl").read_text().splitlines() if x.strip()]
    rows = sorted((r for r in rows if r.get("step")), key=lambda r: r["step"])
    st = np.array([r["step"] for r in rows])
    mae = np.array([r["joint_mae_rad"] for r in rows])
    hold = rows[0]["hold_joint_mae_rad"]
    s, l = train_loss(log)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.2), dpi=150, sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.3]})
    fig.patch.set_facecolor(BG)
    for a in (a1, a2):
        a.set_facecolor(BG)
        a.grid(color=GRID, lw=1)
    a1.plot(s, l, color=REF, lw=1.6)
    a1.set_ylabel("training loss")
    a1.set_ylim(0, l.max() * 1.05 if len(l) else 1)
    a2.axhline(hold, color=REF, lw=1.6, ls=(0, (4, 4)))
    a2.text(st.max(), hold * 1.04, "hold still", va="bottom", ha="right", color=MUTED, fontsize=16)
    a2.plot(st, mae, color=ACCENT, lw=2.6, marker="o", ms=7)
    bi = int(np.where(st == best)[0][0])
    a2.annotate(f"chosen: step {best:,}", (st[bi], mae[bi]), xytext=(0, -34),
                textcoords="offset points", ha="center", color=INK, fontsize=16)
    a2.set_ylabel("held-out joint error (rad)")
    a2.set_xlabel("training step")
    a2.set_ylim(0, max(hold, mae.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(out / "val_curve.png", facecolor=BG)
    plt.close(fig)


def trajectories(run: Path, out: Path, episode: int = 14, every: int = 4) -> None:
    d = np.load(run / "heldout_pred.npz")
    sel = d["episode"] == episode
    pred, gt, frame = d["pred"][sel], d["gt"][sel], d["frame"][sel]
    order = np.argsort(frame)
    pred, gt, frame = pred[order], gt[order], frame[order]
    demo = gt[:, 0, :]  # the demonstrated target at each frame
    moving = np.argsort(-demo[:, :7].std(0))[:4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, j in zip(axs.flat, sorted(moving)):
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=1)
        ax.plot(frame / 10, demo[:, j], color=INK, lw=2.4, label="demonstrated")
        for k in range(0, len(frame), every):
            t = (frame[k] + np.arange(pred.shape[1])) / 10
            ax.plot(t, pred[k, :, j], color=ACCENT, lw=1.6, alpha=0.85,
                    label="pi0.5, 1 s chunk" if k == 0 else None)
        ax.set_title(f"left_{JOINTS[j]} (rad)", fontsize=17, color=INK, loc="left")
    for ax in axs[1]:
        ax.set_xlabel("time in held-out demo (s)")
    axs[0, 0].legend(frameon=False, fontsize=15, loc="best")
    fig.tight_layout()
    fig.savefig(out / "heldout_traj.png", facecolor=BG)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--best", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fonts", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style(Path(args.fonts) if args.fonts else None)
    val_curve(Path(args.run), Path(args.log), args.best, out)
    trajectories(Path(args.run), out)
    print("wrote", out / "val_curve.png", out / "heldout_traj.png")


if __name__ == "__main__":
    main()
