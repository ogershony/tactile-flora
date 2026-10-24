"""Loss curves from lerobot-train Slurm logs (+ optional heldout_l1.json) -> PNG.
usage: python plot_losses.py outputs/<run> LOG [LOG ...]   (several logs = requeued segments)"""
import json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, light surface
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STEP = re.compile(r"step:([\d.]+K?) .*? loss:([\d.]+) .*? l1_loss:([\d.]+) kld_loss:([\d.]+)")
def to_step(s): return int(float(s[:-1]) * 1000) if s.endswith("K") else int(s)
EVAL = re.compile(r"step (\d+): eval_loss=([\d.]+)")

run = Path(sys.argv[1]); logs = sys.argv[2:]
train, ev = {}, {}
for lg in logs:
    for line in Path(lg).read_text(errors="ignore").splitlines():
        if m := STEP.search(line): train[to_step(m[1])] = tuple(map(float, m.group(2, 3, 4)))
        if m := EVAL.search(line): ev[int(m[1])] = float(m[2])
steps = sorted(train)
held = json.loads((run / "heldout_l1.json").read_text()) if (run / "heldout_l1.json").exists() else {}

n = 3 if held else 2
fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), dpi=150, facecolor="#fcfcfb")
def style(ax, title, ylabel):
    ax.set_facecolor("#fcfcfb"); ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("training step", color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
a = axes[0]
a.plot(steps, [train[s][0] for s in steps], color=BLUE, lw=1.6, label="train loss (L1 + 10·KL)")
if ev:
    es = sorted(ev); a.plot(es, [ev[s] for s in es], color=ORANGE, lw=1.6, marker="o", ms=3, label="held-out loss (5 episodes)")
a.set_yscale("log"); style(a, "Total loss", "loss (log)"); a.legend(frameon=False, fontsize=8, labelcolor=INK2)
b = axes[1]
b.plot(steps, [train[s][1] for s in steps], color=BLUE, lw=1.6, label="train L1 (normalized actions)")
b.plot(steps, [train[s][2] for s in steps], color=AQUA, lw=1.6, label="train KL")
b.set_yscale("log"); style(b, "Loss components", "value (log)"); b.legend(frameon=False, fontsize=8, labelcolor=INK2)
if held:
    c = axes[2]; hs = sorted(held, key=int)
    c.plot([int(s) for s in hs], [held[s]["l1_chunk_mean"] for s in hs], color=ORANGE, lw=1.6, marker="o", ms=4, label="mean over 30-step chunk")
    c.plot([int(s) for s in hs], [held[s]["l1_first_step_mean"] for s in hs], color=BLUE, lw=1.6, marker="o", ms=4, label="first action of chunk")
    best = min(hs, key=lambda s: held[s]["l1_chunk_mean"])
    c.annotate(f"best: step {int(best)}", (int(best), held[best]["l1_chunk_mean"]), textcoords="offset points", xytext=(6, 8), fontsize=8, color=INK2)
    style(c, "Held-out action L1 per checkpoint (raw units)", "L1 (rad / gripper 0-1)"); c.legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle(f"ACT on OG7407/baxter-cube-pick-left — {run.name}", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(); out = run / "loss_curves.png"; fig.savefig(out); print(out, f"{len(steps)} train points, {len(ev)} eval points, {len(held)} checkpoints")
