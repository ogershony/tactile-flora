"""Held-out action-L1 for ACT checkpoints on episodes the trainer never saw.

For each checkpoint dir (…/checkpoints/NNNNNN/pretrained_model) it runs the policy the way the
robot will (predict_action_chunk from observations only, no VAE latent from ground-truth actions),
un-normalizes, and reports L1 against the recorded actions in raw units (rad / gripper 0-1).

usage: python eval_heldout.py outputs/<run> [--episodes 45 46 47 48 49] [--device cuda]
"""
import argparse, json, sys
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

REPO = "OG7407/baxter-cube-pick-left"
NAMES = ["s0", "s1", "e0", "e1", "w0", "w1", "w2", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, nargs="+", default=[45, 46, 47, 48, 49])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    ckpts = sorted(p for p in Path(args.run_dir, "checkpoints").glob("[0-9]*/pretrained_model") if p.is_dir())
    if not ckpts:
        sys.exit(f"no checkpoints under {args.run_dir}")

    cfg0 = ACTConfig.from_pretrained(ckpts[0])
    meta = LeRobotDatasetMetadata(REPO)
    ds = LeRobotDataset(REPO, episodes=args.episodes, delta_timestamps=resolve_delta_timestamps(cfg0, meta))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
    print(f"held-out episodes {args.episodes}: {ds.num_frames} frames, chunk={cfg0.chunk_size}")

    results = {}
    for ck in ckpts:
        step = int(ck.parent.name)
        cfg = ACTConfig.from_pretrained(ck)
        cfg.device = args.device
        policy = ACTPolicy.from_pretrained(ck, config=cfg).to(args.device).eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(ck),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        sum_abs = torch.zeros(8, dtype=torch.float64)
        sum_abs_first = torch.zeros(8, dtype=torch.float64)
        n = 0
        n_first = 0
        with torch.no_grad():
            for batch in dl:
                gt = batch["action"].clone()                 # (B, chunk, 8) raw units
                pad = batch["action_is_pad"].clone()         # (B, chunk)
                obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
                obs["task"] = batch.get("task", ["pick up the red cube"] * gt.shape[0])
                obs = pre(obs)
                pred = policy.predict_action_chunk(obs)      # normalized (B, chunk, 8)
                pred = post(pred).cpu()                      # raw units
                valid = (~pad).unsqueeze(-1).double()
                err = (pred.double() - gt.double()).abs() * valid
                sum_abs += err.sum(dim=(0, 1))
                n += valid.sum().item()
                sum_abs_first += err[:, 0].sum(dim=0)
                n_first += gt.shape[0]
        per_dim = (sum_abs / max(n, 1)).tolist()
        first = (sum_abs_first / max(n_first, 1)).tolist()
        results[step] = {"l1_chunk_mean": sum(per_dim) / 8, "l1_first_step_mean": sum(first) / 8,
                         "l1_chunk_per_dim": dict(zip(NAMES, per_dim)), "l1_first_step_per_dim": dict(zip(NAMES, first))}
        print(f"step {step:6d}  L1(chunk) {results[step]['l1_chunk_mean']:.4f}  L1(first step) {results[step]['l1_first_step_mean']:.4f}  "
              f"joints {sum(per_dim[:7])/7:.4f}  grip {per_dim[7]:.4f}")
    out = Path(args.run_dir, "heldout_l1.json")
    out.write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda s: results[s]["l1_chunk_mean"])
    print(f"best by held-out chunk L1: step {best} -> {out}")


if __name__ == "__main__":
    main()
