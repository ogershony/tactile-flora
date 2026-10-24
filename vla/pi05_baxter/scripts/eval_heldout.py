"""Open-loop evaluation of Pi0.5 checkpoints on held-out Baxter demos (and the live-rollout scene).

For every frame of the held-out episodes (5 and 14, never trained on), the policy sees the two
camera images, the joint state and the task text, and predicts a 10-step (1.0 s) chunk of absolute
joint targets. That chunk is compared with what the demonstrator actually commanded over the next
second. Reported per checkpoint:

  joint_mae_rad    mean |predicted - demonstrated| over the 7 joints and the 10 steps
  step0_mae_rad    the same for the first step only (what the arm is sent next)
  final_mae_rad    the same for the last step (1.0 s ahead), the hardest horizon
  gripper_mae      mean |error| of the gripper command (0 open .. 1 closed scale of the data)
  jump_rate        fraction of chunks whose first step is more than 0.25 rad from the current
                   joint state on any joint: the check that aborted 8 of the 10 ACT rollouts
                   (baxter-cube-pick-left-rollouts, "chunk N starts 0.25 rad from the arm")

and a reference that needs no model, "hold" (keep the current joints and gripper for the whole
second). With --rollouts it also runs the policy on frames of the 10 live ACT rollouts (a different
table layout, recorded on the day) and reports rollout_jump_rate and rollout_step0_dev_rad (the
largest first-step joint offset, averaged over frames); there is no ground truth there.

Flow matching samples noise, so the seed is reset before every checkpoint and each chunk is drawn
once. The base model is loaded once; later checkpoints swap in their LoRA weights.

Usage: .venv/bin/python scripts/eval_heldout.py runs/<run>/checkpoints/*/pretrained_model --out eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/baxter-cube-pick-left-v0-rel10"
VAL_EPISODES = [5, 14]
CHUNK = 10
FPS = 10
JUMP_LIMIT = 0.25
RENAME = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}


def predict(policy, pre, post, item) -> np.ndarray:
    keys = ("observation.images.head", "observation.images.wrist", "observation.state", "task")
    obs = {k: item[k] for k in keys}
    batch = pre(obs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        chunk = policy.predict_action_chunk(batch)
    chunk = post(chunk)
    return chunk.float().cpu().numpy().reshape(-1, chunk.shape[-1])[:CHUNK, :8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None, help="append one JSON line per checkpoint here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=None, help="save per-frame predictions (npz) for the LAST ckpt")
    ap.add_argument("--rollouts", type=int, default=0, help="also score every Nth live-rollout frame (0 = skip)")
    args = ap.parse_args()

    ds = LeRobotDataset(
        "OG7407/baxter-cube-pick-left-v0",
        root=DATA,
        episodes=VAL_EPISODES,
        video_backend="pyav",
        delta_timestamps={"action": [k / FPS for k in range(CHUNK)]},
    )
    roll = None
    if args.rollouts:
        from huggingface_hub import snapshot_download

        rid = "OG7407/baxter-cube-pick-left-rollouts"
        roll = LeRobotDataset(rid, root=snapshot_download(rid, repo_type="dataset"), video_backend="pyav")

    policy = pre = post = None
    for ck in args.ckpts:
        ckpt = Path(ck).resolve()
        if policy is None:
            cfg = PreTrainedConfig.from_pretrained(ckpt)
            cfg.pretrained_path = ckpt
            cfg.device = "cuda"
            policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=RENAME)
            policy.eval()
            pre, post = make_pre_post_processors(
                policy_cfg=cfg,
                pretrained_path=str(ckpt),
                preprocessor_overrides={"device_processor": {"device": "cuda"}},
            )
        else:
            res = set_peft_model_state_dict(policy, load_file(str(ckpt / "adapter_model.safetensors")))
            if res.unexpected_keys:
                raise RuntimeError(f"{ckpt}: unexpected adapter keys {res.unexpected_keys[:3]}")
        torch.manual_seed(args.seed)

        errs, step0, final, grip, jumps, hold_errs, hold_grip = [], [], [], [], [], [], []
        dump = {"pred": [], "gt": [], "state": [], "episode": [], "frame": []}
        for i in range(0, len(ds), args.stride):
            item = ds[i]
            gt = item["action"].numpy()  # (CHUNK, 8) absolute, edge-padded at episode end
            state = item["observation.state"].numpy()
            pred = predict(policy, pre, post, item)
            e = np.abs(pred[:, :7] - gt[:, :7])
            errs.append(e.mean())
            step0.append(e[0].mean())
            final.append(e[-1].mean())
            grip.append(np.abs(pred[:, 7] - gt[:, 7]).mean())
            jumps.append(float(np.abs(pred[0, :7] - state[:7]).max() > JUMP_LIMIT))
            hold_errs.append(np.abs(state[None, :7] - gt[:, :7]).mean())
            hold_grip.append(np.abs(state[7] - gt[:, 7]).mean())
            if args.dump:
                dump["pred"].append(pred)
                dump["gt"].append(gt)
                dump["state"].append(state)
                dump["episode"].append(int(item["episode_index"]))
                dump["frame"].append(int(item["frame_index"]))
        row = {
            "ckpt": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
            "step": int(ckpt.parent.name) if ckpt.parent.name.isdigit() else None,
            "n_chunks": len(errs),
            "joint_mae_rad": float(np.mean(errs)),
            "step0_mae_rad": float(np.mean(step0)),
            "final_mae_rad": float(np.mean(final)),
            "gripper_mae": float(np.mean(grip)),
            "jump_rate": float(np.mean(jumps)),
            "hold_joint_mae_rad": float(np.mean(hold_errs)),
            "hold_gripper_mae": float(np.mean(hold_grip)),
        }
        if roll is not None:
            rj, rdev = [], []
            for i in range(0, len(roll), args.rollouts):
                item = roll[i]
                state = item["observation.state"].numpy()
                pred = predict(policy, pre, post, item)
                dev = float(np.abs(pred[0, :7] - state[:7]).max())
                rdev.append(dev)
                rj.append(float(dev > JUMP_LIMIT))
            row.update({"rollout_frames": len(rj), "rollout_jump_rate": float(np.mean(rj)),
                        "rollout_step0_dev_rad": float(np.mean(rdev))})
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if args.dump and ck == args.ckpts[-1]:
            np.savez(args.dump, **{k: np.array(v) for k, v in dump.items()})


if __name__ == "__main__":
    sys.exit(main())
