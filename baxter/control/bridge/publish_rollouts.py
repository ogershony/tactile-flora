#!/usr/bin/env python3
"""Turn recorded policy rollouts (bridge/rollout_recorder.py) into one LeRobot dataset, for
reward evaluation / shaping, and optionally push it to the Hub with the raw tactile streams.

Per frame (15 fps), names and shapes as the demos (OG7407/baxter-cube-pick-left) plus:

    observation.state            float32 [8]  left_s0..w2 (rad), left_gripper (0 closed..1 open), measured
    action                       float32 [8]  what was sent: speed-capped joints, gripper command 0/1
    policy.target                float32 [8]  the policy's plan at that moment (= action before the first chunk)
    observation.images.head      video 360x640  as fed to the policy
    observation.images.wrist     video 360x640
    observation.tactile          float32 [4]  FSR counts L_a, L_b, R_a, R_b (0 where invalid)
    observation.tactile_valid    float32 [1]  1 when the tactile sample was fresh
    observation.tactile_newtons  float32 [4]  only when every rollout was recorded calibrated
    observation.fingertip        float32 [3]  measured, m in Baxter base frame
    gripper_closed               float32 [1]
    episode.success              float32 [1]  operator label: 1 success, 0.5 partial, 0 fail, -1 unlabeled
    episode.auto_success         float32 [1]  the harness's own check (held + lifted)
    next.done                    float32 [1]  1 on the last frame

A rollout whose meta.json has "exclude" (a reason) is left out. The 200 Hz tactile CSVs and each rollout's meta.json (checkpoint, settings, stop reason) go to
raw/epNNNN/ in the dataset repo, with raw/index.json mapping episode_index -> rollout folder.
Runs in .venv-lerobot:

    .venv-lerobot/bin/python bridge/publish_rollouts.py rollouts/20261024 --repo OG7407/baxter-cube-pick-left-rollouts
    ... --push            # upload (private unless --public)
"""

import argparse
import glob
import json
import os
import shutil

import av
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["left_%s" % j for j in ("s0", "s1", "e0", "e1", "w0", "w1", "w2")] + ["left_gripper"]
LABEL = {"success": 1.0, "partial": 0.5, "fail": 0.0, None: -1.0}


def read_video(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def read_rollout(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    d = pq.read_table(os.path.join(path, "data.parquet")).to_pydict()
    col = lambda k: np.array(d[k], np.float32)
    return meta, {
        k: col(k)
        for k in (
            "observation.state",
            "action",
            "policy.target",
            "observation.tactile",
            "observation.tactile_newtons",
            "fingertip",
            "gripper_closed",
        )
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", nargs="+", help="rollouts/<date> folder(s) of epNNNN rollouts")
    ap.add_argument("--repo", required=True, help="Hub dataset id, USER_OR_ORG/NAME")
    ap.add_argument(
        "--root", default=None, help="local build dir (default rollouts/_lerobot/<NAME>)"
    )
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true", help="public dataset (default private)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rollouts = sorted(
        p
        for s in args.src
        for p in glob.glob(os.path.join(s, "ep*"))
        if os.path.exists(os.path.join(p, "meta.json"))
    )
    for p in [p for p in rollouts if json.load(open(os.path.join(p, "meta.json"))).get("exclude")]:
        print(
            "  %s: left out (%s)"
            % (os.path.relpath(p, ROOT), json.load(open(os.path.join(p, "meta.json")))["exclude"])
        )
        rollouts.remove(p)
    if not rollouts:
        raise SystemExit("no finished rollouts (epNNNN/meta.json) in %s" % args.src)
    metas = [read_rollout(p)[0] for p in rollouts]
    newtons = all((m.get("tactile") or {}).get("calibrated") for m in metas)
    root = args.root or os.path.join(ROOT, "rollouts", "_lerobot", args.repo.split("/")[-1])
    if os.path.exists(root):
        shutil.rmtree(root)  # rebuilt from the rollouts every time
    video = {"dtype": "video", "shape": (360, 640, 3), "names": ["height", "width", "channels"]}
    vec = lambda n, names=None: {"dtype": "float32", "shape": (n,), "names": names}
    features = {
        "observation.state": vec(8, NAMES),
        "action": vec(8, NAMES),
        "policy.target": vec(8, NAMES),
        "observation.images.head": video,
        "observation.images.wrist": video,
        "observation.tactile": vec(4, ["L_a", "L_b", "R_a", "R_b"]),
        "observation.tactile_valid": vec(1),
        "observation.fingertip": vec(3, ["x", "y", "z"]),
        "gripper_closed": vec(1),
        "episode.success": vec(1),
        "episode.auto_success": vec(1),
        "next.done": vec(1),
    }
    if newtons:
        features["observation.tactile_newtons"] = vec(4, ["L_a", "L_b", "R_a", "R_b"])
    ds = LeRobotDataset.create(
        repo_id=args.repo,
        fps=15,
        root=root,
        robot_type="baxter",
        use_videos=True,
        features=features,
    )
    index = {}
    for i, path in enumerate(rollouts):
        meta, d = read_rollout(path)
        head = read_video(os.path.join(path, "head.mp4"))
        wrist = read_video(os.path.join(path, "wrist.mp4"))
        n = min(len(d["action"]), len(head), len(wrist))
        valid = np.isfinite(d["observation.tactile"]).all(1)
        target = np.where(np.isfinite(d["policy.target"]), d["policy.target"], d["action"])
        success, auto = LABEL.get(meta.get("label"), -1.0), float(bool(meta.get("auto_success")))
        for k in range(n):
            frame = {
                "observation.state": d["observation.state"][k],
                "action": d["action"][k],
                "policy.target": target[k],
                "observation.images.head": head[k],
                "observation.images.wrist": wrist[k],
                "observation.tactile": np.nan_to_num(d["observation.tactile"][k]),
                "observation.tactile_valid": np.float32([valid[k]]),
                "observation.fingertip": d["fingertip"][k],
                "gripper_closed": d["gripper_closed"][k : k + 1],
                "episode.success": np.float32([success]),
                "episode.auto_success": np.float32([auto]),
                "next.done": np.float32([k == n - 1]),
                "task": meta.get("task", "pick up the red cube"),
            }
            if newtons:
                frame["observation.tactile_newtons"] = np.nan_to_num(
                    d["observation.tactile_newtons"][k]
                )
            ds.add_frame(frame)
        ds.save_episode()
        index[i] = os.path.relpath(path, ROOT)
        print(
            "  %s: %d frames, %s, label %s, auto %s, tactile valid %.0f%%"
            % (
                os.path.relpath(path, ROOT),
                n,
                meta.get("stop_reason"),
                meta.get("label"),
                meta.get("auto_success"),
                100 * valid[:n].mean() if n else 0,
            ),
            flush=True,
        )
    ds.finalize()
    raw = os.path.join(root, "raw")
    for i, rel in index.items():
        dst = os.path.join(raw, "ep%04d" % i)
        os.makedirs(dst, exist_ok=True)
        for name in ("meta.json", "tactile_200hz.csv"):
            if os.path.exists(os.path.join(ROOT, rel, name)):
                shutil.copy(os.path.join(ROOT, rel, name), dst)
    json.dump(index, open(os.path.join(raw, "index.json"), "w"), indent=2)
    print(
        "built %s: %d rollouts%s -> %s"
        % (args.repo, len(index), " (+ newtons)" if newtons else "", root)
    )
    if args.push:
        ds.push_to_hub(private=not args.public, tags=["baxter", "lerobot", "rollouts", "tactile"])
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=raw,
            path_in_repo="raw",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="raw tactile (200 Hz) + rollout meta",
        )
        print(
            "pushed: https://huggingface.co/datasets/%s (%s)"
            % (args.repo, "public" if args.public else "private")
        )


if __name__ == "__main__":
    main()
