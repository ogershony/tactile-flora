#!/usr/bin/env python3
"""Turn recorded Baxter demos into one LeRobot dataset and push it to the Hugging Face Hub.

Input: a folder of demo_recorder sessions (demos/<name>/epNNNN, LeRobot v2.1, one episode
each, 34-wide Dream vectors). Output: one LeRobot dataset (the installed lerobot's format,
v3.0 for lerobot 0.6.x) for a single-arm policy:

    observation.state            float32 [8]  <arm>_s0..w2 (rad), <arm>_gripper (0 closed..1 open)
    action                       float32 [8]  same, as recorded (hand-guided: the next frame)
    observation.images.head      video 360x640  overhead camera on Baxter's head
    observation.images.wrist     video 360x640  that arm's wrist camera

Episodes where the arm hardly moved (demos done with the other arm) are left out, and an
episode is cut where the gripper re-opens after its grasp (a cube released before saving).
Runs in .venv-lerobot:

    .venv-lerobot/bin/python bridge/publish_lerobot.py demos/left_patch --repo USER/baxter-cube-pick-left
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

JOINT_NAMES = ("s0", "s1", "e0", "e1", "w0", "w1", "w2")
WRIST = {"left": "wrist_l", "right": "wrist_r"}
MIN_MOVE = 0.2  # rad: the arm's largest joint excursion for an episode to count as its demo


def read_session(path, arm):
    info = json.load(open(os.path.join(path, "meta", "info.json")))
    table = pq.read_table(
        glob.glob(os.path.join(path, "data", "chunk-000", "*.parquet"))[0]
    ).to_pydict()
    state, action = np.array(table["observation.state"]), np.array(table["action"])
    sn, an = info["features"]["observation.state"]["names"], info["features"]["action"]["names"]
    pick = lambda names, vec: np.stack(
        [vec[:, names.index("%s_arm_%d" % (arm, i))] for i in range(7)]
        + [vec[:, names.index("%s_grip" % arm)]],
        1,
    )
    task = json.loads(open(os.path.join(path, "meta", "tasks.jsonl")).readline())["task"]
    return (
        info["fps"],
        task,
        pick(sn, state).astype(np.float32),
        pick(an, action).astype(np.float32),
    )


def release_frame(grip, closed=0.5):
    """Where a demo ends: the frame the gripper starts opening again after it closed (the
    operator released the cube before saving), else the whole episode. The pick, not the
    drop, is what the policy should learn."""
    shut = np.where(grip < closed)[0]
    if len(shut) == 0:
        return len(grip)
    hold = grip[shut[0] :].min()  # the grip on the cube (~0.29 on 40 mm)
    held = shut[0] + int(np.argmin(grip[shut[0] :]))
    opening = np.where(grip[held:] > hold + 0.05)[0]
    return len(grip) if len(opening) == 0 else held + opening[0]


def read_video(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="folder of epNNNN sessions")
    ap.add_argument("--repo", required=True, help="Hub dataset id, USER_OR_ORG/NAME")
    ap.add_argument("--arm", default="left", choices=("left", "right"))
    ap.add_argument("--root", default=None, help="local build dir (default demos/_lerobot/<NAME>)")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true", help="public dataset (default private)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sessions = sorted(glob.glob(os.path.join(args.src, "ep*")))
    fps = read_session(sessions[0], args.arm)[0]
    root = args.root or os.path.join(
        os.path.dirname(os.path.abspath(args.src.rstrip("/"))), "_lerobot", args.repo.split("/")[-1]
    )
    if os.path.exists(root):
        shutil.rmtree(root)  # rebuilt from the sessions every time
    names = ["%s_%s" % (args.arm, j) for j in JOINT_NAMES] + ["%s_gripper" % args.arm]
    video = {"dtype": "video", "shape": (360, 640, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(
        repo_id=args.repo,
        fps=int(fps),
        root=root,
        robot_type="baxter",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (8,), "names": names},
            "action": {"dtype": "float32", "shape": (8,), "names": names},
            "observation.images.head": video,
            "observation.images.wrist": video,
        },
    )
    kept = skipped = 0
    for path in sessions:
        f, task, state, action = read_session(path, args.arm)
        if f != fps:
            raise SystemExit("%s was recorded at %s fps, the rest at %s" % (path, f, fps))
        if np.abs(state[:, :7] - state[0, :7]).max() < MIN_MOVE:
            skipped += 1
            continue
        end = release_frame(state[:, 7])
        state, action = state[:end], action[:end]
        vids = os.path.join(path, "videos", "chunk-000")
        head = read_video(os.path.join(vids, "observation.images.scene", "episode_000000.mp4"))
        wrist = read_video(
            os.path.join(vids, "observation.images.%s" % WRIST[args.arm], "episode_000000.mp4")
        )
        n = min(len(state), len(head), len(wrist))
        for i in range(n):
            ds.add_frame(
                {
                    "observation.state": state[i],
                    "action": action[i],
                    "observation.images.head": head[i],
                    "observation.images.wrist": wrist[i],
                    "task": task,
                }
            )
        ds.save_episode()
        kept += 1
        print("  %s: %d frames" % (os.path.basename(path), n), flush=True)
    ds.finalize()
    print(
        "built %s: %d episodes (%d other-arm skipped) at %s fps -> %s"
        % (args.repo, kept, skipped, fps, root)
    )
    if args.push:
        ds.push_to_hub(private=not args.public, tags=["baxter", "lerobot", "pick"])
        print(
            "pushed: https://huggingface.co/datasets/%s (%s)"
            % (args.repo, "public" if args.public else "private")
        )


if __name__ == "__main__":
    main()
