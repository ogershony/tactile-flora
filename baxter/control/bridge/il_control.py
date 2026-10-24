#!/usr/bin/env python3
"""ACT imitation policy (LeRobot) control loop for Baxter's left arm: cameras + joints in,
joint targets out.

Runs in .venv-lerobot (lerobot 0.6.1 + roslibpy) and drives the robot through BaxterBridge
over rosbridge. Five ways to run it, safest first:

    --offline EP     no robot: run the policy on a dataset episode (45-49 were held out of
                     training) and report action L1, latency and the safety checks
    --dry-run        real cameras + joints, one chunk printed, NO motion
    (default)        shadow: the full loop, logs what it WOULD command, never moves
    --live           moves the LEFT arm; Ctrl-C, any check, or a finished pick sends it home

Policy: OG7407/baxter-act-cube-left, tag act_baxter_cube_left_30k (lowest held-out L1),
pinned to a Hub commit and kept in models/ (fetched on first use). Trained on
OG7407/baxter-cube-pick-left: 50 hand-guided left-arm picks at 15 fps from the home pose.
  in:  observation.state [8] left_s0..w2 (rad) + left_gripper (0 closed..1 open)
       observation.images.head  the overhead webcam, 640x360 RGB
       observation.images.wrist left_hand_camera (320x200 stream), 640x360 RGB
       (both resized exactly as demo_recorder recorded them)
  out: a chunk of 30 rows [8], row k = the target for (observation time + (k+1)/15 s):
       hand-guided actions are the next frame's joints, the gripper 0.5 s ahead.

The loop asks for a new chunk every --replan s (default n_action_steps / fps = 1 s) on a
thread, so inference (CPU here, no GPU on the laptop) never stalls commanding; rows already
past when a chunk lands are skipped (they are indexed by time, not consumed).

Safety (live): starts from home with the gripper open, as every demo did; commands move
at most --max-speed rad/s per joint (path-preserving); every chunk, before any of it is
used, must start within --max-jump of the measured arm, keep every joint inside the demos'
range (+ --envelope rad, from the checkpoint's own action stats) and keep the fingertip
inside the box the demos covered (+ margins); the MEASURED fingertip is checked every tick;
stale camera frames stop the run. The gripper closes only once the commanded arm has caught
up with the plan (--grip-track): plans are timed at demo speed, the capped arm runs slower,
so the close rows otherwise come with the hand short of the cube. Any failed check ends the run (shadow too, so a shadow
run shows where a live one would stop). Keep a hand on the e-stop.

Tactile (bridge/tactile.py): the fingertip FSR pads are read alongside the policy (not fed to it;
the demos had none). Every tick logs the four pads' counts (and newtons once calibrated) in the
run's CSV, and every 200 Hz sample goes to <log>_tactile.csv on the same clock. --tactile auto
(default) logs them when the Teensy is plugged in; required stops the run without them.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baxter_bridge import JOINTS  # noqa: E402
from baxter_fk import fingertip  # noqa: E402
from demo_recorder import FPS  # noqa: E402
from dream_control import BOX_SLACK, IMAGE_SIZE, STALE_S, WEBCAM, Robot, wait_for_cameras  # noqa: E402
from rollout_recorder import RolloutRecorder  # noqa: E402
from tactile import TactileStream  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDE = "left"  # the policy is the left arm's
TASK = "pick up the red cube"
HUB_REPO = "OG7407/baxter-act-cube-left"
HUB_REVISION = (
    "967939975dbf1b31650f9056a7249ea5f30f5ed0"  # 2026-10-24, "act_baxter_cube_left_30k snapshot"
)
HUB_SUBFOLDER = "checkpoints/act_baxter_cube_left_30k"
MODEL = os.path.join(ROOT, "models", "baxter-act-cube-left", HUB_SUBFOLDER, "pretrained_model")
DATASET = "OG7407/baxter-cube-pick-left"
WRIST_CAM = "left_hand_camera"
WRIST_RES = (320, 200)  # the wrist stream the demos were recorded from
NORMALIZER = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def fetch_model(path=MODEL):
    """The pinned checkpoint's pretrained_model/ folder, downloaded into models/ if missing."""
    if not os.path.exists(os.path.join(path, "model.safetensors")):
        if path != MODEL:
            raise SystemExit("no model.safetensors in %s" % path)
        from huggingface_hub import snapshot_download

        print("downloading %s@%s (%s)..." % (HUB_REPO, HUB_REVISION[:8], HUB_SUBFOLDER), flush=True)
        snapshot_download(
            HUB_REPO,
            revision=HUB_REVISION,
            allow_patterns=[HUB_SUBFOLDER + "/pretrained_model/*"],
            local_dir=os.path.join(ROOT, "models", "baxter-act-cube-left"),
        )
    return path


class Policy(object):
    """The ACT checkpoint with its own normalizers; chunk() maps one observation to [30, 8]."""

    def __init__(self, path, device=None, threads=4):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.numpy import load_file

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(threads)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device  # trained on cuda; the laptop has no GPU
        self.policy = ACTPolicy.from_pretrained(path, config=cfg).to(device).eval()
        override = {"device_processor": {"device": device}}
        self.pre, self.post = make_pre_post_processors(
            cfg,
            pretrained_path=path,
            preprocessor_overrides=override,
            postprocessor_overrides=override,
        )
        self.config, self.device = cfg, device
        stats = load_file(os.path.join(path, NORMALIZER))
        self.action_min, self.action_max = stats["action.min"], stats["action.max"]
        self.replan_s = cfg.n_action_steps / float(FPS)

    def chunk(self, state, head, wrist):
        """state [8] float; head/wrist RGB uint8 HxWx3 (640x360) or float CHW in [0, 1]."""
        torch = self.torch

        def image(x):
            t = torch.as_tensor(np.ascontiguousarray(x))
            if t.dtype == torch.uint8:
                t = t.permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)

        with torch.inference_mode():
            batch = self.pre(
                {
                    "observation.state": torch.as_tensor(np.asarray(state, np.float32)).unsqueeze(
                        0
                    ),
                    "observation.images.head": image(head),
                    "observation.images.wrist": image(wrist),
                    "task": TASK,
                    "robot_type": "baxter",
                }
            )
            rows = self.policy.predict_action_chunk(batch)  # [1, 30, 8], normalized
            # the postprocessor takes one action [B, 8] at a time (as lerobot's policy server runs it)
            rows = torch.stack([self.post(rows[:, k]) for k in range(rows.shape[1])], 1)
        return rows[0].cpu().numpy().astype(float)


def rgb(img_bgr):
    """A live frame as the demos stored it: resized to 640x360 (INTER_AREA), RGB."""
    if (img_bgr.shape[1], img_bgr.shape[0]) != IMAGE_SIZE:
        img_bgr = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img_bgr[:, :, ::-1])


def in_box(p, args, slack=0.0, floor_slack=0.0):
    """Why fingertip p (Baxter base frame) is outside the box, or None."""
    x, y, z = map(float, p)
    if z < args.z_floor - floor_slack:
        return "z=%.3f below the floor %.3f" % (z, args.z_floor)
    if z > args.z_ceiling + slack:
        return "z=%.3f above the ceiling %.3f" % (z, args.z_ceiling)
    if not args.x_range[0] - slack <= x <= args.x_range[1] + slack:
        return "x=%.3f outside %s" % (x, args.x_range)
    if not args.y_range[0] - slack <= y <= args.y_range[1] + slack:
        return "y=%+.3f outside %s" % (y, args.y_range)
    return None


def chunk_violation(rows, state, policy, args):
    """Why a chunk must not run, or None: it must start where the arm is, stay in the joint
    range the demos covered and keep the fingertip in the demos' box, every row."""
    jump = np.abs(rows[0, :7] - state[:7]).max()
    if jump > args.max_jump:
        return "starts %.2f rad from the arm (> --max-jump %.2f)" % (jump, args.max_jump)
    lo, hi = policy.action_min[:7] - args.envelope, policy.action_max[:7] + args.envelope
    out = np.argwhere((rows[:, :7] < lo) | (rows[:, :7] > hi))
    if len(out):
        k, j = out[0]
        return "row %d: %s_%s=%.3f outside the demos' range [%.3f, %.3f] (+-%.2f)" % (
            k,
            SIDE,
            JOINTS[SIDE][j].split("_")[1],
            rows[k, j],
            lo[j] + args.envelope,
            hi[j] - args.envelope,
            args.envelope,
        )
    for k, row in enumerate(rows):
        bad = in_box(fingertip(SIDE, row[:7]), args)
        if bad:
            return "row %d fingertip %s" % (k, bad)
    return None


def plan_target(plan, t):
    """The target at time t on a chunk observed at plan[0] (row k is for t0 + (k+1)/FPS),
    linearly interpolated between rows; None once t is past its last row."""
    t0, rows = plan
    x = (t - t0) * FPS - 1.0
    if x >= len(rows) - 1:
        return None
    if x <= 0:
        return rows[0]
    k = int(x)
    return rows[k] + (rows[k + 1] - rows[k]) * (x - k)


class Request(object):
    """One policy call on its own thread, from an observation taken at `observed_at`."""

    def __init__(self, policy, observed_at, state, head, wrist):
        self.observed_at, self.state = observed_at, state
        self.result = self.error = None
        self.latency = None
        self._thread = threading.Thread(target=self._run, args=(policy, head, wrist), daemon=True)
        self._thread.start()

    def _run(self, policy, head, wrist):
        t = time.monotonic()
        try:
            self.result = policy.chunk(self.state, head, wrist)
        except Exception as e:  # handed to the loop, which stops
            self.error = e
        self.latency = time.monotonic() - t

    def done(self):
        return not self._thread.is_alive()


def offline(policy, args):
    """Run the policy over one dataset episode: action L1 against the demo, latency, checks."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.dataset, episodes=[args.offline])
    actions = np.stack([np.asarray(a, float) for a in ds.hf_dataset["action"]])
    n, H = len(actions), policy.config.chunk_size
    print(
        "episode %d: %d frames; a chunk every %d frames"
        % (args.offline, n, policy.config.n_action_steps)
    )
    l1, first, lat = [], [], []
    for i in range(0, n, policy.config.n_action_steps):
        item = ds[i]
        state = item["observation.state"].numpy()
        t = time.monotonic()
        rows = policy.chunk(
            state, item["observation.images.head"], item["observation.images.wrist"]
        )
        lat.append(time.monotonic() - t)
        true = actions[i : i + H]
        err = np.abs(rows[: len(true)] - true)
        l1.append(err.mean())
        first.append(err[0].mean())
        bad = chunk_violation(rows, state, policy, args)
        print(
            "  frame %3d: L1 chunk %.4f first %.4f | %.2f s | %s"
            % (i, l1[-1], first[-1], lat[-1], "would STOP: " + bad if bad else "checks ok"),
            flush=True,
        )
    print(
        "mean L1 chunk %.4f, first step %.4f (training's held-out 30k: 0.0278 / 0.0175); "
        "inference %.2f s median (%s)"
        % (np.mean(l1), np.mean(first), np.median(lat[1:] or lat), policy.device)
    )
    return 0


def parser(description=None):
    """il_control's options (bridge/il_rollouts.py adds its own to them)."""
    ap = argparse.ArgumentParser(description=description or __doc__.split("\n")[0])
    ap.add_argument(
        "--model",
        default=MODEL,
        help="pretrained_model/ folder (default: the pinned 30k checkpoint)",
    )
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    ap.add_argument("--offline", type=int, metavar="EP", help="no robot: run on dataset episode EP")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument(
        "--dry-run", action="store_true", help="one chunk on the real cameras + joints, NO motion"
    )
    ap.add_argument(
        "--live", action="store_true", help="MOVE the left arm (default: shadow, no motion)"
    )
    ap.add_argument(
        "--seconds", type=float, default=20.0, help="longest run (demos: 3-5 s at full speed)"
    )
    ap.add_argument(
        "--hz", type=float, default=30.0, help="command rate (targets interpolate the 15 Hz rows)"
    )
    ap.add_argument(
        "--replan",
        type=float,
        default=None,
        help="s between chunks (default: the checkpoint's n_action_steps / fps = 1.0)",
    )
    ap.add_argument(
        "--max-speed",
        type=float,
        default=0.3,
        help="per-joint cap, rad/s (the demos peaked at 0.6-1.0; the pick runs ~2-3x slower)",
    )
    ap.add_argument(
        "--lead",
        type=float,
        default=0.25,
        help="command this far (s) ahead on the plan to cancel Baxter's tracking lag",
    )
    ap.add_argument(
        "--max-jump",
        type=float,
        default=0.25,
        help="stop if a chunk starts this far (rad) from the arm (demos: <= 0.15 per frame)",
    )
    ap.add_argument(
        "--envelope",
        type=float,
        default=0.15,
        help="rad a target may go past the demos' per-joint range before a stop",
    )
    ap.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=(0.52, 0.92),
        help="fingertip x, m in Baxter base frame (demos 0.59-0.86)",
    )
    ap.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=(0.08, 0.45),
        help="fingertip y, m (demos +0.15..+0.37: the left half only)",
    )
    ap.add_argument(
        "--z-floor",
        type=float,
        default=-0.01,
        help="lowest fingertip target z, m (table ~0.003; the demos' grasps touch it)",
    )
    ap.add_argument(
        "--z-ceiling", type=float, default=0.32, help="highest fingertip z, m (demos <= 0.25)"
    )
    ap.add_argument(
        "--sag-margin",
        type=float,
        default=0.02,
        help="stop if the MEASURED fingertip falls this far below --z-floor",
    )
    ap.add_argument(
        "--close-below", type=float, default=0.5, help="grip target below this closes the gripper"
    )
    ap.add_argument("--open-above", type=float, default=0.8, help="... above this opens it again")
    ap.add_argument(
        "--grip-track",
        type=float,
        default=0.05,
        help="close only once the commanded arm is within this (rad, every joint) of the plan: "
        "plans are timed at demo speed, so the speed-capped arm reaches the close row early",
    )
    ap.add_argument(
        "--grip-settle",
        type=float,
        default=0.2,
        help="... and has stayed that close this long (s): a new plan sweeping past the arm "
        "is close for a single tick",
    )
    ap.add_argument(
        "--grip-hold",
        type=float,
        default=1.0,
        help="no re-open this soon (s) after a close: the gripper takes ~0.5 s to shut, and a "
        "chunk observed meanwhile still sees it open",
    )
    ap.add_argument(
        "--lifted-z",
        type=float,
        default=0.12,
        help="the pick is done once the gripper holds something with the fingertip this high",
    )
    ap.add_argument("--webcam", default=WEBCAM, help="overhead USB camera: device path or index")
    ap.add_argument(
        "--tactile",
        choices=("auto", "required", "off"),
        default="auto",
        help="fingertip FSR logging: auto = when the Teensy is plugged in; required = stop without it",
    )
    ap.add_argument("--tactile-port", help="Teensy serial device (default: found by USB id)")
    ap.add_argument(
        "--no-record",
        action="store_true",
        help="don't record the live rollout (default: rollouts/<date>/epNNNN, with tactile)",
    )
    ap.add_argument(
        "--note", default="", help="free text kept with the rollout, e.g. the cube's position"
    )
    ap.add_argument("--yes", action="store_true", help="skip the --live confirmation")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.live and (args.dry_run or args.offline is not None):
        ap.error("--live runs alone")

    path = fetch_model(args.model)
    print("loading %s ..." % os.path.relpath(path, ROOT), flush=True)
    policy = Policy(path, args.device)
    print(
        "ACT %s: chunk %d, n_action_steps %d, on %s"
        % (
            os.path.basename(os.path.dirname(path)),
            policy.config.chunk_size,
            policy.config.n_action_steps,
            policy.device,
        )
    )
    if args.offline is not None:
        return offline(policy, args)
    replan = args.replan or policy.replan_s

    from baxter_bridge import BaxterBridge
    from webcam import WebcamStream

    bx, sources, tac = None, {}, None
    try:
        tac = open_tactile(args)
        bx = BaxterBridge().connect()
        sources["wrist"] = bx.stream_camera(WRIST_CAM, 1000 // FPS)
        sources["head"] = WebcamStream(int(args.webcam) if args.webcam.isdigit() else args.webcam)
        wait_for_cameras(sources)
        robot = Robot(bx)
        wh = sources["wrist"].latest()[0].shape
        if (wh[1], wh[0]) != WRIST_RES:
            print(
                "WARNING: wrist stream is %dx%d; the demos were %dx%d:\n"
                "  ./scripts/cx.sh 'rosrun baxter_tools camera_control.py -o left_hand_camera -r 320x200'"
                % ((wh[1], wh[0]) + WRIST_RES)
            )
        rest = json.load(open(os.path.join(ROOT, "config", "rest_pose.json")))[SIDE]
        rest = np.array([rest[n] for n in JOINTS[SIDE]])
        q = robot.joints(SIDE)
        print(
            "left joints %s grip %.2f; %.3f rad from home"
            % (np.round(q, 3), robot.grip(SIDE), np.abs(q - rest).max())
        )

        if args.live:
            if not args.yes:
                print(
                    "\nLIVE: homes both arms, opens the left gripper, then MOVES the left arm at <= %.2f rad/s "
                    "for up to %.0f s. E-stop in hand." % (args.max_speed, args.seconds)
                )
                if input("type 'go' to start: ").strip() != "go":
                    print("aborted.")
                    return 1
            from home import home

            home(bx)
            bx.gripper(SIDE, "open")  # every demo starts open, at home
            time.sleep(0.5)
        elif np.abs(robot.joints(SIDE) - rest).max() > 0.05:
            print("NOTE: the arm is not at home; every demo started there (within 0.01 rad)")

        def observe():
            img_h, _ = sources["head"].latest()
            img_w, _ = sources["wrist"].latest()
            state = np.append(robot.joints(SIDE), robot.grip(SIDE)).astype(np.float32)
            return state, rgb(img_h), rgb(img_w)

        t = time.monotonic()
        state, head, wrist = observe()
        rows = policy.chunk(state, head, wrist)  # also warms the model up (first call is slowest)
        print(
            "first chunk in %.2f s: row 0 %s grip %.2f | row %d %s grip %.2f"
            % (
                time.monotonic() - t,
                np.round(rows[0, :7], 3),
                rows[0, 7],
                len(rows) - 1,
                np.round(rows[-1, :7], 3),
                rows[-1, 7],
            )
        )
        bad = chunk_violation(rows, state, policy, args)
        print("checks:", "would STOP: " + bad if bad else "ok")
        if tac is not None:
            h = tac.health(1.0)
            print(
                "tactile, last 1 s (counts):",
                ", ".join("%s %.0f" % (c, m) for c, m in zip(tac.channels, h["mean"]))
                if h
                else "no data (%s)" % tac.diagnose(),
            )
        if args.dry_run:
            print("dry-run: no motion.")
            return 0
        if args.live:
            bx.set_speed_ratio(SIDE, 1.0)  # our --max-speed is the limit (see dream_control)
        rec = RolloutRecorder() if args.live and not args.no_record else None
        outcome = {"stop_reason": "time limit", "auto_success": False}
        try:
            return run(policy, robot, observe, sources, tac, args, replan, rec, outcome)
        finally:
            if args.live:
                bx.set_speed_ratio(SIDE, 0.3)
                # Baxter doesn't hold the last command: the arm settles under gravity once
                # commands stop, so lift + home on every exit (see home.py)
                from home import home

                print("sending arms home (low hands lift first)...", flush=True)
                home(bx)
                if robot.grip(SIDE) < 0.8:
                    print("left gripper is closed (holding the cube?): lower cuff button opens it")
            if rec is not None and rec.path is not None:
                outcome["label"] = ask_label(outcome)
                print("rollout saved: %s" % os.path.relpath(rec.finish(**outcome), ROOT))
    finally:
        for src in sources.values():
            src.close()
        if tac is not None:
            tac.close()
        if bx is not None:
            bx.close()


def ask_label(outcome):
    """The operator's verdict on a live rollout (the reward label), or None when skipped."""
    if not sys.stdin.isatty():
        return None
    print("\nrun ended: %s (auto success: %s)" % (outcome["stop_reason"], outcome["auto_success"]))
    try:
        ans = (
            input("label this rollout: [s]uccess / [p]artial / [f]ail / Enter = unlabeled: ")
            .strip()
            .lower()
        )
    except EOFError:
        return None
    return {"s": "success", "p": "partial", "f": "fail"}.get(ans[:1])


def open_tactile(args):
    """The fingertip FSR stream, or None (--tactile off, or auto with no Teensy plugged in)."""
    if args.tactile == "off":
        return None
    tac = TactileStream(args.tactile_port)
    if not tac.wait(3.0):
        if args.tactile == "required":
            tac.close()
            raise SystemExit("--tactile required: no tactile data: %s" % tac.diagnose())
        print("NOTE: no tactile data (%s); running WITHOUT tactile logging" % tac.diagnose())
        tac.close()
        return None
    print(tac.describe())
    return tac


def run(policy, robot, observe, sources, tac, args, replan, rec=None, outcome=None):
    """The control loop; `outcome` gets the stop reason and auto success, `rec` (live) one
    recorded rollout: frames at 15 fps + every tactile sample."""
    outcome = {} if outcome is None else outcome
    period = 1.0 / args.hz
    live = args.live
    cmd = robot.joints(SIDE)  # commands start where the arm is
    last_cmd_at = time.monotonic()
    closed = robot.grip(SIDE) < args.close_below
    plan, request, chunks, stalls, lifted_since = None, None, 0, 0, None
    deferred = False  # a close the plan wants but the arm hasn't caught up to
    tracking_since = closed_at = None

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_path = os.path.join(ROOT, "logs", time.strftime("il_%Y%m%d_%H%M%S.csv"))
    log = open(log_path, "w", newline="")
    out = csv.writer(log)
    names = [n.split("_")[1] for n in JOINTS[SIDE]]
    out.writerow(
        ["t", "chunk"]
        + ["target_" + n for n in names]
        + ["target_grip"]
        + ["measured_" + n for n in names]
        + ["measured_grip"]
        + ["cmd_" + n for n in names]
        + ["gripper_closed", "tip_x", "tip_y", "tip_z"]
        + (
            ["tac_" + c for c in tac.channels] + ["tacN_" + c for c in tac.channels] + ["tac_age"]
            if tac is not None
            else []
        )
    )
    print(
        "%s: up to %.0f s, a chunk every %.2f s, log %s"
        % ("LIVE left" if live else "SHADOW (no motion)", args.seconds, replan, log_path),
        flush=True,
    )
    start = next_tick = last_print = time.monotonic()
    tac_nan, tac_warned = np.full(4, np.nan), False
    if rec is not None:
        meta = {
            "policy": {
                "repo": HUB_REPO,
                "revision": HUB_REVISION,
                "path": os.path.relpath(args.model, ROOT),
            },
            "task": TASK,
            "note": args.note,
            "log": os.path.relpath(log_path, ROOT),
            "args": {k: v for k, v in vars(args).items() if k not in ("yes",)},
        }
        print(
            "recording rollout -> %s (tactile: %s)"
            % (
                os.path.relpath(rec.start(tac, meta, start), ROOT),
                "every sample" if tac is not None else "none",
            )
        )
    elif tac is not None:
        tac.start_log(log_path.replace(".csv", "_tactile.csv"), start)  # every sample, same t
        print("tactile: every sample -> %s" % log_path.replace(".csv", "_tactile.csv"))
    try:
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            stale = {
                r: now - src.latest()[1]
                for r, src in sources.items()
                if now - src.latest()[1] > STALE_S
            }
            if stale:
                raise RuntimeError(
                    "camera frames stale (s): %s; camera disconnected?"
                    % {r: round(a, 1) for r, a in stale.items()}
                )
            q = robot.joints(SIDE)
            tip = fingertip(SIDE, q)
            if tac is not None:
                counts, newtons, tac_t = tac.latest()
                tac_age = now - tac_t if tac_t is not None else np.inf
                if tac_age > STALE_S:
                    if args.tactile == "required":
                        raise RuntimeError(
                            "tactile data stale (%.1f s): %s" % (tac_age, tac.diagnose())
                        )
                    if not tac_warned:
                        print(
                            "  WARNING: tactile data stale (%s); logging NaN" % tac.diagnose(),
                            flush=True,
                        )
                    tac_warned = True
                    counts = newtons = None
                else:
                    tac_warned = False
            if live:
                bad = in_box(tip, args, slack=BOX_SLACK, floor_slack=args.sag_margin)
                if bad:
                    raise RuntimeError("measured fingertip %s" % bad)

            # a chunk landed: check all of it before any of it is used
            if request is not None and request.done():
                if request.error is not None:
                    raise RuntimeError(
                        "policy failed: %s: %s" % (type(request.error).__name__, request.error)
                    )
                bad = chunk_violation(request.result, request.state, policy, args)
                if bad:
                    raise RuntimeError("chunk %d %s" % (chunks + 1, bad))
                plan = (request.observed_at, request.result)
                chunks += 1
                if chunks <= 3 or chunks % 5 == 0:
                    print("  chunk %d: %.2f s inference" % (chunks, request.latency), flush=True)
                request = None
            if request is None and (plan is None or now - plan[0] >= replan):
                state, head, wrist = observe()
                request = Request(policy, time.monotonic(), state, head, wrist)

            # command `lead` s ahead on the plan: Baxter's position mode trails a moving command
            target = plan_target(plan, now + args.lead) if plan else None
            t_cmd = time.monotonic()
            max_step = args.max_speed * min(
                t_cmd - last_cmd_at, period
            )  # per elapsed time, never per tick
            last_cmd_at = t_cmd
            if target is None:
                stalls += plan is not None
            else:
                # scale the whole step (not per joint) so a capped move stays on the line to the target
                delta = target[:7] - cmd
                cmd = cmd + delta * min(1.0, max_step / max(np.abs(delta).max(), 1e-9))
                # shadow 2026-10-24: at demo-speed timing the close rows came while the hand was
                # still far from the cube, and the next chunk reopened it 0.1 s later
                # (shadow, same day: a new plan sweeping past the command looked caught up for one
                # tick, hence --grip-settle)
                lag = np.abs(target[:7] - cmd).max()
                tracking_since = (tracking_since or now) if lag <= args.grip_track else None
                settled = tracking_since is not None and now - tracking_since >= args.grip_settle
                if not closed and target[7] < args.close_below and not settled:
                    if not deferred:
                        print(
                            "  t=%4.1fs gripper close deferred: arm %.3f rad behind the plan"
                            % (now - start, lag),
                            flush=True,
                        )
                    deferred = True
                elif not closed and target[7] < args.close_below:
                    closed, deferred, closed_at = True, False, now
                    if live:
                        robot.bx.gripper(SIDE, "close", block=False)
                    print(
                        "  t=%4.1fs gripper CLOSE (target %.2f, arm %.3f rad from the plan) hand at "
                        "(%.3f, %+.3f, %.3f)"
                        % ((now - start, target[7], lag) + tuple(fingertip(SIDE, cmd))),
                        flush=True,
                    )
                else:
                    deferred = deferred and target[7] < args.close_below
                if (
                    closed
                    and target[7] > args.open_above
                    and (closed_at is None or now - closed_at >= args.grip_hold)
                ):
                    closed = False
                    if live:
                        robot.bx.gripper(SIDE, "open", block=False)
                    print(
                        "  t=%4.1fs gripper OPEN (target %.2f)" % (now - start, target[7]),
                        flush=True,
                    )
            if live:
                # sent every tick, also while waiting: Baxter drops to its own hold (and sags)
                # within a fraction of a second when commands stop
                robot.bx.set_joint_positions(SIDE, dict(zip(JOINTS[SIDE], cmd)))

            grip = robot.grip(SIDE)
            out.writerow(
                ["%.3f" % (now - start), chunks]
                + [("%.4f" % v) for v in (target if target is not None else np.full(8, np.nan))]
                + [("%.4f" % v) for v in q]
                + ["%.3f" % grip]
                + [("%.4f" % v) for v in cmd]
                + [int(closed)]
                + [("%.4f" % v) for v in tip]
                + (
                    [("%.0f" % v) for v in (counts if counts is not None else tac_nan)]
                    + [("%.3f" % v) for v in (newtons if newtons is not None else tac_nan)]
                    + ["%.3f" % tac_age]
                    if tac is not None
                    else []
                )
            )
            if rec is not None and rec.due(now):
                rec.add(
                    now,
                    {
                        "observation.state": np.append(q, grip),
                        "action": np.append(cmd, 0.0 if closed else 1.0),
                        "policy.target": target if target is not None else np.full(8, np.nan),
                        "observation.tactile": counts
                        if tac is not None and counts is not None
                        else np.full(4, np.nan),
                        "observation.tactile_newtons": newtons
                        if tac is not None and newtons is not None
                        else np.full(4, np.nan),
                        "tactile_age": tac_age
                        if tac is not None and np.isfinite(tac_age)
                        else np.nan,
                        "fingertip": tip,
                        "chunk": chunks,
                        "gripper_closed": float(closed),
                    },
                    rgb(sources["head"].latest()[0]),
                    rgb(sources["wrist"].latest()[0]),
                )
            if now - last_print >= 1.0:
                last_print = now
                print(
                    "  t=%4.1fs chunks=%d tip (%.3f, %+.3f, %.3f) grip %.2f%s %s"
                    % (
                        now - start,
                        chunks,
                        tip[0],
                        tip[1],
                        tip[2],
                        grip,
                        " pads %.0f/%.0f" % (counts[0], counts[1])
                        if tac is not None and counts is not None
                        else "",
                        "|target-q|max %.3f" % np.abs(target[:7] - q).max()
                        if target is not None
                        else (
                            "waiting for the first chunk"
                            if plan is None
                            else "chunk ran out, waiting"
                        ),
                    ),
                    flush=True,
                )

            # done: holding something, lifted (gripper reports gripping, or stopped short of closed)
            holding = robot.bx.gripper_state(SIDE).get("gripping") or (closed and grip > 0.15)
            if live and closed and holding and tip[2] > args.lifted_z:
                lifted_since = lifted_since or now
                if now - lifted_since > 1.0:
                    print("picked: holding at z=%.3f after %.1f s." % (tip[2], now - start))
                    outcome.update(stop_reason="picked", auto_success=True)
                    break
            else:
                lifted_since = None

            next_tick = max(next_tick + period, time.monotonic())  # skip missed ticks after a stall
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C).")
        outcome["stop_reason"] = "ctrl-c"
    except Exception as e:  # a failed check or error: recorded, then raised
        outcome["stop_reason"] = "%s: %s" % (type(e).__name__, e)
        raise
    finally:
        log.close()
        if tac is not None:
            tac.stop_log()
        print("done: %d chunks, %d stalled ticks; log %s" % (chunks, stalls, log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
