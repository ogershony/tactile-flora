#!/usr/bin/env python3
"""Run the ACT policy (il_control) live, over and over, recording every rollout with its tactile
data and resetting the cube in between, for reward evaluation / shaping.

    .venv-lerobot/bin/python bridge/il_rollouts.py --rollouts 20

Each rollout: both arms home, left gripper open -> il_control's live loop with every one of its
checks (recorded to rollouts/<date>/epNNNN) -> home. Then the reset:
  cube in the gripper (any pick, or a run that ended holding it) -> Baxter puts it down at a
      random spot in the demos' grasp area (--place-x / --place-y), and the next rollout starts;
      that spot is kept in the rollout's meta.json as the next rollout's cube_xy
  cube not in the gripper (a miss) -> the gripper opens and the program waits: put the cube back
      on the patch, Enter (q + Enter quits)
Labels are automatic: success when the harness saw the cube held and lifted, else fail
(label_source "auto"); --ask prompts for each one instead (Enter keeps the automatic label).

Stops after --rollouts, on Ctrl-C (the arm goes home first), or after --max-stops safety
stops in a row. A dropped overhead webcam is reopened rather than ending the session.
Someone must stay at the e-stop the whole time: the reset moves the arm too.
"""

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import il_control  # noqa: E402
from il_control import ROOT, SIDE, WRIST_CAM, Policy, fetch_model, open_tactile, run  # noqa: E402
from collect_demos import DOWN, Z_PLACE, Z_UP  # noqa: E402
from demo_recorder import FPS  # noqa: E402
from dream_control import STALE_S, Robot, wait_for_cameras  # noqa: E402
from home import home  # noqa: E402
from rollout_recorder import RolloutRecorder  # noqa: E402

HOLDING = (
    0.15,
    0.8,
)  # gripper opening (0..1) that means something is in it: cube 0.29-0.37, empty ~0.06
PLACE_SPEED = 0.2  # rad/s for the last few cm down / up


def move(bx, xyz, speed=0.3):
    q = bx.ik(SIDE, xyz, DOWN)
    if q is None:
        raise RuntimeError("no IK for the left hand at (%.3f, %+.3f, %.3f)" % tuple(xyz))
    bx.move_to(SIDE, q, max_speed=speed)


def place(bx, xy):
    """From home, holding the cube: above xy -> down -> open -> up -> home (both arms)."""
    move(bx, (xy[0], xy[1], Z_UP))
    move(bx, (xy[0], xy[1], Z_PLACE), PLACE_SPEED)
    bx.gripper(SIDE, "open")
    move(bx, (xy[0], xy[1], Z_UP), PLACE_SPEED)
    home(bx, log=lambda *_: None)


def holding(robot):
    return HOLDING[0] < robot.grip(SIDE) < HOLDING[1]


def ensure_head(sources, webcam):
    """Reopen the overhead webcam if its frames stopped (the cable drops now and then)."""
    from webcam import WebcamStream

    img, t = sources["head"].latest()
    if img is not None and time.monotonic() - t < STALE_S:
        return
    print("overhead webcam stalled: reopening (check its cable)...", flush=True)
    sources["head"].close()
    while True:
        try:
            if os.path.exists(webcam) or webcam.isdigit():
                sources["head"] = WebcamStream(int(webcam) if webcam.isdigit() else webcam)
                wait_for_cameras({"head": sources["head"]})
                return
        except RuntimeError as e:
            print("  not yet: %s" % e, flush=True)
        time.sleep(2.0)


def main():
    ap = il_control.parser(__doc__.split("\n")[0])
    ap.add_argument("--rollouts", type=int, default=10, help="how many rollouts to record")
    ap.add_argument(
        "--place-x",
        type=float,
        nargs=2,
        default=(0.69, 0.83),
        help="reset spots, fingertip x range, m (demo grasps 0.68-0.84)",
    )
    ap.add_argument(
        "--place-y",
        type=float,
        nargs=2,
        default=(0.18, 0.33),
        help="reset spots, fingertip y range, m (demo grasps +0.17..+0.32)",
    )
    ap.add_argument(
        "--max-stops",
        type=int,
        default=3,
        help="end the session after this many safety stops in a row",
    )
    ap.add_argument(
        "--ask", action="store_true", help="prompt for each label (Enter keeps the automatic one)"
    )
    ap.add_argument(
        "--rollout-dir", default=None, help="where rollouts go (default rollouts/<date>)"
    )
    args = ap.parse_args()
    args.live = True
    args.dry_run, args.offline, args.no_record = False, None, False
    base_note = args.note

    path = fetch_model(args.model)
    print("loading %s ..." % os.path.relpath(path, ROOT), flush=True)
    policy = Policy(path, args.device)
    replan = args.replan or policy.replan_s

    from baxter_bridge import BaxterBridge
    from webcam import WebcamStream

    bx, sources, tac = None, {}, None
    done = stops = 0
    cube_xy = None  # where the last reset put the cube (None: placed by hand)
    try:
        tac = open_tactile(args)
        bx = BaxterBridge().connect()
        sources["wrist"] = bx.stream_camera(WRIST_CAM, 1000 // FPS)
        sources["head"] = WebcamStream(int(args.webcam) if args.webcam.isdigit() else args.webcam)
        wait_for_cameras(sources)
        robot = Robot(bx)

        def observe():
            state = np.append(robot.joints(SIDE), robot.grip(SIDE)).astype(np.float32)
            return (
                state,
                il_control.rgb(sources["head"].latest()[0]),
                il_control.rgb(sources["wrist"].latest()[0]),
            )

        print(
            "\n%d rollouts, cube reset automatically when it ends up in the gripper. "
            "E-stop in hand the WHOLE time (the reset moves the arm too); Ctrl-C stops."
            % args.rollouts
        )
        if not args.yes and input("type 'go' to start: ").strip() != "go":
            print("aborted.")
            return 1
        home(bx)
        if holding(robot):  # e.g. still holding the last session's cube
            cube_xy = (random.uniform(*args.place_x), random.uniform(*args.place_y))
            print("cube in the gripper: placing it at (%.3f, %+.3f) first" % cube_xy, flush=True)
            place(bx, cube_xy)
        else:
            bx.gripper(SIDE, "open")

        while done < args.rollouts:
            ensure_head(sources, args.webcam)
            home(bx, log=lambda *_: None)
            bx.gripper(SIDE, "open")
            time.sleep(0.5)
            print(
                "\n=== rollout %d/%d, cube %s ==="
                % (
                    done + 1,
                    args.rollouts,
                    "at (%.3f, %+.3f) (auto reset)" % cube_xy if cube_xy else "placed by hand",
                ),
                flush=True,
            )
            args.note = "; ".join(
                x
                for x in (
                    base_note,
                    "cube_xy %.3f %.3f" % cube_xy if cube_xy else "cube placed by hand",
                )
                if x
            )
            rec = RolloutRecorder(args.rollout_dir)
            outcome = {
                "stop_reason": "time limit",
                "auto_success": False,
                "cube_xy": list(cube_xy) if cube_xy else None,
                "session_rollout": done + 1,
            }
            bx.set_speed_ratio(SIDE, 1.0)  # our --max-speed is the limit (see il_control)
            try:
                run(policy, robot, observe, sources, tac, args, replan, rec, outcome)
            except Exception as e:  # a failed check / error: recorded, session goes on
                print("  rollout stopped: %s" % e, flush=True)
            finally:
                bx.set_speed_ratio(SIDE, 0.3)
                home(bx)  # lift + home on every exit
            stops = 0 if outcome["stop_reason"] in ("picked", "time limit", "ctrl-c") else stops + 1
            auto = "success" if outcome["auto_success"] else "fail"
            outcome.update(label=auto, label_source="auto")
            if args.ask and outcome["stop_reason"] != "ctrl-c":
                manual = il_control.ask_label(outcome)
                if manual:
                    outcome.update(label=manual, label_source="operator")
            if outcome["stop_reason"] == "ctrl-c":
                outcome.update(label=None, label_source=None)  # stopped by hand: not judged
            saved = rec.finish(**outcome)
            done += saved is not None
            print(
                "rollout %d %s (%s, label %s)"
                % (
                    done,
                    "saved: " + os.path.relpath(saved, ROOT)
                    if saved
                    else "NOT recorded (stopped before it began)",
                    outcome["stop_reason"],
                    outcome["label"],
                ),
                flush=True,
            )
            if outcome["stop_reason"] == "ctrl-c":
                break
            if stops >= args.max_stops:
                print("%d safety stops in a row: ending the session, check the logs" % stops)
                break
            if done >= args.rollouts:
                break

            # --- reset ------------------------------------------------------------------
            if holding(robot):
                cube_xy = (random.uniform(*args.place_x), random.uniform(*args.place_y))
                print("reset: placing the cube at (%.3f, %+.3f)" % cube_xy, flush=True)
                place(bx, cube_xy)
                if holding(robot):
                    raise RuntimeError("the gripper still reads holding after the place: check it")
            else:
                bx.gripper(SIDE, "open")
                cube_xy = None
                if (
                    input(
                        "cube not in the gripper: put it back on the patch, Enter (q = quit): "
                    ).strip()
                    == "q"
                ):
                    break
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C).")
    finally:
        if bx is not None:
            try:
                bx.set_speed_ratio(SIDE, 0.3)
                print("sending arms home...", flush=True)
                home(bx)
            except Exception as e:  # still close everything below
                print("homing failed: %s" % e)
        for src in sources.values():
            src.close()
        if tac is not None:
            tac.close()
        if bx is not None:
            bx.close()
        print(
            "session: %d rollout(s) recorded in %s"
            % (
                done,
                args.rollout_dir
                or os.path.relpath(os.path.join(ROOT, "rollouts", time.strftime("%Y%m%d")), ROOT),
            )
        )
        print(
            "publish: .venv-lerobot/bin/python bridge/publish_rollouts.py rollouts/%s "
            "--repo OG7407/baxter-cube-pick-left-rollouts --push" % time.strftime("%Y%m%d")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
