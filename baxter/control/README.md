# baxter/control

Baxter bring-up, teleop demo collection, tactile logging and ACT policy rollouts,
folded in from an external repository.

- Source: https://github.com/triton-droids/baxter_control
- Branch: `bringup-2026-10-20` (the `main` branch only holds the bring-up skeleton)
- Commit: `43a3e05c5c544f0cfad1019945e31482c247d993`
  ("Run the trained ACT policy on Baxter and record rollouts with tactile data")
- Copied 2026-10-24. The files were run through `ruff format` only; nothing was
  linted, auto-fixed or restructured. `baxter/control` is in ruff's
  `extend-exclude` in the repo's `pyproject.toml`.
- Upstream `README.md` is kept as `README.upstream.md` and upstream `.gitignore`
  as `gitignore.upstream`, so neither shadows this repo's own files.
- `ros_ws/src/` upstream holds only a `.gitkeep` (no `CMakeLists.txt`).

Left out on purpose: the Dream-1 VLA files `bridge/dream_control.py`,
`bridge/dream_finetune.py`, `bridge/dream_warmup.py` and
`bridge/requirements-dream.txt`. They belong to a different experiment. Note that
`il_control.py`, `il_rollouts.py`, `collect_demos.py` and `demo_recorder.py`
still import a few names from `dream_control` (`WEBCAM`, `IMAGE_SIZE = (640, 360)`,
`STALE_S = 1.0`, `BOX_SLACK = 0.03`, `Robot`, `wait_for_cameras`,
`enable_baxter_ik`), so those scripts are not runnable from this copy as is; the
copy is a record of what ran on the robot, not a runnable package.

None of the Python here is part of the `tactile_flora` package. The scripts
expect the upstream layout (a `.venv-lerobot` with lerobot 0.6.1 + roslibpy, a
`.venv-dream`, a Baxter reachable over rosbridge on `ws://localhost:9090`).

## Files

### Bring-up (laptop <-> Baxter over a direct Ethernet link, ROS Indigo + Baxter SDK 1.2.0 in Docker)

- `README.upstream.md`: the upstream bring-up guide (network, mDNS, container, enable).
- `baxter.env`: the one place for Baxter's hostname/IP (`Larry.local`, 169.254.7.193),
  the laptop interface (`eno2`) and static IP (169.254.0.10/16), ROS master port.
- `build.sh`: builds the `baxter-dev` Docker image from `docker/Dockerfile`.
- `run.sh`: starts the container on the host network with Baxter's `.local` name
  pre-resolved via `--add-host`, `ros_ws/` bind-mounted.
- `docker/Dockerfile`: `ros:indigo-ros-base` + Baxter SDK v1.2.0 (built into
  `/opt/baxter_ws`) + `rosbridge_suite`, frozen EOL apt sources.
- `scripts/entrypoint.sh`: stamps `baxter.env` into Rethink's `baxter.sh`, sets
  `ROS_MASTER_URI`/`ROS_IP` in the container.
- `scripts/baxter-connect.sh`: six cascading checks (carrier, static IP, name
  resolution, ping, TCP 11311, `rostopic list` + robot state); `--fix` assigns the IP.
- `scripts/find-baxter.py`: stdlib mDNS scanner that finds Baxter without avahi-utils.
- `scripts/cx.sh`: runs one command inside a long-lived `baxter-session` container
  with ROS pointed at Baxter.
- `scripts/nav-logger.sh` + `ros_ws/bringup/nav_logger.py`: Python 2 / SDK script
  that prints Navigator and cuff DigitalIO changes, used to map the buttons.
- `scripts/record-demo.sh` + `ros_ws/bringup/demo_left.csv`: Rethink's
  `joint_recorder.py` run through the container; the CSV is the first hand-taught
  left-arm trajectory from bring-up (1705 rows, ~100 Hz, 17 columns).
- `baxter_test_results.md`: the bring-up log (robot 1.2.0.57, gripper numbers,
  camera rates, button map, blockers such as the 2012 RTC clock).
- `gitignore.upstream`: upstream's ignore list (`demos/`, `rollouts/`, `models/`,
  `logs/`, venvs).

### Bridge (Python 3 on the laptop, talks to Baxter through rosbridge)

- `bridge/start-rosbridge.sh`: launches `rosbridge_websocket` in the session
  container on `:9090`.
- `bridge/requirements.txt`: host venv pins (roslibpy, opencv, numpy, pillow).
- `bridge/baxter_bridge.py`: `BaxterBridge`, the Python 3 control layer over
  roslibpy: joint angles, gripper state/command, endpoint pose, IK service,
  camera frames/streams, speed-capped `move_to` (POSITION_MODE at 50 Hz),
  `set_speed_ratio`, face display.
- `bridge/baxter_fk.py`: numpy forward kinematics from `config/baxter.urdf`;
  `fingertip(side, q)` gives the `<side>_gripper` link position in the base frame.
- `bridge/home.py`: sends both arms to `config/rest_pose.json`, lifting a low
  hand (z < 0.07 m) first by raising s1.
- `bridge/keystone_test.py`: the rosbridge proof (joint_states rate, xdisplay publish).
- `bridge/webcam.py`: `Webcam` / `WebcamStream` for the overhead USB camera.
- `bridge/perception.py`: HSV blob detection and the pixel->table `TableHomography`.
- `bridge/calibrate.py`: interactive overhead-camera-to-table homography fit.
- `bridge/sag_test.py`: measures how far the fingertips sag below the commanded
  pose and how far they drop when commands stop.

### Teleop and demo collection

- `bridge/teach.py` + `bridge/skills.py` + `skills/pick.json` + `bridge/grasp_block.py`:
  the earlier keyframe-skill pipeline (zero-G teach, Enter captures a keyframe,
  object-relative replay through IK; `pick.json` has three keyframes:
  approach open / lower close / lift). Not used for the dataset.
- `bridge/collect_demos.py`: the demo collector; `--hand N` is the mode the
  dataset came from (see below). `--episodes N` is a scripted, self-resetting
  collector that was written for the Dream fine-tune.
- `bridge/demo_recorder.py`: `EpisodeRecorder`, samples joints, gripper and three
  cameras at `FPS = 15` on a thread and writes one LeRobot v2.1 session per episode.

### Tactile sensor

- `bridge/tactile.py`: `TactileStream`, reads the Teensy over USB serial on a
  thread, fits the Teensy clock onto `time.monotonic()`, applies the counts->N
  calibration when present, logs every sample to CSV, health checks.
- `config/tactile.yaml`: the `tactile_flora` `BaxterConfig` for this rig
  (`sample_rate_hz: 200.0`, `channels: [L_a, L_b, R_a, R_b]`, `r_ref_ohm: 4700.0`,
  `calibration_path: config/tactile_calibration.json`, saturation and
  flat-channel thresholds).
- `config/00-teensy.rules`: PJRC udev rules so a non-root user can flash and read
  the Teensy.
- `scripts/flash_tactile.sh`: installs arduino-cli + the PJRC Teensy core,
  compiles this repo's `tactile_flora/baxter/firmware/teensy_fsr` sketch and
  uploads it to the Teensy 4.0 (`teensy:avr:teensy40`).

### Dataset publishing

- `bridge/publish_lerobot.py`: demos -> one single-arm LeRobot dataset (v3.0
  format for lerobot 0.6.x), optionally pushed to the Hub.
- `bridge/publish_rollouts.py`: recorded policy rollouts -> one LeRobot dataset
  with tactile channels and outcome labels, plus the raw 200 Hz CSVs.

### Policy rollouts

- `bridge/il_control.py`: the ACT control loop (offline / dry-run / shadow / live),
  with the safety checks.
- `bridge/il_rollouts.py`: runs `il_control`'s live loop repeatedly, records every
  rollout, resets the cube in between.
- `bridge/rollout_recorder.py`: `RolloutRecorder`, one folder per rollout with
  videos, frame table, 200 Hz tactile CSV and `meta.json`.

### Config

- `config/baxter.urdf`: the robot's URDF dumped from the ROS parameter server
  (61 links); `config/baxter_arm.urdf` + `config/baxter_arm.json` are the arm
  description and TCP calibration for the Dream IK stack.
- `config/rest_pose.json`: the untucked neutral pose of both arms (radians),
  captured 2026-10-20; used as "home".
- `config/button_map.json`: left-arm cuff/navigator mapping
  (`left_lower_cuff` = zero-G squeeze, `left_lower_button` = oval/grasp,
  `left_upper_button`, nav buttons, wheel).
- `config/camera_pose.json`, `config/table_homography.json`: the look-down pose
  and the overhead-camera-to-table homography from the skill pipeline.

## Recording a demo, end to end

1. Bring-up: `./run.sh` or `./scripts/cx.sh`, `./bridge/start-rosbridge.sh`,
   `enable_robot.py -e`. The overhead USB webcam (`dream_control.WEBCAM`, a
   SONiX "Streaming Camera" on `/dev/v4l/by-id/...`) is mounted on Baxter's head;
   the wrist cameras stream at 320x200 (Baxter streams at most two cameras at once,
   so the head camera is closed and the USB webcam replaces it).
2. `.venv-lerobot/bin/python bridge/collect_demos.py --hand N` (task string
   `"pick up the red cube"`, output `demos/<YYYYMMDD>/epNNNN`). For each episode:
   both arms go home (`home.py`), the operator places the cube and presses Enter
   to start recording; squeezes the lower cuff to guide the arm in zero-G;
   the upper cuff button closes the gripper (lower button opens it); lifts;
   then Enter saves, `d` + Enter discards, `q` + Enter quits.
3. `EpisodeRecorder` samples at 15 fps: measured joints of both arms, gripper
   position (Baxter's 0..100 scaled to 0..1, so open is ~0.96 and closed ~0.04),
   and the three cameras (`scene` = overhead webcam, `wrist_l`, `wrist_r`), each
   resized to 640x360. In `--hand` mode (`action_from="next_state"`) the action
   is the next frame's measured joints and the gripper state `GRIP_LEAD_S = 0.5` s
   ahead. Tool poses are computed with the Dream IK stack's FK.
4. An episode is rejected if it is shorter than 20 frames or if the gripper never
   went below 0.5 (added after 50 episodes were recorded with the cuff-control node
   down, in which the gripper never closed).
5. Output per episode: `meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}`,
   `data/chunk-000/episode_000000.parquet`,
   `videos/chunk-000/observation.images.<role>/episode_000000.mp4` (H.264, 640x360).

The tactile sensor is not recorded during demos; `collect_demos.py` does not
import `tactile.py`.

### Tactile sensor

Four force-sensitive-resistor pads, two per gripper fingertip pair (`L_a`,
`L_b`, `R_a`, `R_b` on Teensy pins A0..A3), each in a divider
`3.3 V -- FSR -- A_n -- R_REF (4.7 kOhm) -- GND`, so counts rise with force. A
Teensy 4.0 reads them with the 12-bit ADC, 16-sample software oversampling, and
prints `<micros>,<a0>,<a1>,<a2>,<a3>` every 5 ms (200 Hz). Sending `Z` makes the
firmware average 1 s and echo `Z,<b0>,<b1>,<b2>,<b3>`; `tactile.py` ignores those
replies. The sensor part is marked unconfirmed in both the firmware and
`tactile.yaml` (ASIN B0CZ6L5NMM; `R_ref` and the saturation level are FlexiForce
A201 defaults; `sensor_confirmed: false`, `adc_max_counts: 4095`,
`saturation_counts: 4090`).

`tactile.py` finds the Teensy by USB id (16c0:0483, `/dev/ttyACM*`) or, since
2026-10-24, through a Silicon Labs CP2104 USB-UART adapter (10c4:ea60,
`/dev/ttyUSB*`, 921600 baud). It uses `tactile_flora`'s `parse_line`, `ClockFit`
and `CalibrationFile`, pinned to tactile-flora commit `533e4e6`, and delays the
first clock fit until `clock_refit_s = 5.0` s of samples have arrived. Forces in
newtons are only produced once `config/tactile_calibration.json` exists
(`tf baxter calibrate --config config/tactile.yaml`, weights
`[100, 200, 500, 1000, 2000]` g, 200 samples each); otherwise raw counts only.

## From demos to `OG7407/baxter-cube-pick-left`

`bridge/publish_lerobot.py demos/<name> --repo OG7407/baxter-cube-pick-left --arm left --push`:

- Reads each `epNNNN` session, picks the 7 left-arm joints and `left_grip` out of
  the 34-wide Dream vectors for both `observation.state` and `action`
  (8 floats: `left_s0..left_w2` in rad, `left_gripper` 0 closed .. 1 open).
- Skips an episode whose largest joint excursion is below `MIN_MOVE = 0.2` rad
  (a demo done with the other arm).
- Cuts each episode with `release_frame`: the first frame where the gripper
  (state) drops below 0.5 marks the grasp; the minimum gripper value after that is
  the hold (about 0.29 on the 40 mm cube); the episode ends at the first frame
  after the hold where the gripper exceeds `hold + 0.05`, i.e. the first sign of
  the gripper re-opening (commit `ce5692f`: "cut at the first sign of the gripper
  opening, not at 0.5"). If it never re-opens the whole episode is kept.
- Videos: `observation.images.head` from the `scene` camera,
  `observation.images.wrist` from `wrist_l`, both 360x640; frame count is the
  minimum over state, head and wrist.
- Writes a lerobot 0.6.x dataset to `demos/_lerobot/<NAME>` and pushes it
  (private unless `--public`, tags `baxter`, `lerobot`, `pick`).

`il_control.py` describes the result as "50 hand-guided left-arm picks at 15 fps
from the home pose", with episodes 45-49 held out from ACT training.

## Policy rollouts into `OG7407/baxter-cube-pick-left-rollouts`

Policy: `OG7407/baxter-act-cube-left`, subfolder
`checkpoints/act_baxter_cube_left_30k`, Hub revision
`967939975dbf1b31650f9056a7249ea5f30f5ed0`, fetched into `models/` on first use.
It takes `observation.state` [8], `observation.images.head` and
`observation.images.wrist` (640x360 RGB) and returns a chunk of 30 rows [8]; row k
is the target for observation time + (k+1)/15 s.

`bridge/il_control.py` modes, safest first: `--offline EP` (dataset episode, no
robot), `--dry-run` (real cameras + joints, one chunk, no motion), default shadow
(the full loop, logs what it would command, never moves), `--live` (moves the
left arm; asks for `go` unless `--yes`).

Loop: a new chunk is requested every `--replan` s (default `n_action_steps / fps`
= 1.0 s) on a thread; commands go out at `--hz 30` and are interpolated between
the 15 Hz rows, `--lead 0.25` s ahead on the plan. The tactile stream is read
alongside the policy but not fed to it (`--tactile auto|required|off`).

Safety checks (any failure ends the run, shadow too):

- before a chunk is used: its first row must be within `--max-jump 0.25` rad of the
  measured arm (the joint-jump guard); every row must stay inside the demos'
  per-joint action range from the checkpoint's own stats plus `--envelope 0.15`
  rad; every row's fingertip (FK) must be inside the box `--x-range 0.52 0.92`,
  `--y-range 0.08 0.45`, `--z-floor -0.01`, `--z-ceiling 0.32` m;
- every tick (live): the measured fingertip must be inside the same box with
  `BOX_SLACK = 0.03` and `--sag-margin 0.02` below the floor; a camera frame
  older than `STALE_S = 1.0` s stops the run; with `--tactile required`, stale
  tactile data stops it too;
- per-joint speed cap `--max-speed 0.3` rad/s on the whole step; Baxter's own
  speed ratio is set to 1.0 during the run and back to 0.3 after;
- gripper: closes when the plan's gripper target is below `--close-below 0.5`,
  but only once the commanded arm has tracked the plan within `--grip-track 0.05`
  rad for `--grip-settle 0.2` s (otherwise the close is deferred); re-opens when
  the target exceeds `--open-above 0.8` and at least `--grip-hold 1.0` s have
  passed since the close.

Stop reasons recorded in `meta.json`: `picked` (auto success: gripper reports
`gripping`, or closed with opening > 0.15, and the fingertip above
`--lifted-z 0.12` m for more than 1 s), `time limit` (`--seconds 20`), `ctrl-c`,
or `<ExceptionType>: <message>` for a failed check, e.g. `RuntimeError: chunk 3
starts 0.31 rad from the arm (> --max-jump 0.25)`, `measured fingertip z=...
below the floor ...`, `camera frames stale ...`, `policy failed: ...`. On every
exit the arms are sent home.

Labels: `il_control.py --live` asks the operator `[s]uccess / [p]artial / [f]ail /
Enter = unlabeled` after each run. `il_rollouts.py --rollouts N` labels
automatically (`success` when `auto_success`, else `fail`, `label_source
"auto"`); `--ask` lets the operator override (`label_source "operator"`); a
Ctrl-C run gets `label: null`. It ends the session after `--max-stops 3` safety
stops in a row. Between rollouts, if the gripper opening is between 0.15 and 0.8
(cube held; empty closes to ~0.06, the cube reads 0.29-0.37) the robot places the
cube at a random spot in `--place-x 0.69 0.83`, `--place-y 0.18 0.33`
(stored as the next rollout's `cube_xy`); otherwise it opens the gripper and
waits for the operator to put the cube back.

Per rollout (`rollouts/<YYYYMMDD>/epNNNN/`, `RolloutRecorder`):

- `head.mp4`, `wrist.mp4`: H.264, 640x360 RGB, 15 fps, the frames as fed to the policy;
- `data.parquet`: one row per 15 fps frame with `observation.state` [8] (measured),
  `action` [8] (speed-capped joints sent + gripper command 0/1), `policy.target`
  [8] (NaN before the first chunk), `observation.tactile` [4] counts,
  `observation.tactile_newtons` [4] (NaN when uncalibrated or stale), `tactile_age`,
  `fingertip` [3], `chunk`, `gripper_closed`;
- `tactile_200hz.csv`: every Teensy sample (`t, teensy_micros, counts_L_a, ...,
  counts_R_b[, N_L_a, ...]`), `t` on the same clock as `data.parquet`;
- `meta.json`: policy repo/revision/path, task, note (incl. `cube_xy`), log path,
  all CLI args, start time, fps, tactile port/channels/calibrated flag,
  `stop_reason`, `auto_success`, `label`, `label_source`, `cube_xy`,
  `session_rollout`, `frames`, `duration_s`.

`bridge/publish_rollouts.py rollouts/<date> --repo OG7407/baxter-cube-pick-left-rollouts --push`
builds a lerobot dataset at 15 fps with `observation.state`, `action`,
`policy.target`, the two videos, `observation.tactile` [4] (`L_a, L_b, R_a, R_b`,
0 where invalid), `observation.tactile_valid`, `observation.fingertip`,
`gripper_closed`, `episode.success` (1 success, 0.5 partial, 0 fail, -1
unlabeled), `episode.auto_success`, `next.done`, and `observation.tactile_newtons`
only when every rollout was recorded with a calibration. Rollouts whose
`meta.json` has an `exclude` reason are left out. The raw `tactile_200hz.csv` and
`meta.json` of each rollout go to `raw/epNNNN/` in the dataset repo with
`raw/index.json` mapping episode index to rollout folder.
