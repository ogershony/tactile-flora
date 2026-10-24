# Baxter bring-up test results

## VERDICT: GO ✅ (as of 2026-10-20)
All GO criteria pass: §1-3 (network/ROS/enable/motion), §4 (grippers), §5.3 (cam rates), §7.2 (zero-G teaching), §8.1 (IK).
- CRITICAL checks (§1,2,3): ALL PASS (enable needed a robot reboot).
- Teaching path (§7): WORKS end-to-end (zero-G -> record -> playback).
- Grippers: PASS. Cameras: PASS (~20Hz). IK: PASS. Py3<->Baxter bridge keystone: PASS.

### Recorded numbers
- Software version: robot 1.2.0.57 / SDK 1.2.0 (match).
- Gripper (electric parallel both arms): closed~4, open~96; grasp WITH 5cm block gripping=True pos=55 force=30, ON AIR gripping=False pos=4 force=0. Threshold: grasp_ok = gripping()==True AND pos>~15.
- Camera rates @640x400: L 20.02 Hz, R 20.04 Hz (only 2 of 3 cams at once).
- Button map: left_lower_cuff=zero-G squeeze; left_lower_button=grasp trig; left_upper_button=oval; nav button0/1/2+wheel. (config/button_map.json)
- Rest pose: config/rest_pose.json (both arms, untucked neutral).
- IK: valid solutions both arms.

### Blockers / flags (none block the demo)
- TIME SYNC (§1.6): robot clock stuck 2012 (dead RTC battery, no root to fix). Does NOT affect demo (skills use homography+IK service, not tf). Club: replace RTC coin cell. Resets each boot.
- ENABLE needs a robot REBOOT when realtime loop overruns ("used too much time"). Known-good remedy: sudo /sbin/reboot (ruser allowed), ~2min, comes back enabled.
- CAMERAS: only 2 of 3 stream at once; rapid close/reopen can drop a controller -> fix with `rosservice call /cameras/reset`. Use USB webcam for overhead.

### BUILD PHASE
BAXTERBRIDGE | DONE | bridge/baxter_bridge.py -- Py3 control layer over roslibpy/rosbridge. Validated ALL primitives: joint_angles, gripper_state, endpoint_pose, ik (valid), camera_frame, set_face/set_face_text, move_to (interpolated POSITION_MODE @50Hz speed-capped; err ~0.001 rad), gripper open/close (4.5/96.0). Keystone 4/4 (commanded motion via bridge) complete. GOTCHA: wrist cams are bgra8 (4ch) -> camera_frame drops alpha, returns HxWx3 BGR. Gripper id=65538.

SKILL PIPELINE | VALIDATED | Taught 'pick' (3 kf: approach open / lower close / lift) via teach.py (Enter=capture, native oval=gripper) -> skills/pick.json. Replay via skills.py executor: IK+move_to+gripper all fired. grasp_fail detection WORKED (gripper closed empty to 4.37, gripping False -> returned grasp_fail, aborted before lift) = scene-4 recovery signal proven. Grasp MISSED the block = open-loop precision (absolute replay + few-mm repeatability + placement). FIX: object-relative skills + perception (offset to detected block xy); big objects; slow approach. teach.py button lesson: oval=left_lower_button is HARDWIRED to gripper; use Enter to capture, not a cuff button.
SKILL PIPELINE (cont.) | BOTH BRANCHES VALIDATED | At kf2, measured endpoint error vs taught = 0.1 cm (1mm! not several mm -- replay is excellent). "Skipped block" was purely block-not-between-fingers (hand placement can't hit mm precision -> needs perception). Gripper-only success test: block slid between open fingers, close -> gripping=True pos=79 force=30 = GRASP_OK. So grasp_ok (pos>15 & gripping) AND grasp_fail both proven. Executor loop (teach->save->IK->move_to 1mm->gripper->success/fail detect->abort+report) complete. NEXT: perception (wrist-cam homography) to make pick object-relative.

PERCEPTION->PICK | WORKING (2026-10-21) | Full autonomous perception-driven pick succeeds. Overhead USB webcam (/dev/video0) detects object -> homography -> table xy -> object-relative pick -> grasp confirmed (gripping=True pos=48.7 force=30). Key fixes: (1) reference_xy must = grasp-keyframe xy (where gripper closed), NOT a separate --object detection (can go stale if object moves); teach.py now auto-derives it. (2) TALL-object parallax: flat-block homography lands ~3in off a tall cylinder's base; FIX = calibrate homography with the actual demo object (blue cylinder) -> parallax baked in (6/6 inliers, 4.5mm residual). Homography is object-height-specific; nadir (straight-down) cam mount would make it height-agnostic. Pipeline: webcam->perception.py->homography->skills.py executor via grasp_block.py.

### §11.1 wrap-up (2026-10-20 EOD)
STEP 11.1 | PASS | tuck_arms.py -t -> arms in parked/tucked pose (safe for power-off). enable_robot.py -d issued. NOTE: robot will NOT stay disabled (reports enabled:True after disable + wait) -- this robot auto-enables/holds enable (same as boot-came-up-enabled). Not a problem for shutdown: tucked pose is stable, arms can't drop. Safe shutdown = tuck (done) then power off. User shutting down via sudo /sbin/shutdown -h now.

### Still TODO (optional / Dream-1 stretch): §5.4 save frames, §5.6 USB cam, §6 head/sensors, §8.2 IK move, §9 streaming, §10 dreamlabs prereqs (URDF dump, install), §11 park/disable.

---

Working branch: bringup-2026-10-20. Toolchain is ROS Indigo + Baxter SDK 1.2.0 (Python 2), NOT Noetic. Robot=Larry.local=169.254.7.193, laptop eno2=169.254.0.10/16.

STEP 1.1 | PASS | LAPTOP_IP=169.254.0.10/16 assigned to eno2 (user ran sudo baxter-connect.sh --fix). Link carrier ok.
STEP 1.2 | PASS | Larry.local resolves to 169.254.7.193; ping replies; TCP 11311 open (roscore up). CRITICAL.
STEP 1.3 | PASS | Container baxter-dev up (ROS Indigo). ROS_MASTER_URI=http://169.254.7.193:11311, ROS_IP=169.254.0.10. Helper scripts/cx.sh runs cmds in detached container.
STEP 1.4 | PASS | rostopic list: 271 topics, 219 under /robot/. /robot/joint_states and /robot/state both present. CRITICAL.
STEP 1.5 | PASS | joint_states: 17 joints (head_nod, head_pan, left_e0..w2, right_e0..w2, torso_t0) with pos/vel/effort. CRITICAL.
STEP 1.6 | FAIL | Time sync off by ~441,466,121 s (~14 yr). Robot clock=1351077080 (2012-10-24), laptop=1792543201 (2026-10-20). Robot RTC stuck in 2012. Fix needs robot-side NTP/date set (user/club action); agent will NOT change robot config. Non-critical for GO, but breaks tf/time-stamped ops. FIX ATTEMPT FAILED: ruser lacks sudoers permission for `date -s` on robot (and agent won't reconfigure robot). Needs robot admin creds OR club to fix RTC/NTP. REASSESSED: does NOT block sections 2-9 (motion converges on joint positions, camera Hz uses local arrival time, IK service ignores stamp) nor section 10 prereqs (install/URDF dump not time-sensitive). Real impact is tf coherence for actual Dream-1 realtime control later. Decision: proceed, flag for club. DEBUG: `sudo -l` on robot shows ruser root rights limited to shutdown/reboot/rc-service rethink start|stop/rc-status only -- NO date/ntpdate/ntp control. Robot is Gentoo/OpenRC. Root clock fix impossible with ruser. Real fix = replace dead RTC coin-cell battery in robot's internal PC (club hardware task); resets to 2012 every boot until then. Interim for tf coherence: sync LAPTOP to robot via laptop-side `sudo ntpdate 169.254.7.193` (reversible, breaks laptop TLS/dated ops while set) -- only when realtime/tf control is actually needed. NOTE: tf skew does NOT affect the hackathon demo (skills use wrist-cam homography + IK service, not tf).

== KEYSTONE: Py3 brain <-> Baxter bridge (de-risks Dream-1 architecture) ==
FACT | Robot stack is ROS Indigo / Python 2.7 -- baxter_interface is Py2-only. LLM/Dream/perception need Py3.11+. Bridge is mandatory.
KEYSTONE | PASS (2/4 ops, no-motion) | rosbridge-suite 0.7.16 installed in container (available in snapshot); roslibpy 2.1.0 in host .venv (Py3.12). roslibpy->rosbridge(:9090)->Baxter works. SUBSCRIBE /robot/joint_states = 128.6 Hz through bridge (20Hz loop needs 6x less -> not a bottleneck). PUBLISH image to /robot/xdisplay works. Remaining 2 ops (joint move, gripper cmd) need motion = pending user clear. Test: bridge/keystone_test.py.
STEP 10.4 | PASS (early) | rosbridge installable+installed in container.
STEP 10.3 | PASS | URDF dumped via rospy.get_param -> config/baxter.urdf. 50044 bytes, robot name "baxter", links base+left_hand+right_hand present, 61 links total.

STEP 2.1 | PASS | Robot firmware /rethink/software_version=1.2.0.57; SDK baxter_interface=1.2.0; major.minor match (1.2). Not Intera (SDK mode confirmed). Bonus: InteractionReflexController + gravity_compensation params present -> zero-G teaching controller exists (good for §7). CRITICAL.
STEP 2.2 | PASS | Clean pre-enable state: enabled/stopped/error all False, estop_button=0. Safe to enable.
STEP 2.3 | FAIL (CRITICAL) | enable_robot.py -e -> "Failed to enable robot". State stayed clean (no estop/error) -> not a safety fault; robot did not report enabled:True in time.
  Diagnostics: /Controllers/Realtime Control Loop level=1 WARN "Realtime loop used too much time in the last 30 seconds". set_super_enable subscriber=/realtime_loop @Larry.local. /Brain,/Computers Stale (NORMAL in SDK mode per README).
  Likely causes: (1) realtime control loop overrun -> controllers not healthy enough to arm (most likely; long-idle robot); (2) publish return-path: robot /realtime_loop can't reliably connect back to ROS_IP 169.254.0.10 for enable cmd (subscribe works + same subnet -> less likely); (3) transient timeout in enable wait.
  Remedy ladder (user action; ruser allowed sudo reboot + rc-service rethink): retry -> restart rethink service -> reboot robot. PLAN PAUSED pending user.
  UPDATE: Retry (A) failed identically -> not transient (cause 3 ruled out). Closed-loop PUBLISH test (DigitalIO torso_left_outer_light: before=False, ON->True, OFF->False) PASSED -> publish return-path healthy (cause 2 ruled out). Subscribe=128Hz. DIAGNOSIS CONFIRMED: cause 1, robot realtime control loop overrun; motors won't arm. Robot-side fix needed: restart Rethink stack (rc-service rethink stop/start) or reboot. Nothing wrong laptop/container side.
STEP 2.3 | PASS (CRITICAL, via reboot) | User rebooted robot (sudo reboot); came back enabled:True ready:True stopped/error False estop 0. Arms live. NOTE: "Realtime loop used too much time" diagnostic still present but non-blocking (robot enabled fine with it) -> treat as benign chronic warning on this robot. FIX FOR RECURRENCE: if enable fails, reboot robot (rc-service restart was skipped; reboot is the known-good remedy per user's prior experience).
POST-REBOOT | rosbridge auto-reconnected to fresh master (134.8 Hz subscribe OK).
STEP 3.1 | PASS | 7 joints/arm read (no motion). Post-boot pose: L e0=-1.1083 e1=1.9064 s0=-0.1515 s1=-0.9258 w0=0.6849 w1=0.9576 w2=-0.4955; R e0=1.1386 e1=1.919 s0=0.0989 s1=-0.9534 w0=-0.6519 w1=0.9837 w2=0.4617.
STEP 3.2 | PASS (CRITICAL) | left_w2 wiggle +0.2 @speed0.2: -0.4966 -> reached -0.3053 (tgt -0.2966) -> returned -0.4882. Return err 0.0084 rad (<0.02). Moved visibly, accurate return. This also = keystone op 3/4 (joint move) proven, albeit via baxter_interface not bridge.
STEP 3.3 | PASS (CRITICAL) | right_w2 wiggle +0.2 @speed0.2: 0.4621 -> reached 0.6535 (tgt 0.6621) -> returned 0.4705. Return err 0.0084 rad (<0.02). Both arms track+return accurately.
STEP 3.4 | PASS | tuck_arms.py -u: arms were already untucked, moved to neutral. Standard Baxter untuck pose (elbows up, forearms fwd). Rest pose saved to config/rest_pose.json. SECTION 3 (arm motion) COMPLETE -- all CRITICAL passes.
STEP 4.1 | PASS | Both grippers electric parallel, pre-calibrated. LEFT closed=4.5 open=96.0 err=none; RIGHT closed=3.9 open=95.9 err=none. (0=closed,100=open.)
STEP 4.2 | PASS | Grasp detection (left, ~5cm block). WITH BLOCK: gripping=True pos=55.3 force=30.2. ON AIR: gripping=False pos=4.2 force=0.0. Clear separation. GRASP-FAIL THRESHOLD for demo scene-4 recovery: grasp_ok = gripping()==True AND position>~15; else grasp_fail. gripping() is object-size-agnostic (prefer it); position scales w/ object width (~55 for 5cm block). SECTION 4 (grippers) COMPLETE.
STEP 7.1 | PASS | All 7 left-arm controls registered (user ran scripts/nav-logger.sh live). Mapping saved config/button_map.json. KEY: left_lower_cuff (DigitalIO) = cuff SQUEEZE, held True 3.6s while squeezed -> zero-G trigger ALIVE. left_lower_button=grasp trigger, left_upper_button=oval, nav button0/1/2 + wheel(base 250)=navigator. (Lesson: run interactive logger in user terminal for live feedback, not batched via agent.)
STEP 7.2 | PASS (make-or-break) | Zero-G confirmed by user: arm floats freely while cuff squeezed, holds position when released. Hand-teaching mechanism viable -> demo scenes 1/2/4 on solid ground. This is the key GO criterion for teaching path.
STEP 7.3 | PASS | Recorded ros_ws/bringup/demo_left.csv via joint_recorder.py (user, live terminal). 17 cols (time+7L+left_gripper+7R+right_gripper), 1705 rows/17.0s (~100Hz). left_gripper 4.0->96.1 (toggle captured); right_gripper flat 95.9. Full teach->record pipeline works.
STEP 7.4 | PASS (pending user visual confirm) | joint_position_file_playback.py replayed all 1705 records (loop 1/1), "Exiting example...", NO errors. Robot moved to start then replayed autonomously incl gripper toggle. SECTION 7 (teaching path: zero-G -> record -> playback) COMPLETE -- the demo's critical capability is proven. User confirmed arm visually replayed taught motion + gripper toggle.
STEP 8.1 | PASS (GO criterion) | IK service both arms: SUCCESS, isValid=True, solved from current-angle seed (result_type 2). Clock skew confirmed irrelevant to IK (stamp secs:0, solved fine).
STEP 5.1 | PASS | 3 cameras: head_camera, left_hand_camera, right_hand_camera. Initially both wrist cams open.
STEP 5.2 | PASS (after recovery) | Both wrist cams opened @640x400. GOTCHA: rapid close/reopen churn dropped right_hand_camera's controller ("Cannot locate a service"); only 2 of 3 cams get controllers at once (USB bw limit). RECOVERY: rosservice call /cameras/reset, then close head_camera, then open left+right @640x400. Both open clean.
STEP 5.3 | PASS (GO criterion) | Wrist cam rates @640x400: LEFT 20.02 Hz, RIGHT 20.04 Hz (both >15). 
STEP 5.5 | PASS (observed) | 2-of-3 camera limit confirmed: opening wrist cams dropped head_camera controller. Use USB webcam for overhead view (per demo plan).
