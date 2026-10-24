#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# record-demo.sh -- record a hand-taught trajectory with Baxter's
# joint_recorder, run INTERACTIVELY in your terminal so you control timing.
# Writes CSV into ros_ws/bringup/ (bind-mounted, so it lands in the repo).
#
#   ./scripts/record-demo.sh                 # -> ros_ws/bringup/demo_left.csv
#   ./scripts/record-demo.sh demo_pick.csv   # custom name
#
# While it runs: squeeze the cuff, move the LEFT arm through the motion, and
# open/close the gripper once with a cuff button. Press Ctrl-C to stop.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-demo_left.csv}"
"$HERE/scripts/cx.sh" true >/dev/null

docker exec -it baxter-session bash -c "
  source /opt/ros/indigo/setup.bash >/dev/null 2>&1
  source /opt/baxter_ws/devel/setup.bash >/dev/null 2>&1
  export ROS_MASTER_URI=http://169.254.7.193:11311
  export ROS_IP=169.254.0.10
  unset ROS_HOSTNAME
  mkdir -p /root/ros_ws/bringup
  echo 'Recording to ros_ws/bringup/$NAME -- move the arm, toggle gripper, Ctrl-C to stop.'
  rosrun baxter_examples joint_recorder.py -f /root/ros_ws/bringup/$NAME
"
