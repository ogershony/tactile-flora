#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# nav-logger.sh -- run the navigator/cuff button mapper INTERACTIVELY in your
# own terminal, so you see each control register live as you press it.
#
#   ./scripts/nav-logger.sh            # left arm, 120s
#   ./scripts/nav-logger.sh right 60   # right arm, 60s
#
# Press Ctrl-C when you're done; then paste the output back.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-left}"
SECS="${2:-120}"

# make sure the session container is up (cx.sh starts it lazily)
"$HERE/scripts/cx.sh" true >/dev/null

docker exec -it baxter-session bash -c "
  source /opt/ros/indigo/setup.bash >/dev/null 2>&1
  source /opt/baxter_ws/devel/setup.bash >/dev/null 2>&1
  export ROS_MASTER_URI=http://169.254.7.193:11311
  export ROS_IP=169.254.0.10
  unset ROS_HOSTNAME
  python /root/ros_ws/bringup/nav_logger.py --arm $ARM --seconds $SECS
"
