#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cx.sh -- run a command inside a long-lived baxter-dev container with the ROS
# environment already pointed at Baxter. Mirrors run.sh (host network, name
# pre-resolved via --add-host, ros_ws + baxter.env bind-mounted) but detached,
# so repeated bring-up checks reuse one container instead of spawning an
# interactive shell each time.
#
#   ./scripts/cx.sh rostopic list
#   ./scripts/cx.sh 'rostopic echo -n 1 /robot/joint_states'
#
# Start-up is lazy: the container is created on first use and reused after.
# Use ./scripts/cx.sh --stop to tear it down.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/baxter.env"

RESOLVED="169.254.7.193"      # Larry.local, resolved on the host
NAME=baxter-session

if [[ "${1:-}" == "--stop" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --network host \
    --add-host "${BAXTER_HOSTNAME}:${RESOLVED}" \
    --add-host "${BAXTER_HOSTNAME%%.*}:${RESOLVED}" \
    -e BAXTER_RESOLVED_IP="$RESOLVED" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$HERE/ros_ws:/root/ros_ws" \
    -v "$HERE/baxter.env:/opt/baxter.env:ro" \
    baxter-dev sleep infinity >/dev/null
fi

docker exec "$NAME" bash -c '
  source /opt/ros/indigo/setup.bash >/dev/null 2>&1
  source /opt/baxter_ws/devel/setup.bash >/dev/null 2>&1
  source /root/ros_ws/devel/setup.bash >/dev/null 2>&1
  export ROS_MASTER_URI=http://'"$RESOLVED"':'"${ROS_MASTER_PORT:-11311}"'
  export ROS_IP='"$MY_IP"'
  unset ROS_HOSTNAME
  '"$*"
