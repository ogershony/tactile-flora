#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start-rosbridge.sh -- launch rosbridge_server inside the baxter-session
# container (started lazily by scripts/cx.sh) so the Python-3 brain can reach
# Baxter over ws://localhost:9090. Idempotent: no-op if already listening.
#
#   ./bridge/start-rosbridge.sh          # start if not running
#   ./bridge/start-rosbridge.sh --log    # start and tail the log
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=baxter-session

# make sure the container exists (cx.sh starts it lazily)
"$HERE/scripts/cx.sh" true >/dev/null

if ss -ltn 2>/dev/null | grep -q ':9090'; then
  echo "rosbridge already listening on :9090"
else
  docker exec -d "$NAME" bash -c '
    source /opt/ros/indigo/setup.bash
    source /opt/baxter_ws/devel/setup.bash
    export ROS_MASTER_URI=http://169.254.7.193:11311
    export ROS_IP=169.254.0.10
    unset ROS_HOSTNAME
    roslaunch rosbridge_server rosbridge_websocket.launch > /tmp/rosbridge.log 2>&1'
  for i in $(seq 1 20); do
    ss -ltn 2>/dev/null | grep -q ':9090' && break
    sleep 0.5
  done
  if ss -ltn 2>/dev/null | grep -q ':9090'; then
    echo "rosbridge up on :9090"
  else
    echo "rosbridge did NOT come up -- log:"; docker exec "$NAME" tail -20 /tmp/rosbridge.log
    exit 1
  fi
fi

[[ "${1:-}" == "--log" ]] && docker exec "$NAME" tail -f /tmp/rosbridge.log
