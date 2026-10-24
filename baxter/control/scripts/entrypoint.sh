#!/bin/bash
# Stamps baxter.env values into a copy of Rethink's baxter.sh, then hands over.
set -e

WS=/root/ros_ws
mkdir -p "$WS/src"

ENV_FILE="${BAXTER_ENV:-/opt/baxter.env}"
if [[ -f "$ENV_FILE" ]]; then source "$ENV_FILE"; fi

# Prefer the IP the launcher resolved on the host -- mDNS does not work in here.
TARGET="${BAXTER_RESOLVED_IP:-${BAXTER_HOSTNAME:-baxter.local}}"

if [[ ! -f "$WS/baxter.sh" ]]; then
  cp /opt/baxter.sh.template "$WS/baxter.sh"
fi

sed -i "s|^baxter_hostname=.*|baxter_hostname=\"${TARGET}\"|" "$WS/baxter.sh"
sed -i "s|^your_ip=.*|your_ip=\"${MY_IP:-169.254.0.10}\"|" "$WS/baxter.sh"
sed -i "s|^ros_version=.*|ros_version=\"indigo\"|" "$WS/baxter.sh"
chmod +x "$WS/baxter.sh"

# baxter.sh insists on a catkin workspace with a devel/ next to it.
if [[ ! -f "$WS/devel/setup.bash" ]]; then
  source /opt/ros/indigo/setup.bash
  source /opt/baxter_ws/devel/setup.bash
  ( cd "$WS" && catkin_make >/dev/null 2>&1 ) || true
  # catkin runs as root; hand the artifacts back to whoever owns the repo.
  if [[ -n "${HOST_UID:-}" ]]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" \
      "$WS/build" "$WS/devel" "$WS/src" "$WS/baxter.sh" 2>/dev/null || true
  fi
fi

# Set the ROS environment here rather than leaving it to ./baxter.sh. That
# script spawns a subshell, so forgetting it (or losing the line to a paste
# that races container startup) leaves ROS_MASTER_URI at localhost and every
# command fails with "master may not be running yet".
export ROS_MASTER_URI="http://${TARGET}:${ROS_MASTER_PORT:-11311}"
export ROS_IP="${MY_IP:-169.254.0.10}"
unset ROS_HOSTNAME

echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
echo "ready -- run /opt/baxter-connect.sh to verify, or ./baxter.sh for a"
echo "Rethink-style subshell (already configured, not required)."
exec "$@"
