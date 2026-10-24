#!/usr/bin/env bash
# Starts the SDK container on the host network, with Baxter's name already
# resolved (mDNS does not cross the container boundary).
set -euo pipefail
cd "$(dirname "$0")"
source ./baxter.env

RESOLVED="$BAXTER_HOSTNAME"
if ! [[ "$BAXTER_HOSTNAME" =~ ^[0-9.]+$ ]]; then
  RESOLVED="$(getent ahostsv4 "$BAXTER_HOSTNAME" 2>/dev/null | awk 'NR==1{print $1}')" || true
  if [[ -z "$RESOLVED" ]] && command -v avahi-resolve-host-name >/dev/null 2>&1; then
    RESOLVED="$(avahi-resolve-host-name -4 "$BAXTER_HOSTNAME" 2>/dev/null | awk '{print $2}')" || true
  fi
  if [[ -z "$RESOLVED" ]]; then
    echo "warning: could not resolve $BAXTER_HOSTNAME on the host."
    echo "         Starting anyway; run /opt/baxter-connect.sh inside to debug."
    RESOLVED="$BAXTER_HOSTNAME"
  else
    echo "$BAXTER_HOSTNAME -> $RESOLVED"
  fi
fi

# Map both the short and the .local name to the resolved address inside.
HOSTS=()
if ! [[ "$BAXTER_HOSTNAME" =~ ^[0-9.]+$ ]]; then
  HOSTS+=(--add-host "${BAXTER_HOSTNAME}:${RESOLVED}")
  HOSTS+=(--add-host "${BAXTER_HOSTNAME%%.*}:${RESOLVED}")
fi

docker rm -f baxter >/dev/null 2>&1 || true
exec docker run -it --rm --name baxter \
  --network host \
  "${HOSTS[@]}" \
  -e BAXTER_RESOLVED_IP="$RESOLVED" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$PWD/ros_ws:/root/ros_ws" \
  -v "$PWD/baxter.env:/opt/baxter.env:ro" \
  baxter-dev "$@"
