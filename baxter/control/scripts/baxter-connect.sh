#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# baxter-connect.sh -- verifies steps 3 and 6 of the bring-up: the physical
# link, the IP layer, and the ROS layer. Run it on the host or inside the
# container; it skips the checks that don't apply where it is.
#
#   ./scripts/baxter-connect.sh          # check only
#   ./scripts/baxter-connect.sh --fix    # also assign the static IP (sudo)
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${BAXTER_ENV:-}"
if [[ -z "$ENV_FILE" ]]; then
  for c in "$HERE/baxter.env" /opt/baxter.env /root/ros_ws/baxter.env; do
    [[ -f "$c" ]] && { ENV_FILE="$c"; break; }
  done
fi
FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

[[ -n "$ENV_FILE" && -f "$ENV_FILE" ]] || {
  echo "no baxter.env found (looked in $HERE, /opt, /root/ros_ws)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { grn "  ok    $*"; }
bad()  { red "  FAIL  $*"; FAILED=1; }
warn() { ylw "  warn  $*"; }
FAILED=0

IN_CONTAINER=0
[[ -f /.dockerenv ]] && IN_CONTAINER=1

# --- step 3a: physical link -------------------------------------------------
step "[1/6] Ethernet link on $ETH_IFACE"
if [[ ! -d "/sys/class/net/$ETH_IFACE" ]]; then
  bad "interface $ETH_IFACE does not exist. Available: $(ls /sys/class/net | tr '\n' ' ')"
  echo "        Fix ETH_IFACE in $ENV_FILE."
else
  carrier="$(cat "/sys/class/net/$ETH_IFACE/carrier" 2>/dev/null || echo 0)"
  if [[ "$carrier" == "1" ]]; then
    ok "cable is plugged in and the link is up"
  else
    bad "no carrier -- cable unplugged, Baxter powered off, or the port is down"
    echo "        Try: sudo ip link set $ETH_IFACE up   (then re-seat the cable)"
  fi
fi

# --- step 3b: our IP on that interface --------------------------------------
step "[2/6] Static IP $MY_IP/$MY_CIDR on $ETH_IFACE"
have_ip=0
if ip -4 addr show dev "$ETH_IFACE" 2>/dev/null | grep -q "inet $MY_IP/"; then
  have_ip=1; ok "address is configured"
else
  cur="$(ip -4 -br addr show dev "$ETH_IFACE" 2>/dev/null | awk '{$1=$1};1')"
  if (( FIX )); then
    echo "  ...  assigning $MY_IP/$MY_CIDR (sudo)"
    sudo ip link set "$ETH_IFACE" up
    sudo ip addr add "$MY_IP/$MY_CIDR" dev "$ETH_IFACE" 2>/dev/null
    if ip -4 addr show dev "$ETH_IFACE" | grep -q "inet $MY_IP/"; then
      have_ip=1; ok "assigned"
    else
      bad "could not assign the address"
    fi
  else
    bad "not set. Currently: ${cur:-<nothing>}"
    echo "        Fix: $0 --fix"
    echo "        or:  sudo ip addr add $MY_IP/$MY_CIDR dev $ETH_IFACE && sudo ip link set $ETH_IFACE up"
  fi
fi
(( IN_CONTAINER )) && warn "inside the container -- this is the host's stack via --network host"

# --- step 2: name resolution ------------------------------------------------
step "[3/6] Resolving $BAXTER_HOSTNAME"
BAXTER_IP=""
if [[ "$BAXTER_HOSTNAME" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  BAXTER_IP="$BAXTER_HOSTNAME"; ok "already an IP, nothing to resolve"
else
  BAXTER_IP="$(getent ahostsv4 "$BAXTER_HOSTNAME" 2>/dev/null | awk 'NR==1{print $1}')"
  if [[ -z "$BAXTER_IP" ]] && command -v avahi-resolve-host-name >/dev/null 2>&1; then
    BAXTER_IP="$(avahi-resolve-host-name -4 "$BAXTER_HOSTNAME" 2>/dev/null | awk '{print $2}')"
  fi
  if [[ -n "$BAXTER_IP" ]]; then
    ok "$BAXTER_HOSTNAME -> $BAXTER_IP"
  else
    bad "cannot resolve $BAXTER_HOSTNAME"
    if (( IN_CONTAINER )); then
      echo "        mDNS does not work inside the container. Use run.sh, which"
      echo "        resolves the name on the host and injects it with --add-host."
    else
      echo "        Is avahi-daemon running?  systemctl status avahi-daemon"
      echo "        Discover it:  avahi-browse -art | grep -i workstation"
      echo "        Or put Baxter's raw IP in BAXTER_HOSTNAME."
    fi
  fi
fi

# --- step 3c: ping ----------------------------------------------------------
step "[4/6] Ping ${BAXTER_IP:-$BAXTER_HOSTNAME}"
if [[ -z "$BAXTER_IP" ]]; then
  warn "skipped -- nothing to ping until the name resolves"
elif ping -c 2 -W 2 "$BAXTER_IP" >/dev/null 2>&1; then
  ok "Baxter answers ICMP -- physical + IP connectivity confirmed"
else
  bad "no reply from $BAXTER_IP"
  echo "        Check: robot fully booted, cable seated, and that $MY_IP is on"
  echo "        the same subnet as $BAXTER_IP."
fi

# --- step 6a: is the ROS master listening? ----------------------------------
step "[5/6] ROS master on ${BAXTER_IP:-?}:$ROS_MASTER_PORT"
if [[ -z "$BAXTER_IP" ]]; then
  warn "skipped"
elif timeout 5 bash -c "cat < /dev/null > /dev/tcp/$BAXTER_IP/$ROS_MASTER_PORT" 2>/dev/null; then
  ok "port is open -- roscore is up on the robot"
else
  bad "port $ROS_MASTER_PORT is not accepting connections"
  echo "        Baxter may still be booting, or its ROS master died. Give it"
  echo "        another minute after the Rethink logo clears."
fi

# --- step 6b: actual ROS traffic --------------------------------------------
step "[6/6] ROS topics and robot state"
if ! command -v rostopic >/dev/null 2>&1; then
  warn "no ROS on this machine -- run this again inside the container:"
  echo "        ./run.sh   then   cd ~/ros_ws && ./baxter.sh"
  echo "        then      /opt/baxter-connect.sh"
elif [[ -z "${ROS_MASTER_URI:-}" || "$ROS_MASTER_URI" == *localhost* ]]; then
  bad "ROS_MASTER_URI is '${ROS_MASTER_URI:-unset}' -- you have not sourced baxter.sh"
  echo "        Fix: cd ~/ros_ws && ./baxter.sh"
else
  ok "ROS_MASTER_URI=$ROS_MASTER_URI  ROS_IP=${ROS_IP:-${ROS_HOSTNAME:-unset}}"
  topics="$(timeout 15 rostopic list 2>&1)"
  if [[ $? -ne 0 || -z "$topics" ]]; then
    bad "rostopic list failed or timed out"
    echo "$topics" | sed 's/^/        /' | head -5
    echo "        This is almost always ROS_IP: Baxter must be able to reach you"
    echo "        back at ${ROS_IP:-your IP}. Confirm it matches $MY_IP."
  else
    n_robot="$(grep -c '^/robot/' <<<"$topics")"
    if (( n_robot > 0 )); then
      ok "$(wc -l <<<"$topics") topics, $n_robot under /robot/ -- connected"
    else
      bad "topics are visible but none under /robot/ -- is this really Baxter?"
    fi
    if command -v rosrun >/dev/null 2>&1; then
      state="$(timeout 20 rosrun baxter_tools enable_robot.py -s 2>&1)"
      echo "$state" | sed 's/^/        /'
      # AssemblyState prints booleans as True/False and estop_button as an int.
      if grep -qiE '^\s*stopped:\s*true' <<<"$state" \
         || grep -qiE '^\s*estop_button:\s*[1-9]' <<<"$state" \
         || grep -qiE '^\s*error:\s*true' <<<"$state"; then
        bad "robot reports stopped / e-stopped / in error"
        echo "        Twist the red button out, then:"
        echo "        rosrun baxter_tools enable_robot.py -r   (reset)"
      elif grep -qiE '^\s*stopped:\s*false' <<<"$state"; then
        ok "e-stop is clear"
        # ready means "enabled and operating"; it is false whenever the
        # robot is disabled, which is the normal resting state. The clean
        # pre-enable state is what robot_enable.py calls a reset: disabled,
        # not stopped, no error, no e-stop.
        if grep -qiE '^\s*enabled:\s*true' <<<"$state"; then
          grep -qiE '^\s*ready:\s*true' <<<"$state" \
            && ok "enabled and ready to operate" \
            || warn "enabled but not ready -- probably still homing"
        else
          ok "clean reset state -- enable_robot.py -e when you want it live"
        fi
      fi
    fi
  fi
fi

# --- verdict ----------------------------------------------------------------
echo
if (( FAILED )); then
  red "Not ready. Fix the FAIL lines above, top to bottom -- they cascade."
  exit 1
else
  grn "All checks passed. Safe to enable:"
  echo "  rosrun baxter_tools enable_robot.py -e"
  echo "  rosrun baxter_tools tuck_arms.py -u"
fi
