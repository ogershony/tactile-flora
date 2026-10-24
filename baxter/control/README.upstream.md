# baxter_control

Bring-up for a Rethink Baxter over a direct Ethernet link from this laptop,
with the ROS Indigo + Baxter SDK 1.2.0 toolchain in a container.

The rule for the first session: **get `rostopic list` returning `/robot/...`
topics before writing a single line of control code.** Everything below is in
service of that.

```
baxter.env                 # the one place you set hostname / IP / interface
build.sh                   # builds the image
run.sh                     # starts it, with Baxter's name pre-resolved
scripts/baxter-connect.sh  # checks the whole connection chain, step by step
docker/Dockerfile          # ROS Indigo + SDK 1.2.0
ros_ws/                    # your catkin workspace (bind-mounted into the container)
```

---

## 1. Physical

1. Power Baxter on and wait ~2 minutes for the Rethink logo to clear.
2. Ethernet cable from this laptop straight into the port on Baxter's torso
   CPU. Direct, no switch.
3. Twist the e-stop out. The screen should show no fault.

## 2. Find Baxter's name

Baxter advertises itself over mDNS. `avahi-daemon` is running on this laptop
and `nsswitch.conf` has `mdns4_minimal`, so `.local` names resolve natively —
but `avahi-utils` is *not* installed, so there is no `avahi-browse` to discover
with. Use the bundled scanner instead (pure stdlib, nothing to install):

```bash
./scripts/find-baxter.py -i 169.254.0.10
```

It sends mDNS queries with the unicast-response bit set, so replies come back
to an ephemeral port and it never has to bind 5353 (avahi already owns that).

This robot answers to **`Larry.local` — 169.254.7.193**, already set in
`baxter.env`. Note the interface needs its static IP *before* this works;
mDNS has nothing to send from otherwise.

## 3. Configure the link

Edit `baxter.env` — hostname, interface (`eno2` on this machine), and the
static IP. Then:

```bash
./scripts/baxter-connect.sh --fix
```

`--fix` assigns `169.254.0.10/16` to the interface (needs sudo); without it the
script only reports. It walks the chain in order:

| # | check | what a failure means |
|---|-------|----------------------|
| 1 | carrier on `eno2` | cable, port, or robot power |
| 2 | static IP present | run with `--fix` |
| 3 | hostname resolves | wrong name, or mDNS not answering |
| 4 | ping | wrong subnet, or robot still booting |
| 5 | TCP 11311 open | ROS master not up on the robot yet |
| 6 | `rostopic list` + robot state | ROS env — usually `ROS_IP` |

The checks cascade, so fix the topmost `FAIL` first and re-run.

## 4. Container

```bash
./build.sh      # once
./run.sh        # every session
```

`run.sh` resolves Baxter's `.local` name **on the host** and injects it with
`--add-host`, because mDNS does not work inside a container — there is no
avahi in there to answer. It also uses `--network host`: ROS's XML-RPC and
TCPROS hand out addresses and expect direct return connections, which a NAT'd
Docker bridge breaks.

`ros_ws/` is bind-mounted, so your code lives in the repo, not in the image.
The SDK itself is prebuilt at `/opt/baxter_ws` and your workspace overlays it.

## 5. Point ROS at Baxter

The container's entrypoint stamps your `baxter.env` values into Rethink's
`baxter.sh`, so it is already correct when you get a prompt. Every session:

```bash
cd ~/ros_ws && ./baxter.sh
```

This drops you in a new shell with `ROS_MASTER_URI` aimed at Baxter's onboard
master and `ROS_IP` set to your address, so traffic routes in both directions.

## 6. Verify from inside

```bash
/opt/baxter-connect.sh
```

Same script, now with ROS available, so checks 5 and 6 actually run. A long
list of `/robot/...` topics means you are connected.

## 7. Enable and move

```bash
rosrun baxter_tools enable_robot.py -s     # state
rosrun baxter_tools enable_robot.py -e     # enable
rosrun baxter_tools tuck_arms.py -u        # untuck
```

If `-s` reports stopped/e-stopped: twist the red button out, then
`enable_robot.py -r` to reset, then `-e`.

Once those two commands run clean, the Limb API / MoveIt / camera-loop
architecture has something real to talk to.

---

## Troubleshooting

**`rostopic list` hangs.** Almost always `ROS_IP`. Baxter resolves the address
you advertised and opens a connection *back* to you; if that address is wrong
or unreachable, the master responds but the data never flows. Check `ROS_IP`
inside the `baxter.sh` shell matches the address actually on `eno2`.

**Nothing resolves but ping by IP works.** Skip mDNS: put the IP in
`baxter.env` as `BAXTER_HOSTNAME`. Everything downstream handles that.

**`rostopic list` works but subscribing times out.** This is the one that
looks like a robot fault and isn't. Baxter's nodes advertise themselves to the
master by *hostname* — `/realtime_loop` publishes at `http://Larry.local:32883/`
— so a container that cannot resolve `Larry.local` talks to the master fine
and then fails on every subscribe. `run.sh` prevents it by resolving on the
host and passing `--add-host`. If you start a container by hand, pass it too.

**`ready: False` is not a fault.** The field means "enabled *and* operating"
(`AssemblyState.msg`), so it reads false whenever the robot is idle. The clean
pre-enable state is `enabled/stopped/error` all false with both e-stop fields
at 0 — that is what `robot_enable.py` calls a reset, and it is what you want
to see before enabling.

**All joints warn "Current limiting active."** Normal while disabled. So are
stale `/Brain`, `/Computers` and `/RCLoader` diagnostics — those belong to
Rethink's task-level software, which does not run in SDK mode.

**Baxter is on a different subnet.** If its address is not `169.254.x.x`, set
`MY_IP`/`MY_CIDR` in `baxter.env` to something on *its* subnet and re-run
`--fix`.

**apt fails if you edit the Dockerfile.** Trusty and Indigo are EOL; the
sources are pinned to `archive.ubuntu.com` and `snapshots.ros.org`, and the
ROS snapshot's key has expired so it is marked `[trusted=yes]`. There is
nothing newer to point at.
