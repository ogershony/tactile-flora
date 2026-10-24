# baxter_fsr: plug-and-play FSR fingertips for Baxter

Self-contained copy of the Baxter capture stack from `tactile_flora/baxter`: ESP32 firmware, the
USB/WiFi drivers, the live monitor, calibration, the ROS 1 FSR node and the episode recorder.
The settings and the calibration of the left gripper are included, so no calibration is needed.

## Run it on another computer

```bash
git clone -b baxter-fsr https://github.com/ogershony/tactile-flora.git
cd tactile-flora/baxter_fsr
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is missing
uv sync                                              # Python 3.11 venv in baxter_fsr/.venv
sudo usermod -aG dialout $USER && newgrp dialout     # serial port access (log out/in once later)
```

Plug the left board (HUZZAH32, CP2104) in by USB. `configs/baxter.yaml` addresses it by its
stable id (`/dev/serial/by-id/usb-Silicon_Labs_CP2104_..._02E4CAE2-if00-port0`), the same on any
Linux machine, so nothing needs editing. Run every command from `baxter_fsr/`.

```bash
uv run tf baxter monitor --config configs/baxter.yaml            # live counts + newtons, no ROS
```

Unloaded pads read 92 counts (the ESP32 floor) and 0.00 N; pressing a pad raises both.

## ROS 1 (on the Baxter workstation)

```bash
source /opt/ros/noetic/setup.bash && source ~/ros_ws/devel/setup.bash   # needs baxter_core_msgs
uv run python -m tactile_flora.baxter.ros.fsr_node --config configs/baxter.yaml
```

Publishes `/tactile/raw` and `/tactile/force` (Float32MultiArray, `[L_a, L_b, R_a, R_b]`, counts
and newtons), `/tactile/health` (JSON in std_msgs/String) and the `/tactile/zero` Trigger service.
Record one episode (cameras, gripper state, force) with:

```bash
uv run tf baxter record --config configs/baxter.yaml --notes "mug"
```

Requirements and caveats:
- A Python 3 ROS 1 (Noetic). `rospy` and the message packages are imported from the sourced ROS
  paths into the Python 3.11 venv; Python 2 distros (Indigo, Kinetic) will not work.
- The ROS codecs have NOT been run on a ROS machine yet (they are unit-tested against a fake
  transport). Check `rostopic echo /tactile/force` first.
- The recorder needs gripper state for both arms with both grippers open at the start, and
  subscribes to the head and both hand cameras. Baxter streams at most two cameras at once.

## What is configured

- `configs/baxter.yaml`: USB port by id, R_ref 5.1 kOhm, `flat_variance: 0.0` (unloaded pads sit
  flat at the floor, so the recorder's pre-start flat check is off).
- `configs/baxter_calibration.json` (2026-10-24): L_a and L_b share one counts-to-newton curve
  pooled from 11 kitchen-scale pushes, so equal counts read equal force. 0 N at 92 counts,
  2.64 N at 300, 7.52 N cap above 465. Absolute newtons are approximate (about 2 N in the upper
  range); the task uses presence of force. R_a and R_b are not calibrated and publish 0 N.

To recalibrate a pad: `uv run tf baxter calibrate --config configs/baxter.yaml --pads L_b`
(procedure in `tactile_flora/baxter/firmware/esp32_fsr/README.md`, which also has the wiring,
flashing and WiFi notes). Only the `baxter` commands of `tf` work in this folder; the dataset
steps (`tf baxter label`, `tf baxter to-lerobot`) run from the full repository.
