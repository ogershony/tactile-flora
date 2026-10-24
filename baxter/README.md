# Baxter

Everything that ran on or for the Rethink Baxter, in three places:

| Path | What | Runs where |
|---|---|---|
| `../tactile_flora/baxter/` | The tactile sensor stack in the Python package: Teensy 4.0 firmware for four FSR pads at 200 Hz, the ROS node that stamps the serial stream on the robot clock, the episode recorder, weight calibration and the LeRobot writer. Driven by `tf baxter record / calibrate / label / to-lerobot`. | Robot workstation (`[baxter]` extra) |
| [control/](control/README.md) | Bring-up over a direct Ethernet link (ROS Indigo in Docker), hand-guided demo collection with the cuff buttons, tactile logging, publishing demos as `OG7407/baxter-cube-pick-left`, and running the ACT policy live with safety checks while recording rollouts and tactile data as `OG7407/baxter-cube-pick-left-rollouts`. Folded in from triton-droids/baxter_control; a record of what ran, not a runnable package. | Laptop connected to Baxter |
| [act/](act/README.md) | Training the ACT pick policy with LeRobot on the SDSC cluster (Slurm scripts, held-out action L1 per checkpoint, publishing to `OG7407/baxter-act-cube-left`), with the final checkpoint's log and curves. | SDSC cluster |

The rollouts those pieces produced are evaluated in [../writeup/baxter-rollouts/](../writeup/baxter-rollouts/README.md)
through the `baxter_rollouts` source adapter in the package.
