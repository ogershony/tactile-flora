"""Write one tiny synthetic REASSEMBLE recording plus extracted mp4s (run from repo root)."""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

from tests.fixtures.lerobot_writer import N_FRAMES, write_mp4

HERE = Path(__file__).parent
STEM = "2025-01-11-14-45-48"  # a file with a known F/T issue in the README table
T0 = 500.0
N_FT = 100  # 10 frames at 10 Hz span 0.9 s; F/T at ~100 Hz here


def main() -> None:
    shutil.rmtree(HERE / "extracted", ignore_errors=True)
    for p in HERE.glob("*.h5"):
        p.unlink()
    cam_t = T0 + np.arange(N_FRAMES) / 10.0
    ft_t = T0 - 0.05 + np.arange(N_FT) / 100.0
    frac = np.interp(ft_t - T0, [0, 0.3, 0.4, 0.7, 0.8, 1.0], [0, 0, 1, 1, 0, 0])
    force = np.stack([frac * 2, np.zeros(N_FT), frac * 10], axis=1)
    torque = np.zeros((N_FT, 3))
    width = (0.08 * (1 - frac)).reshape(-1, 1)
    with h5py.File(HERE / f"{STEM}.h5", "w") as f:
        rs = f.create_group("robot_state")
        rs["measured_force"] = force
        rs["measured_torque"] = torque
        rs["gripper_positions"] = width
        ts = f.create_group("timestamps")
        ts["hama1"] = cam_t.reshape(-1, 1)
        ts["measured_force"] = ft_t.reshape(-1, 1)
        ts["measured_torque"] = ft_t.reshape(-1, 1)
        ts["gripper_positions"] = ft_t.reshape(-1, 1)
        seg = f.create_group("segments_info")
        for i, (a, b, ok, text) in enumerate(
            [(0.0, 0.4, True, "Pick square peg 3."), (0.4, 0.9, False, "Insert square peg 3.")]
        ):
            g = seg.create_group(str(i))
            g["start"], g["end"], g["success"], g["text"] = T0 + a, T0 + b, ok, text
            low = g.create_group("Low_level")
            lg = low.create_group("0")
            lg["start"], lg["end"], lg["success"], lg["text"] = T0 + a, T0 + b, ok, "grasp"
    out = HERE / "extracted" / STEM
    for k, name in enumerate(("hama1", "hama2", "hand")):
        write_mp4(out / f"{name}.mp4", 10, N_FRAMES, shade=40 * k)


if __name__ == "__main__":
    main()
