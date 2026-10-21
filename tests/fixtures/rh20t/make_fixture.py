"""Write one tiny synthetic RH20T cfg3 port in the real layout (run from repo root)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tests.fixtures.lerobot_writer import N_FRAMES, write_lerobot_v3

HERE = Path(__file__).parent
CFG = HERE / "rh20t_cfg3"


def main() -> None:
    shutil.rmtree(CFG, ignore_errors=True)
    plateau = np.array([0, 0, 2, 6, 10, 10, 10, 6, 2, 0], dtype=np.float32)
    ft = np.zeros((N_FRAMES, 6), dtype=np.float32)
    ft[:, 2] = plateau
    ft[:, 0] = plateau * 0.3
    grip = np.array([0, 0, 0, 50, 100, 100, 100, 50, 0, 0], dtype=np.float32).reshape(-1, 1)
    write_lerobot_v3(
        CFG,
        fps=10,
        columns={
            "observation.robot_ft": ft,
            "observation.state.gripper": grip,
            "meta.rating": np.full((N_FRAMES, 1), 7, dtype=np.int64),
        },
        episode_meta={"tasks": ["put the cup on the plate, then retry if failed"]},
        video_keys=["observation.images.cam_f0001", "observation.images.cam_f0002"],
        stats={"observation.state.gripper": {"max": [110.0]}},
    )
    (CFG / "meta" / "rh20t_episodes.json").write_text(
        json.dumps([{"episode_index": 0, "folder": "task_0001_user_0001_scene_0006_cfg_0003"}])
    )


def write_long(root: Path, rating: int = 7) -> None:
    """A 16 s single-episode port (10 Hz, ``rating``) under ``root`` for the end-to-end test."""
    cfg = root / "rh20t_cfg3"
    shutil.rmtree(root, ignore_errors=True)
    n = 160
    t = np.arange(n) / 10.0
    plateau = np.interp(t, [0, 2, 4, 10, 12, 16], [0, 0, 10, 10, 0, 0]).astype(np.float32)
    rng = np.random.default_rng(1)
    ft = np.zeros((n, 6), dtype=np.float32)
    ft[:, 2] = plateau + rng.normal(0, 0.05, n).astype(np.float32)
    ft[:, 0] = plateau * 0.3 + rng.normal(0, 0.05, n).astype(np.float32)
    grip = (100 * np.interp(t, [0, 3, 4, 10, 11, 16], [0, 0, 1, 1, 0, 0])).astype(np.float32)
    write_lerobot_v3(
        cfg,
        fps=10,
        columns={
            "observation.robot_ft": ft,
            "observation.state.gripper": grip.reshape(-1, 1),
            "meta.rating": np.full((n, 1), rating, dtype=np.int64),
        },
        episode_meta={"tasks": ["put the cup on the plate"]},
        video_keys=["observation.images.cam_f0001", "observation.images.cam_f0002"],
        stats={"observation.state.gripper": {"max": [110.0]}},
        n_frames=n,
    )
    (cfg / "meta" / "rh20t_episodes.json").write_text(
        json.dumps([{"episode_index": 0, "folder": "task_0001_user_0001_scene_0006_cfg_0003"}])
    )


if __name__ == "__main__":
    main()
