"""Write one tiny synthetic AIRoA MoMa episode in the real layout (run from repo root)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from tests.fixtures.lerobot_writer import N_FRAMES, write_lerobot_v3

HERE = Path(__file__).parent


def main() -> None:
    for d in ("data", "meta", "videos"):
        shutil.rmtree(HERE / d, ignore_errors=True)
    plateau = np.array([0, 0, 2, 6, 10, 10, 10, 6, 2, 0], dtype=np.float32)
    wrench = np.zeros((N_FRAMES, 6), dtype=np.float32)
    wrench[:, 2] = plateau
    state = np.zeros((N_FRAMES, 8), dtype=np.float32)
    state[:, 5] = np.array([0, 0, 0.2, 0.6, 1, 1, 1, 0.6, 0.2, 0], dtype=np.float32)
    write_lerobot_v3(
        HERE,
        fps=10,
        columns={"observation.wrist.wrench": wrench, "observation.state": state},
        episode_meta={
            "tasks": ["pick up the cup and do not fail"],
            "uuid": "ep-2026-03-14-abc",
            "task_type": "short_horizon",
            "task_success": True,
            "primitive_action": "pick",
            "label": "pick cup",
            "hsr_id": "hsr12",
            "version": "1.1",
        },
        video_keys=["observation.image.head", "observation.image.hand"],
        stats={"observation.state": {"min": [0.0] * 8, "max": [1.0] * 8}},
    )


if __name__ == "__main__":
    main()
