"""Write one tiny synthetic SharpaDex season in the real layout (run from repo root)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tests.fixtures.lerobot_writer import N_FRAMES, write_lerobot_v3

HERE = Path(__file__).parent
EXPORT = HERE / "pick_cup" / "season_7_train" / "lerobot_v3.0"
CLOSURE_DIM = 40  # synthetic right_index_mcp position in the 65-D state


def main() -> None:
    shutil.rmtree(HERE / "pick_cup", ignore_errors=True)
    plateau = np.array([0, 0, 1, 3, 5, 5, 5, 3, 1, 0], dtype=np.float32)
    tactile = np.zeros((N_FRAMES, 60), dtype=np.float32)
    tactile[:, 32] = plateau  # right thumb fz
    tactile[:, 38] = plateau * 0.5  # right index fz
    state = np.zeros((N_FRAMES, 65), dtype=np.float32)
    state[:, CLOSURE_DIM] = np.array([0, 0, 0.2, 0.6, 1, 1, 1, 0.6, 0.2, 0], dtype=np.float32) * 1.5
    names = [f"joint_{i}" for i in range(65)]
    names[CLOSURE_DIM] = "right_index_mcp"
    sub = np.array([0] * 5 + [1] * 5)
    write_lerobot_v3(
        EXPORT,
        fps=30,
        columns={
            "observation.tactile": tactile,
            "observation.state": state,
            "subtask_index": sub,
            "task_index": np.zeros(N_FRAMES, dtype=np.int64),
        },
        episode_meta={"tasks": ["pick up the cup, retry on failure"]},
        video_keys=[
            "observation.images.head_left",
            "observation.images.wrist_left",
            "observation.images.wrist_right",
        ],
        feature_names={"observation.state": names},
    )
    pq.write_table(
        pa.table(
            {
                "subtask_index": [0, 1],
                "text": ["reach the cup", "grasp the cup"],
                "task_skill": ["reach", "grasp"],
            }
        ),
        EXPORT / "meta" / "subtasks.parquet",
    )


if __name__ == "__main__":
    main()
