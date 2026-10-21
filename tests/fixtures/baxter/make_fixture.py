"""Write one tiny synthetic Baxter episode in the contract documented in sources/baxter.py.

Run: ``uv run python tests/fixtures/baxter/make_fixture.py``. Output goes to ``data/`` next to
this file. The recorder clock starts at 50 s and the first video frame is at 50.1 s
(``video_t0_s``) so the clock shift is exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
from tests.fixtures._video import write_mp4  # noqa: E402

HERE = Path(__file__).parent
N = 10
FPS = 30
T0 = 50.1


def main() -> None:
    """Create the fixture."""
    root = HERE / "data"
    stem = "episode_000000"
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "force").mkdir(exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    ts = np.arange(N) / FPS
    grip = np.clip((np.arange(N) - 2) / 4, 0, 1)
    force_fr = np.stack([grip * 2, grip * 3, grip * 0, grip * 0], axis=1)
    pd.DataFrame(
        {
            "timestamp": ts,
            "frame_index": np.arange(N),
            "episode_index": np.zeros(N, dtype=int),
            "observation.state": [[0.0] * 15 for _ in range(N)],
            "observation.gripper_left": grip,
            "observation.gripper_right": np.zeros(N),
            "observation.force": [list(r) for r in force_fr],
        }
    ).to_parquet(root / "data" / "chunk-000" / f"{stem}.parquet")
    n_f = 60  # 200 Hz over 0.3 s
    tf = 50.0 + np.arange(n_f) / 200.0
    ramp = np.clip((np.arange(n_f) - 20) / 20, 0, 1)
    pd.DataFrame(
        {"t": tf, "L_a": ramp * 2, "L_b": ramp * 3, "R_a": ramp * 0.1, "R_b": ramp * 0.2}
    ).to_parquet(root / "force" / f"{stem}.parquet")
    for role in ("scene", "wrist_left", "wrist_right"):
        write_mp4(root / "videos" / f"observation.images.{role}" / f"{stem}.mp4", n=N, fps=FPS)
    row = {
        "episode_index": 0,
        "object": "crushed soda can",
        "firmness": "firm",
        "policy": {"grasp_height": 0.1, "lateral_offset": 0.0, "gripper_force": 30.0},
        "occlusion": False,
        "operator_notes": "synthetic",
        "recording_day": "2026-11-02",
        "calibration_version": "v0",
        "video_t0_s": T0,
    }
    (root / "meta" / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    labels = {
        "0": {
            "quality_1hz": [3, 3, 2, 2, 1, 3],
            "outcome": 2,
            "object": "soda can",
            "agreement": 0.8,
        }
    }
    (root / "labels.json").write_text(json.dumps(labels))


if __name__ == "__main__":
    main()
