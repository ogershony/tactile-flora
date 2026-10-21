"""Write one tiny synthetic HRDexDB trial (allegro_v5) in the real on-disk layout.

Run: ``uv run python tests/fixtures/hrdexdb/make_fixture.py``. Output goes to ``data/`` next to
this file. The object name contains an outcome word on purpose (adapter must strip it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))
from tests.fixtures._video import write_mp4  # noqa: E402

HERE = Path(__file__).parent
N = 10
FPS = 30


def main() -> None:
    """Create the fixture."""
    trial = HERE / "data" / "allegro_v5" / "retry_mug" / "20260101_000000"
    (trial / "raw" / "hand").mkdir(parents=True, exist_ok=True)
    (trial / "raw" / "arm").mkdir(parents=True, exist_ok=True)
    (trial / "raw" / "timestamps").mkdir(parents=True, exist_ok=True)
    (trial / "grasp_result.json").write_text(
        json.dumps({"grasp_success": True, "human_paired_episode": 0})
    )
    # tactile int32 (T, 4): fingertip 1 loaded most, fingertip 3 second, plateau from frame 3.
    ramp = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=np.int32)
    tactile = np.stack([ramp * 10, ramp * 500, ramp * 5, ramp * 200], axis=1).astype(np.int32)
    np.save(trial / "raw" / "hand" / "tactile.npy", tactile)
    t = 100.0 + np.arange(N) / FPS  # sensor clock starts at 100 s
    np.save(trial / "raw" / "hand" / "time.npy", t)
    # position (T2, 16) on a different time base (offset by 10 ms, 8 samples)
    n2 = 8
    pos = np.linspace(0.0, 2.0, n2)[:, None] * np.ones((1, 16))
    np.save(trial / "raw" / "hand" / "position.npy", pos)
    np.save(trial / "raw" / "hand" / "position_time.npy", 100.01 + np.arange(n2) / 24.0)
    np.save(trial / "raw" / "arm" / "position.npy", np.zeros((N, 6)))
    write_mp4(trial / "vid" / "23263780.mp4", N, FPS)
    write_mp4(trial / "vid" / "23263781.mp4", N, FPS)


if __name__ == "__main__":
    main()
