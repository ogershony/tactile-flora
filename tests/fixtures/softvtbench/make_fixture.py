"""Write one tiny synthetic SoftVTBench subset (object-soft, one task, two demos).

Run: ``uv run python tests/fixtures/softvtbench/make_fixture.py``. Output goes to ``data/``
next to this file. The language contains an outcome word on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))
from tests.fixtures._video import write_mp4  # noqa: E402

HERE = Path(__file__).parent
N = 10
FPS = 20
SUBSET = "object-soft"
TASK = "LIBERO_Object_pick_egg"


def main() -> None:
    """Create the fixture."""
    root = HERE / "data" / SUBSET
    h5_dir = root / "libero_object" / TASK / "replayed_demos"
    h5_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = root / "video_datasets" / TASK / "videos"
    ramp = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32)
    with h5py.File(h5_dir / f"{TASK}_en_replayed_demo.hdf5", "w") as f:
        data = f.create_group("data")
        for i in range(2):
            demo = data.create_group(f"demo_{i}")
            demo.attrs.update(
                asset_name="egg",
                demo_id=i,
                language="pick up the egg successfully",
                success=True,
                soft_task_kind="pick",
                task_id=3,
                task_suite="libero_object",
            )
            obs = demo.create_group("obs")
            sign = np.array([1.0, -1.0], dtype=np.float32)
            obs["finger_force"] = (ramp[:, None] * np.array([10.0, 20.0]) * sign).astype(np.float32)
            obs["gripper_close_norm"] = ramp
            obs["gripper_width"] = 1 - ramp
            obs["gripper_net_force"] = np.zeros((N, 1, 2, 3), dtype=np.float32)
            obs["fem_deformation_max"] = ramp * 0.01
            obs["fem_deformation_rms"] = ramp * 0.005
            kab = ramp * np.array([0, 0, 4, 8, 12, 12, 6, 3, 0, 0], dtype=np.float32) / 12 * 12
            obs["fem_kabsch_max_pct"] = kab
            obs["fem_kabsch_rms_pct"] = kab / 2
            write_mp4(vid_dir / f"demo_{i}_agentview_rgb.mp4", n=N, fps=FPS)
            write_mp4(vid_dir / f"demo_{i}_eye_in_hand_rgb.mp4", n=N, fps=FPS)


if __name__ == "__main__":
    main()
