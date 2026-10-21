"""Write one tiny synthetic HapticWAM episode in the real loose layout (run from repo root)."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from zarr.dtype import VariableLengthBytes

HERE = Path(__file__).parent
TASK = "egg_fail"
EP = "ep_egg_fail_20260801_000"
T0 = 1000.0  # master-clock seconds of the first scene frame


def _jpeg(i: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (20 * i % 256, 80, 160)).save(buf, format="JPEG")
    return buf.getvalue()


def _group(ep: Path, name: str, data: np.ndarray, ts: np.ndarray) -> None:
    g = zarr.open_group(str(ep / name), mode="w", zarr_format=2)
    g.create_array("data", data=data)
    g.create_array("ts", data=ts)


def main() -> None:
    for d in ("tasks", "manifests"):
        shutil.rmtree(HERE / d, ignore_errors=True)
    ep = HERE / "tasks" / TASK / EP
    ep.mkdir(parents=True)
    scene_ts = T0 + np.arange(10) / 15.0
    g = zarr.open_group(str(ep / "camera_scene_color"), mode="w", zarr_format=2)
    frames = np.empty(10, dtype=object)
    for i in range(10):
        frames[i] = _jpeg(i)
    arr = g.create_array("data", shape=(10,), dtype=VariableLengthBytes(), chunks=(10,))
    arr[:] = frames
    g.create_array("ts", data=scene_ts)
    ft = T0 - 0.05 + np.arange(5) / 6.0
    plateau = np.array([0.0, 2.0, 10.0, 10.0, 1.0])
    for name, k in (("tactile_left_wrench", 1.0), ("tactile_right_wrench", 0.8)):
        w = np.zeros((5, 6))
        w[:, 2] = -plateau * k  # compressive Fz is negative; adapter takes |Fz|
        _group(ep, name, w, ft)
    gt = T0 - 0.02 + np.arange(40) / 95.0
    grip = np.zeros((40, 2))
    grip[:, 0] = np.clip((np.arange(40) - 10) / 10.0, 0, 1)
    _group(ep, "gripper", grip, gt)
    meta = {
        "task": TASK, "text": "Pick up the egg gently, failure demo: over-squeeze",
        "operator": "op1", "tags": ["deliberate_failure"], "success": True, "status": "finalized",
    }  # fmt: skip
    (ep / "meta.json").write_text(json.dumps(meta))
    row = {
        "episode": EP, "task": TASK, "success": True, "failure_demo": True,
        "tactile_contact": True, "split": "val", "session": "20260801", "operator": "op1",
        "duration_s": 0.6, "peak_force_N": 10.0, "max_contact_mm2": 55.0,
        "path": f"tasks/{TASK}/{EP}",
    }  # fmt: skip
    (HERE / "manifests").mkdir()
    (HERE / "manifests" / "all.jsonl").write_text(json.dumps(row) + "\n")


def _long_profile(t: np.ndarray, peak: float) -> np.ndarray:
    """Free space for 2 s, ramp up, hold ``peak`` between 4 s and 10 s, release by 12 s."""
    return np.interp(t, [0, 2, 4, 10, 12, 16], [0, 0, peak, peak, 0, 0])


def write_long(root: Path) -> None:
    """Two 16 s episodes (a failure demo and a success demo of the same object) under ``root``.

    Used by the end-to-end verify and manifest test: long enough for normalization, sensor health
    and dense windows. The failure demo squeezes a little harder (peak 7 vs 6).
    """
    shutil.rmtree(root, ignore_errors=True)
    rows = []
    rng = np.random.default_rng(0)
    for task, ep_name, peak, failure in (
        ("egg_fail", "ep_egg_fail_20260801_000", 7.0, True),
        ("egg", "ep_egg_20260801_001", 6.0, False),
    ):
        ep = root / "tasks" / task / ep_name
        ep.mkdir(parents=True)
        n = 240
        scene_ts = T0 + np.arange(n) / 15.0
        g = zarr.open_group(str(ep / "camera_scene_color"), mode="w", zarr_format=2)
        frames = np.empty(n, dtype=object)
        for i in range(n):
            frames[i] = _jpeg(i)
        arr = g.create_array("data", shape=(n,), dtype=VariableLengthBytes(), chunks=(n,))
        arr[:] = frames
        g.create_array("ts", data=scene_ts)
        ft = T0 + np.arange(800) / 50.0
        for name, k in (("tactile_left_wrench", 1.0), ("tactile_right_wrench", 0.8)):
            w = np.zeros((800, 6))
            w[:, 2] = -(_long_profile(ft - T0, peak * k) + rng.normal(0, 0.05, 800))
            _group(ep, name, w, ft)
        gt = T0 + np.arange(1520) / 95.0
        grip = np.zeros((1520, 2))
        grip[:, 0] = np.interp(gt - T0, [0, 3, 4, 10, 11, 16], [0, 0, 1, 1, 0, 0])
        _group(ep, "gripper", grip, gt)
        meta = {"task": task, "text": "Pick up the egg", "operator": "op1", "success": not failure}
        (ep / "meta.json").write_text(json.dumps(meta))
        rows.append(
            {
                "episode": ep_name, "task": task, "success": not failure,
                "failure_demo": failure, "tactile_contact": True, "split": "train",
                "session": "20260801", "operator": "op1", "duration_s": 16.0,
                "peak_force_N": peak, "max_contact_mm2": 55.0, "path": f"tasks/{task}/{ep_name}",
            }
        )  # fmt: skip
    (root / "manifests").mkdir()
    (root / "manifests" / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
