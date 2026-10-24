"""Minimal client for serve_pi05.py, plus a smoke test against a recorded demo frame.

In the Baxter runner:

    from pi05_client import query
    actions = query("http://<gpu-host>:8765", head_rgb, wrist_rgb, joints8)   # (10, 8) absolute

head_rgb / wrist_rgb are HxWx3 uint8 RGB arrays (the cameras as recorded, 640x360); joints8 is
[left_s0, left_s1, left_e0, left_e1, left_w0, left_w1, left_w2, left_gripper].

Smoke test (server running):  .venv/bin/python scripts/pi05_client.py http://localhost:8765
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.request

import numpy as np
from PIL import Image


def _b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def query(url: str, head_rgb: np.ndarray, wrist_rgb: np.ndarray, joints8, task: str = "pick up the red cube",
          timeout_s: float = 5.0) -> np.ndarray:
    body = json.dumps({"head": _b64(head_rgb), "wrist": _b64(wrist_rgb),
                       "state": [float(v) for v in joints8], "task": task}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/act", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return np.asarray(out["actions"], dtype=np.float32)


if __name__ == "__main__":
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
    root = Path(__file__).resolve().parents[1] / "data/baxter-cube-pick-left-v0-rel10"
    ds = LeRobotDataset("OG7407/baxter-cube-pick-left-v0", root=root, episodes=[14], video_backend="pyav",
                        delta_timestamps={"action": [k / 10 for k in range(10)]})
    for i in (0, 20, 40):
        item = ds[i]
        to_rgb = lambda t: (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  # noqa: E731
        t0 = time.perf_counter()
        a = query(url, to_rgb(item["observation.images.head"]), to_rgb(item["observation.images.wrist"]),
                  item["observation.state"].tolist())
        rt = (time.perf_counter() - t0) * 1000
        gt = item["action"].numpy()
        print(f"frame {i}: round trip {rt:.0f} ms, joint MAE vs demo {np.abs(a[:, :7] - gt[:, :7]).mean():.4f} rad, "
              f"first-step jump {np.abs(a[0, :7] - item['observation.state'].numpy()[:7]).max():.4f} rad")
