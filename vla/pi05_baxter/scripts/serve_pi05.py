"""HTTP policy server for the Baxter pi0.5 checkpoint (stdlib only, one request at a time).

Run it on the machine with the GPU; the Baxter runner calls it over the network:

    .venv/bin/python scripts/serve_pi05.py runs/<run>/checkpoints/<step>/pretrained_model --port 8765

POST /act, JSON body:
    {"head": "<base64 JPEG or PNG, RGB, the head camera as recorded (640x360)>",
     "wrist": "<base64 JPEG or PNG, the left wrist camera>",
     "state": [left_s0, left_s1, left_e0, left_e1, left_w0, left_w1, left_w2, left_gripper],
     "task": "pick up the red cube",          (optional, this is the default)
     "seed": 0}                                (optional, fixes the flow-matching noise)
returns
    {"actions": [[8 floats] x 10], "dt_s": 0.1, "latency_ms": ...}

`actions` are ABSOLUTE joint targets in the dataset's units (rad, gripper 0..1 scale of the
demos), one every 0.1 s starting at the current frame: the model predicts offsets from `state`
and the server adds `state` back. Execute a prefix of the chunk (for example 5 steps) and query
again; actions[0] is close to `state` by construction.
GET /health returns {"ok": true, "ckpt": ...}.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/baxter-cube-pick-left-v0-rel10"
RENAME = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}
TASK = "pick up the red cube"


class Pi05:
    def __init__(self, ckpt: Path):
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = "cuda"
        meta = LeRobotDatasetMetadata("OG7407/baxter-cube-pick-left-v0", root=DATA)
        self.policy = make_policy(cfg=cfg, ds_meta=meta, rename_map=RENAME)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.ckpt = str(ckpt)
        self.lock = Lock()

    @staticmethod
    def _image(b64: str) -> torch.Tensor:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W) in [0, 1]

    def act(self, req: dict) -> dict:
        state = torch.tensor(req["state"], dtype=torch.float32)
        if state.shape != (8,):
            raise ValueError(f"state must have 8 values, got {tuple(state.shape)}")
        obs = {
            "observation.images.head": self._image(req["head"]),
            "observation.images.wrist": self._image(req["wrist"]),
            "observation.state": state,
            "task": req.get("task", TASK),
        }
        with self.lock:
            if "seed" in req:
                torch.manual_seed(int(req["seed"]))
            t0 = time.perf_counter()
            batch = self.pre(obs)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = self.policy.predict_action_chunk(batch)
            chunk = self.post(chunk)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
        actions = chunk.float().cpu().numpy().reshape(-1, chunk.shape[-1])[:, :8]
        return {"actions": actions.round(5).tolist(), "dt_s": 0.1, "latency_ms": round(dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    model = Pi05(Path(args.ckpt).resolve())

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": model.ckpt})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/act":
                self._send(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, model.act(req))
            except Exception as e:  # report the error to the runner instead of dropping the socket
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *a):
            pass

    print(f"serving {model.ckpt} on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
