"""Read every clip of a manifest through the training/eval clip path (CPU) and list failures.

Usage: sweep_clips.py <manifest> [--blank]  (--blank: the video_blank ablation's mid-gray clips)
"""
import sys
import time
from pathlib import Path

from tactile_flora.config import ForceConfig, load_config
from tactile_flora.eval import harness
from tactile_flora.eval.inference import clip_array
from tactile_flora.prompts.flora import build_prompt, force_sentence_for
from tactile_flora.video.readers import SourceRootReader

BLANK = "--blank" in sys.argv
f = load_config(Path("configs/force.v4.yaml"), ForceConfig)
prov = harness.make_provider(Path(sys.argv[1]), {"outcome"}, f, Path("configs/sources"), None, Path("data"))
r = SourceRootReader(Path("configs/sources"))
bad, n, t0 = [], 0, time.time()
for k, x in harness._load_batches(prov).items():
    ep = x.episode
    cam = ep.cameras.get("scene") or next(iter(ep.cameras.values()))
    for w in x.windows:
        v = build_prompt(w, None, task=ep.task, camera_roles=tuple(ep.cameras), fps=cam.fps,
                         frame_indices=harness._frame_indices(w, cam.fps, cam.n_frames),
                         force_sentence=force_sentence_for(f.render), collage="flora",
                         video_blank=BLANK).video
        try:
            clip_array(ep, v, r)
        except Exception as e:  # noqa: BLE001
            bad.append((k, repr(e)[:200]))
        n += 1
        if n % 200 == 0:
            print(n, len(bad), round(time.time() - t0), flush=True)
print("DONE clips", n, "bad", len(bad), "seconds", round(time.time() - t0), flush=True)
for b in bad[:30]:
    print("BAD", b, flush=True)
