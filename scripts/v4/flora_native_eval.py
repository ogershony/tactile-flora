"""Score clips with the released (or a merged fine-tuned) FLORA through NVIDIA's own batch_utils.

The prompt, collage planning, frame-clock metadata, structured decoding and expected-score
computation are batch_utils' (imported from the model snapshot, never copied); only the frames
come from our pre-rendered cache (scripts/v4/hrdexdb_fetch_render.py), already at the released
collage cell size. With ``force_block`` in a job the block is placed after the video and a
force sentence is appended to the system prompt, exactly as tactile_flora.prompts.flora does;
without it the messages are batch_utils.core.terminal_messages verbatim.

Jobs JSONL: {"id", "task", "frames_dir", "cams": [...collage order...], "force_block"?: str,
"label"?: int, "group"?: str, "cell96"?: true}. ``cell96`` reproduces the legacy repo input (each
view thumbnailed into a black-padded 96x96 cell) for the before/after comparison.

Usage: python scripts/v4/flora_native_eval.py --model /content/models/Flora-Rewarder \
    --jobs jobs.jsonl --output out.jsonl [--max-num-seqs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

FORCE_SENTENCE_COMPACT = (
    "A force summary from the robot's fingertip or wrist force sensors follows the clip: first "
    "contact, peak force level (F0 none to F4 crush-level, relative to this object's typical "
    "grasp), any contact loss, sharp force drops and the final level, and a force-only "
    "classifier's probability of success when one is available."
)


def load_view(
    frames_dir: Path, cam: str, indices: list[int], rendered: list[int]
) -> list[np.ndarray]:
    """The cached frames of one camera at ``indices``.

    Indices come from the first camera; a camera whose stream is a frame shorter (8 of 444
    rendered trials) was rendered at its own indices, so an index it lacks reads the nearest
    rendered one within one frame, as video/readers.FrameCacheReader does.
    """
    have = np.asarray(sorted(rendered))
    out = []
    for i in indices:
        j = int(have[np.abs(have - i).argmin()])
        if abs(j - i) > 1:
            raise FileNotFoundError(f"{frames_dir / cam}: no rendered frame within 1 of {i}")
        out.append(np.asarray(Image.open(frames_dir / cam / f"{j:05d}.jpg").convert("RGB")))
    return out


def cell96(frame: np.ndarray) -> np.ndarray:
    """The legacy repo input: thumbnail into a black-padded 96 px cell."""
    img = Image.fromarray(frame)
    img.thumbnail((96, 96))
    cell = np.zeros((96, 96, 3), dtype=np.uint8)
    cell[: img.height, : img.width] = np.asarray(img)
    return cell


def build_clip(core, job: dict) -> tuple[np.ndarray, dict, int]:  # noqa: ANN001
    """(T, H, W, 3) collage frames, the frame-clock metadata and the frame count of one job."""
    d = Path(job["frames_dir"])
    meta = json.loads((d / "video.json").read_text())
    cams = list(job["cams"])
    ref = meta[cams[0]]
    indices = core.terminal_frame_indices(ref["n_frames"], ref["fps"])
    if [int(i) for i in indices] != [int(i) for i in ref["indices"]]:
        # rounding differs only at exact .5 ties; take the cached frame nearest each index
        cached = np.asarray(ref["indices"])
        indices = [int(cached[np.abs(cached - i).argmin()]) for i in indices]
    views = [load_view(d, c, indices, meta[c]["indices"]) for c in cams]
    if job.get("cell96"):
        views = [[cell96(f) for f in v] for v in views]
        rows, cols = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}[len(views)]
        frames = []
        for t in range(len(indices)):
            canvas = np.zeros((rows * 96, cols * 96, 3), dtype=np.uint8)
            for k, v in enumerate(views):
                r, c = divmod(k, cols)
                canvas[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96] = v[t]
            frames.append(canvas)
    else:
        sources = tuple((meta[c]["width"], meta[c]["height"]) for c in cams)
        rects, width, height = core.plan_collage(sources)
        frames = []
        for t in range(len(indices)):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for v, (x, y, w, h) in zip(views, rects, strict=True):
                f = v[t]
                if f.shape[:2] != (h, w):
                    f = np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.BILINEAR))
                canvas[y : y + h, x : x + w] = f
            frames.append(canvas)
    md = core.video_metadata(indices, ref["fps"], ref["n_frames"])
    return np.stack(frames), md, len(indices)


def messages_for(core, job: dict) -> list[dict]:  # noqa: ANN001
    """batch_utils' terminal messages, with the force block after the video when given."""
    msgs = core.terminal_messages(job["task"], with_rationale=False)
    block = job.get("force_block")
    if not block:
        return msgs
    system = msgs[0]["content"] + "\n" + job.get("force_sentence", FORCE_SENTENCE_COMPACT)
    content = list(msgs[1]["content"])
    # content = [text before video, video, text after]; the block goes right after the video
    after = content[-1]
    assert after["type"] == "text"
    content[-1] = {"type": "text", "text": block + "\n\n" + after["text"]}
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def main() -> None:
    """Score every job not already in --output (resumable)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument(
        "--batch-utils", default=None, help="dir holding batch_utils/ (default: --model)"
    )
    a = ap.parse_args()
    sys.path.insert(0, a.batch_utils or a.model)
    from batch_utils import core  # type: ignore[import-not-found]
    from batch_utils.engine import COLLAGE_VIDEO_SIZE, load_processor, render  # type: ignore

    jobs = [json.loads(x) for x in a.jobs.read_text().splitlines() if x.strip()]
    done = set()
    if a.output.exists():
        done = {json.loads(x)["id"] for x in a.output.read_text().splitlines() if x.strip()}
    jobs = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs to score ({len(done)} already in {a.output})", flush=True)
    if not jobs:
        return
    processor = load_processor(a.model)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=a.model,
        max_model_len=a.max_model_len,
        max_num_seqs=a.max_num_seqs,
        gpu_memory_utilization=a.gpu_memory_utilization,
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs={"do_sample_frames": False, "size": COLLAGE_VIDEO_SIZE},
        max_logprobs=core.TERMINAL_TOP_LOGPROBS,
        seed=0,
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=core.terminal_max_tokens(False),
        logprobs=core.TERMINAL_TOP_LOGPROBS,
        structured_outputs=StructuredOutputsParams(regex=core.terminal_regex(False)),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with a.output.open("a") as out:
        for s in range(0, len(jobs), a.max_num_seqs):
            batch = jobs[s : s + a.max_num_seqs]
            reqs, extra = [], []
            for j in batch:
                frames, md, n = build_clip(core, j)
                prompt = render(processor, messages_for(core, j), thinking=False)
                reqs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"video": (frames, md)},
                        "mm_processor_kwargs": {"do_sample_frames": False},
                    }
                )
                extra.append((n, frames.shape))
            outs = llm.generate(reqs, params, use_tqdm=False)
            for j, o, (n, shape) in zip(batch, outs, extra, strict=True):
                comp = o.outputs[0]
                score, _ = core.parse_terminal(comp.text, False)
                positions = [
                    (
                        tok.decode([t]),
                        float(c[t].logprob),
                        [(tok.decode([i]), float(v.logprob)) for i, v in c.items()],
                    )
                    for t, c in zip(comp.token_ids, comp.logprobs, strict=True)
                ]
                probs, es = core.score_probabilities(positions, score)
                rec = {k: j[k] for k in ("id", "label", "group", "cams") if k in j}
                rec.update(
                    {
                        "score": score,
                        "expected_score": round(es, 5),
                        "probabilities": {k: round(v, 5) for k, v in probs.items()},
                        "frames": n,
                        "collage": list(shape[1:3]),
                        "prompt_tokens": len(o.prompt_token_ids),
                        "force": bool(j.get("force_block")),
                    }
                )
                out.write(json.dumps(rec) + "\n")
            out.flush()
            k = min(s + len(batch), len(jobs))
            print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
