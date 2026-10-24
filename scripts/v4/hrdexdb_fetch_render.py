"""Fetch HRDexDB Inspire trials (sensor files + labels) and pre-render FLORA-native camera frames.

For every trial of the requested embodiments: download grasp_result.json, the raw hand files the
adapter reads and raw/timestamps/{timestamp,frame_id}.npy into ``--raw`` (the adapter layout
``<hand>/<object>/<scene>/...``), and for every requested camera stream-decode the mp4 straight
from the Hub (no mp4 is stored), keep the outcome frame indices (labels.windows
outcome_frame_indices, the released terminal sampling: min(64, floor(10 s * fps)) frames, even,
first and last included) and write each frame bilinearly resized to ``--cell`` (default 384x288,
the released collage cell for a 4:3 view at about 100k pixels) as JPEG quality 95 4:4:4 under
``--frames/<episode_id>/<camera>/<index:05d>.jpg``. ``video.json`` beside them records the
stream's fps, header frame count, size and the kept indices; a camera absent upstream (HTTP 404)
is recorded as ``{"missing": true}`` and the trial keeps its other views. Resumable: finished
trials (``done.json``) are skipped. Labels are read only to be stored for the manifest builder.

Usage: python scripts/v4/hrdexdb_fetch_render.py --cams 22641005,22645029,... --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = "https://huggingface.co/datasets/HRDexDB/HRDexDB/resolve/98b8f8ea2e361a3c7df38b5d8f3f8395913773f2"
HAND_FILES = {
    "inspire_dftp": ["raw/hand/force.npy", "raw/hand/position.npy", "raw/hand/time.npy"],
    "inspire_f1": [
        "raw/hand/right_tactile.npy",
        "raw/hand/right_tactile_time.npy",
        "raw/hand/right_joint_states.npy",
        "raw/hand/right_joint_states_time.npy",
    ],
}
COMMON = ["grasp_result.json", "raw/timestamps/timestamp.npy", "raw/timestamps/frame_id.npy"]


def outcome_frame_indices(n_frames: int, fps: float) -> list[int]:
    """Exact copy of tactile_flora.labels.windows.outcome_frame_indices."""
    m = min(64, max(2, math.floor(10 * n_frames / fps + 1e-9)))
    m -= m % 2
    return [math.floor(i * (n_frames - 1) / (m - 1) + 0.5) for i in range(m)]


TRIES = 14


def backoff(k: int, err: Exception) -> None:
    """Sleep before retry ``k``: the Hub's anonymous limit is 3,000 resolves per fixed 5-minute
    window per IP (HTTP 429 with retry-after up to 300 s), so a 429 waits out the window.
    """
    if "429" in str(err) or "Too Many Requests" in str(err):
        time.sleep(60 + 30 * min(k, 8))
    else:
        time.sleep(min(5 * 2**k, 120))


def fetch(url: str, dest: Path, tries: int = TRIES) -> None:
    """Download ``url`` to ``dest`` atomically unless it is already there."""
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # rate limits and resets: back off and retry
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise RuntimeError(f"{url}: HTTP 404") from e  # absent upstream: no retries
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            backoff(k, e)


class _KeepRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D102
        return None


def cdn_url(url: str) -> str:
    """Spend one Hub resolve on ``url`` and return the CDN address it redirects to.

    The decoder opens a new HTTP request for every seek; against the resolve URL each one is
    another resolve (3,000 per 5 min per IP), against the CDN URL none is. Measured on the Colab
    render 2026-10-23: ~1,250 resolves in 90 s with 12 workers, so the limit, not the CPU, set the
    pace. A 404 (camera absent upstream) is raised at once with "404" in the message.
    """
    opener = urllib.request.build_opener(_KeepRedirect)
    for k in range(TRIES):
        try:
            with opener.open(urllib.request.Request(url, method="HEAD"), timeout=60):
                return url
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location")
            if e.code in (301, 302, 303, 307, 308) and loc:
                return urllib.parse.urljoin(url, loc)
            if e.code == 404:
                raise RuntimeError(f"{url}: HTTP 404") from e
            if k == TRIES - 1:
                raise RuntimeError(f"{url}: {e}") from e
            backoff(k, e)
        except Exception as e:  # resets and timeouts
            if k == TRIES - 1:
                raise RuntimeError(f"{url}: {e}") from e
            backoff(k, e)
    raise AssertionError


def probe_camera(url: str) -> dict:
    """Header only (fps, frame count, size): no frame is decoded or stored."""
    import av

    for k in range(TRIES):
        try:
            with av.open(cdn_url(url), options={"timeout": "60000000"}) as c:
                s = c.streams.video[0]
                fps = float(s.average_rate) if s.average_rate else 30.0
                n = int(s.frames)
                if n <= 0:
                    raise RuntimeError("stream has no frame count in its header")
                w, h = int(s.codec_context.width), int(s.codec_context.height)
            return {
                "fps": fps,
                "n_frames": n,
                "width": w,
                "height": h,
                "indices": outcome_frame_indices(n, fps),
                "probe_only": True,
            }
        except Exception as e:
            if k == TRIES - 1 or "HTTP 404" in str(e):
                raise RuntimeError(f"{url}: {e}") from e
            backoff(k, e)
    raise AssertionError


DECODE_THREADS = 0  # 0 = PyAV AUTO; set by --decode-threads (oversubscription costs throughput)


def render_camera(url: str, out: Path, cell: tuple[int, int]) -> dict:
    """Decode the stream once, keep the outcome frame indices resized to ``cell`` as JPEG."""
    import av
    from PIL import Image

    for k in range(TRIES):
        try:
            with av.open(cdn_url(url), options={"timeout": "60000000"}) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                if DECODE_THREADS:
                    s.codec_context.thread_count = DECODE_THREADS
                fps = float(s.average_rate) if s.average_rate else 30.0
                header = int(s.frames)
                n = header if header > 0 else None
                if n is None:
                    raise RuntimeError("stream has no frame count in its header")
                want = outcome_frame_indices(n, fps)
                wanted = sorted(set(want))
                out.mkdir(parents=True, exist_ok=True)
                it = iter(wanted)
                target = next(it)
                pos = -1
                width = height = 0
                for pos, frame in enumerate(c.decode(s)):
                    if pos != target:
                        continue
                    img = frame.to_image()
                    width, height = img.size
                    if img.size != cell:
                        img = img.resize(cell, resample=Image.Resampling.BILINEAR)
                    img.convert("RGB").save(out / f"{pos:05d}.jpg", quality=95, subsampling=0)
                    target = next(it, None)
                    if target is None:
                        break
                if target is not None:
                    raise RuntimeError(
                        f"decoded {pos + 1} frames, header says {n}; missing {target}"
                    )
            return {"fps": fps, "n_frames": n, "width": width, "height": height, "indices": want}
        except Exception as e:
            if k == TRIES - 1 or "HTTP 404" in str(e):
                raise RuntimeError(f"{url}: {e}") from e
            backoff(k, e)
    raise AssertionError


def do_trial(
    ep: str,
    cams: list[str],
    raw: Path,
    frames: Path,
    cell: tuple[int, int],
    probe_only: bool = False,
) -> str:
    done = frames / ep / ("probe.json" if probe_only else "done.json")
    if done.is_file():
        return f"skip {ep}"
    t0 = time.time()
    hand = ep.split("/")[0]
    for rel in COMMON + HAND_FILES[hand]:
        fetch(f"{REPO}/{ep}/{rel}", raw / ep / rel)
    meta = {}
    for cam in cams:
        url = f"{REPO}/{ep}/vid/{cam}.mp4"
        try:
            meta[cam] = probe_camera(url) if probe_only else render_camera(url, frames / ep / cam, cell)
        except RuntimeError as e:
            if "404" not in str(e):
                raise
            # the camera does not exist upstream for this trial: record it, keep the other views
            meta[cam] = {"missing": True}
    (frames / ep).mkdir(parents=True, exist_ok=True)
    if not probe_only or not (frames / ep / "video.json").is_file():
        (frames / ep / "video.json").write_text(json.dumps(meta))
    done.write_text(json.dumps({"seconds": round(time.time() - t0, 1)}))
    return f"ok {ep} {time.time() - t0:.0f}s"


def main() -> None:
    """Process every requested trial in a process pool (resumable)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/raw/hrdexdb/metadata/episodes.parquet")
    ap.add_argument("--hands", default="inspire_dftp,inspire_f1")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--raw", default="data/raw/hrdexdb")
    ap.add_argument("--frames", default="data/frames/hrdexdb")
    ap.add_argument("--cell", default="384x288")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated episode ids (overrides --hands)")
    ap.add_argument("--probe-only", action="store_true", help="video.json from headers, no frames")
    ap.add_argument("--decode-threads", type=int, default=0, help="threads per decoder (0 = auto)")
    ap.add_argument("--ids-file", default="", help="file with one episode id per line, in order")
    a = ap.parse_args()
    import pandas as pd

    raw, frames = Path(a.raw), Path(a.frames)
    for name in ("episodes", "cameras", "objects", "embodiments", "pairings"):
        fetch(f"{REPO}/metadata/{name}.parquet", raw / "metadata" / f"{name}.parquet")
    eps = pd.read_parquet(a.episodes)
    global DECODE_THREADS
    DECODE_THREADS = a.decode_threads
    if a.ids_file:
        ids = [x.strip() for x in Path(a.ids_file).read_text().splitlines() if x.strip()]
    elif a.only:
        ids = a.only.split(",")
    else:
        ids = sorted(eps[eps.embodiment.isin(a.hands.split(","))].episode_id)
    if a.limit:
        ids = ids[: a.limit]
    cams = a.cams.split(",")
    w, h = (int(v) for v in a.cell.split("x"))
    print(f"{len(ids)} trials x {len(cams)} cameras -> {frames}", flush=True)
    t0, n_ok, n_fail = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(do_trial, ep, cams, raw, frames, (w, h), a.probe_only): ep for ep in ids}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
                n_ok += 1
            except Exception as e:
                msg = f"FAIL {futs[f]}: {str(e)[:300]}"
                n_fail += 1
            if k % 10 == 0 or msg.startswith("FAIL"):
                print(
                    f"[{k}/{len(ids)} {time.time() - t0:.0f}s ok={n_ok} fail={n_fail}] {msg}",
                    flush=True,
                )
    print(f"DONE ok={n_ok} fail={n_fail} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
