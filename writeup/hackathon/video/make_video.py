"""Build the Tactile FLORA hackathon video (1920x1080, 30 fps, H.264 + AAC).

Every number drawn on screen is parsed from the fact sheet (facts.md) or read from the Pi0.5
results file; nothing is typed here. The narration lives in narration.json; decimals in it are
checked against the fact sheet before anything is rendered.

Voice: ElevenLabs text to speech, key read at run time from ~/.config/voxclip/env
(ELEVENLABS_API_KEY). The key is never printed or stored. Audio is cached per segment text, so a
re-run only pays for segments whose text changed. With no key the video is built silent, timed at
150 words per minute, with the same on-screen captions; the next run with a key adds the voice.

Footage: the public Baxter datasets OG7407/baxter-cube-pick-left-v0 (kinesthetic demos) and
OG7407/baxter-cube-pick-left-rollouts (ACT rollouts with 200 Hz fingertip pads), two HRDexDB
comparison clips rendered by writeup/make_videos.py on origin/main, and the v4 per-clip scores
(docs/v4/results/per_clip_scores.csv).

    python make_video.py                  # full render
    python make_video.py --stills 5,40,90 # PNG frames at those seconds, for checking layout
    python make_video.py --plan           # print the timeline and exit

Run with an interpreter that has av, numpy, pandas, pillow, matplotlib, requests and
huggingface_hub.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FACTS_DEFAULT = HERE / "facts.md"
BUILD = Path(os.environ.get("TF_VIDEO_BUILD", Path.home() / ".cache/tactile_flora_video"))
CACHE = BUILD / "cache"
TTS_DIR = BUILD / "tts"
PI05 = Path(os.environ.get("PI05_RUNS", REPO / "vla/pi05_baxter/runs"))
RESULTS_JSON = PI05 / "final_results.json"
HELDOUT_NPZ = PI05 / "heldout_pred.npz"
OUT_MP4 = HERE / "tactile_flora_hackathon.mp4"
OUT_POSTER = HERE / "poster.png"
ENV_FILE = Path.home() / ".config/voxclip/env"

W, H, FPS = 1920, 1080, 30
AUDIO_RATE = 48000
MAX_BYTES = 19_000_000  # final MP4 cap (the deck embeds it under 20 MB)
CRF_STEPS = (20, 22, 24, 26, 28)
LEAD, TAIL, XFADE = 0.45, 0.55, 0.4  # s: silence before / after each narration segment, crossfade
WPM_FALLBACK = 150

# Palette. Series colours follow the v4 results page; the rest is chosen for a dark stage.
BG = (11, 12, 15)
PANEL = (21, 23, 28)
LINE = (52, 56, 64)
FG = (241, 239, 234)
MUTED = (150, 155, 164)
FAINT = (98, 103, 112)
ACCENT = (232, 72, 58)  # Baxter red; also marks failures (always with a label or icon)
SERIES = {"force": "#2a78d6", "control": "#eb6834", "released": "#1baf7a", "classifier": "#eda100"}
PAD_A, PAD_B = "#ff7a59", "#b9a2ff"  # two left-finger pads
PI_COLOR = "#4cc9f0"  # Pi0.5 predictions
HOLD_COLOR = "#8a8f99"


def hexrgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ----------------------------------------------------------------------------------------------
# Facts and results
# ----------------------------------------------------------------------------------------------


class Facts:
    """Values parsed from facts.md. A missing pattern is an error, never a silent default."""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text()
        t = self.text

        def m(pat: str):
            r = re.search(pat, t, re.S)
            if not r:
                raise SystemExit(f"facts.md: pattern not found: {pat}")
            return r.groups()

        self.event, = m(r"- (GPT-6 Astra Robotics Hardware Hackathon)")
        self.demo_date, = m(r"Demo (\d{4}-\d{2}-\d{2})")
        self.repo, = m(r"Repo: (https://github\.com/\S+)")
        g = m(r"(\d+) live rollouts.*?\): (\d+) success, (\d+) failures\. (\d+) of the \d+ failures were safety aborts")
        self.n_runs, self.n_success, self.n_fail, self.n_jump = map(int, g)
        self.jump_lo, self.jump_hi = m(r"starts ([\d.]+)-([\d.]+) rad from the arm")
        self.max_jump, = m(r"--max-jump ([\d.]+)")
        self.pad_rest, self.pad_peak_a, self.pad_peak_b = m(
            r"baseline about (\d+); the successful grasp \(ep 0\) peaks at (\d+) / (\d+)")
        self.pad_hz, = m(r"read at (\d+) Hz")
        self.n_pads, = m(r"(\d+) force-sensitive resistor \(FSR\) pads")
        self.n_demos, self.demo_frames, self.demo_hz = m(r"(\d+) kinesthetic demos.*?([\d,]+) frames at (\d+) Hz")
        self.model_b, = m(r"Qwen3\.5 (\d+)B")
        self.n_train_clips, = m(r"([\d,]+) clips \(\d+ HRDexDB")
        self.views, self.view_w, self.view_h, self.n_frames = m(r"native input \((\d) views x (\d+)x(\d+), (\d+) frames\)")
        self.n_test, self.n_test_fail, self.n_objects = m(r"\((\d+) trials, (\d+) failures, (\d+) objects never trained on\)")
        self.au_rel, self.au_force, self.au_ctrl, self.prim, self.prim_lo, self.prim_hi, self.prim_p = m(
            r"AUROC: released ([\d.]+) -> force adapter ([\d.]+), control ([\d.]+)\. Primary ([+\-][\d.]+) "
            r"\[([+\-][\d.]+), ([+\-][\d.]+)\], Holm p ([\d.e\-]+)")
        self.fr_rel, self.fr_force, self.fr_ctrl, self.sr_force, self.sr_ctrl = m(
            r"own threshold: ([\d.]+) -> ([\d.]+) \(force\) / ([\d.]+) \(control\), success recall ([\d.]+) / ([\d.]+)")
        self.fr90_rel, self.fr90_new = m(r"90% success recall: ([\d.]+) -> ([\d.]+)")
        self.ra_n, self.ra_fail, self.ra_rel, self.ra_force, self.ra_ctrl = m(
            r"REASSEMBLE test \((\d+) segments, (\d+) failures\): ([\d.]+) -> ([\d.]+) \(force\) / ([\d.]+) \(control\)")
        self.fnull, self.fnull_lo, self.fnull_hi = m(
            r"adds nothing \(([+\-][\d.]+) \[([+\-][\d.]+), ([+\-][\d.]+)\] HRDexDB\)")
        self.v3_px, self.v3_gain = m(r"In v3 at (\d+) px thumbnails, force added ([+\-][\d.]+) AUROC")
        self.fo_hrdex, self.fo_ra = m(r"Force-only classifier.*?AUROC ([\d.]+) HRDexDB, ([\d.]+) REASSEMBLE")
        self.pi_rank, = m(r"LoRA r(\d+) on every linear layer")
        self.pi_chunk, self.pi_chunk_s, self.pi_hz = m(r"(\d+)-step chunks \(([\d.]+) s at (\d+) Hz\)")
        self.pi_ntrain, self.pi_val_a, self.pi_val_b = m(r"(\d+) demos train, demos (\d+) and (\d+) held out")
        self.gpu_gb, = m(r"laptop RTX 5090 \((\d+) GB")
        self.team = m(r"- (Oren Gershony) \(ogershony\), (Kenan Blair)")

    def numbers(self) -> list[str]:
        return [x.lstrip("+-") for x in re.findall(r"(?<![\w.])[+\-]?\d+\.\d+", self.text)]


def load_results() -> dict | None:
    if RESULTS_JSON.is_file():
        try:
            return json.loads(RESULTS_JSON.read_text())
        except json.JSONDecodeError:
            return None
    return None


def jump_phrase(r: dict) -> str:
    n = r.get("n_chunks")
    if n:
        k = int(round(r["jump_rate"] * n))
        return f"none of its {n} chunks would trip the jump check" if k == 0 else f"{k} of its {n} chunks would trip the jump check"
    return "none of its chunks would trip the jump check" if r["jump_rate"] == 0 else f"{r['jump_rate']:.0%} of its chunks would trip the jump check"


def narration(results: dict | None) -> list[dict]:
    segs = json.loads((HERE / "narration.json").read_text())["segments"]
    out = []
    for s in segs:
        if "text_template" in s:
            if results:
                rs = s["result_sentence_template"].format(jump_phrase=jump_phrase(results), **results)
            else:
                rs = s["result_sentence_pending"]
            text = s["text_template"].format(result_sentence=rs)
        else:
            text = s["text"]
        out.append({"id": s["id"], "text": text})
    return out


def check_numbers(segs: list[dict], facts: Facts, results: dict | None) -> None:
    """Every decimal in the narration must match a fact (or result) to its own precision."""
    from decimal import ROUND_HALF_UP, Decimal

    known = facts.numbers()
    if results:
        known += [repr(float(v)) for v in results.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for s in segs:
        for tok in re.findall(r"(?<![\w.])\d+\.\d+", s["text"]):
            q = Decimal(1).scaleb(-len(tok.split(".")[1]))
            if not any(Decimal(y).quantize(q, ROUND_HALF_UP) == Decimal(tok) for y in known if "e" not in y.lower()):
                raise SystemExit(f"narration number {tok} in '{s['id']}' is not in facts.md or the results")


# ----------------------------------------------------------------------------------------------
# Voice
# ----------------------------------------------------------------------------------------------

TTS_MODEL = "eleven_multilingual_v2"
VOICE_PREFS = ["Brian", "Eric", "Daniel", "George", "Chris", "Will"]
VOICE_FALLBACK = ("nPczCjzI2devNBz1zQrb", "Brian")
VOICE_SETTINGS = {"stability": 0.55, "similarity_boost": 0.8, "style": 0.05, "use_speaker_boost": True, "speed": 1.0}


def api_key() -> str | None:
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"].strip() or None
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("ELEVENLABS_API_KEY="):
                v = line.split("=", 1)[1].strip().strip("'\"")
                return v or None
    return None


def pick_voice(key: str) -> tuple[str, str]:
    import requests

    vf = BUILD / "voice.json"
    if vf.is_file():
        v = json.loads(vf.read_text())
        return v["voice_id"], v["name"]
    try:
        r = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key}, timeout=30)
        r.raise_for_status()
        voices = r.json().get("voices", [])
        by_name = {v["name"].split(" ")[0].split("-")[0].strip(): v for v in voices}
        for p in VOICE_PREFS:
            if p in by_name:
                vid, name = by_name[p]["voice_id"], by_name[p]["name"]
                break
        else:
            vid, name = (voices[0]["voice_id"], voices[0]["name"]) if voices else VOICE_FALLBACK
    except Exception as e:  # restricted key: no voices_read permission
        print(f"voice list unavailable ({type(e).__name__}); using {VOICE_FALLBACK[1]}", flush=True)
        vid, name = VOICE_FALLBACK
    BUILD.mkdir(parents=True, exist_ok=True)
    vf.write_text(json.dumps({"voice_id": vid, "name": name, "model": TTS_MODEL}))
    return vid, name


def tts_segment(text: str, prev: str | None, nxt: str | None, key: str, voice_id: str) -> tuple[Path, dict]:
    import requests

    h = hashlib.sha1(json.dumps([text, voice_id, TTS_MODEL, VOICE_SETTINGS], sort_keys=True).encode()).hexdigest()[:16]
    mp3, js = TTS_DIR / f"{h}.mp3", TTS_DIR / f"{h}.json"
    if mp3.is_file() and js.is_file():
        return mp3, json.loads(js.read_text())
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    body = {"text": text, "model_id": TTS_MODEL, "voice_settings": VOICE_SETTINGS}
    if prev:
        body["previous_text"] = prev
    if nxt:
        body["next_text"] = nxt
    for attempt in range(4):
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=180,
        )
        if r.status_code == 200:
            break
        if r.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(4 * (attempt + 1))
            continue
        raise RuntimeError(f"ElevenLabs TTS failed: HTTP {r.status_code} {r.text[:300]}")
    d = r.json()
    mp3.write_bytes(base64.b64decode(d["audio_base64"]))
    al = d.get("alignment") or d.get("normalized_alignment") or {}
    js.write_text(json.dumps(al))
    return mp3, al


def decode_audio(path: Path) -> np.ndarray:
    c = av.open(str(path))
    rs = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
    chunks = []
    for fr in c.decode(audio=0):
        for o in rs.resample(fr):
            chunks.append(o.to_ndarray().reshape(-1))
    for o in rs.resample(None):
        chunks.append(o.to_ndarray().reshape(-1))
    c.close()
    return np.concatenate(chunks).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Timeline
# ----------------------------------------------------------------------------------------------


@dataclass
class Seg:
    id: str
    text: str
    start: float = 0.0
    dur: float = 0.0
    speech: float = 0.0  # seconds of narration
    audio: np.ndarray | None = None
    char_t: list[float] = field(default_factory=list)  # start time of each character (s, from speech start)
    pre: float = 0.0  # pre-roll before the narration lead-in

    def t_of(self, phrase: str) -> float:
        """Local time (s from segment start) at which `phrase` is spoken."""
        i = self.text.lower().find(phrase.lower())
        if i < 0:
            raise KeyError(f"cue '{phrase}' not in segment {self.id}")
        if self.char_t and len(self.char_t) == len(self.text):
            return self.pre + LEAD + self.char_t[i]
        return self.pre + LEAD + self.speech * i / max(1, len(self.text))

    def t_end_of(self, phrase: str) -> float:
        i = self.text.lower().find(phrase.lower())
        j = i + len(phrase) - 1
        if self.char_t and len(self.char_t) == len(self.text):
            return self.pre + LEAD + self.char_t[min(j, len(self.char_t) - 1)]
        return self.pre + LEAD + self.speech * j / max(1, len(self.text))


MIN_DUR = {"title": 3.8, "cold_open": 11.5, "close": 9.0}
PRE_ROLL = {"cold_open": 0.8}


def build_timeline(segs: list[dict], use_voice: bool) -> tuple[list[Seg], str]:
    key = api_key() if use_voice else None
    voice = "none (silent build, captions only)"
    out = []
    if key:
        vid, vname = pick_voice(key)
        voice = f"{vname} ({TTS_MODEL})"
        try:  # fail whole, not per segment: a half-voiced video is worse than a silent one
            for i, s in enumerate(segs):
                prev = segs[i - 1]["text"] if i > 0 else None
                nxt = segs[i + 1]["text"] if i + 1 < len(segs) else None
                tts_segment(s["text"], prev, nxt, key, vid)
        except Exception as e:
            print(f"WARNING: {e}; building silent with captions", flush=True)
            key, voice = None, "none (ElevenLabs request failed; silent build, captions only)"
    for i, s in enumerate(segs):
        seg = Seg(s["id"], s["text"])
        if key:
            prev = segs[i - 1]["text"] if i > 0 else None
            nxt = segs[i + 1]["text"] if i + 1 < len(segs) else None
            mp3, al = tts_segment(s["text"], prev, nxt, key, vid)  # cached by the loop above
            seg.audio = decode_audio(mp3)
            seg.speech = len(seg.audio) / AUDIO_RATE
            chars = al.get("characters") or []
            starts = al.get("character_start_times_seconds") or []
            if "".join(chars) == s["text"] and len(starts) == len(chars):
                seg.char_t = [float(x) for x in starts]
        else:
            seg.speech = len(s["text"].split()) / WPM_FALLBACK * 60.0
        out.append(seg)
    t = 0.0
    for seg in out:
        seg.pre = PRE_ROLL.get(seg.id, 0.0)
        seg.start = t
        seg.dur = max(MIN_DUR.get(seg.id, 0.0), seg.pre + LEAD + seg.speech + TAIL)
        t += seg.dur
    return out, voice


def seg_speech_start(seg: Seg) -> float:
    return seg.start + seg.pre + LEAD


# ----------------------------------------------------------------------------------------------
# Assets (decoded once into build/cache as .npy, memory-mapped by the render workers)
# ----------------------------------------------------------------------------------------------


def ds_path(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="dataset"))


def decode_range(mp4: Path, a: int, b: int, size: tuple[int, int] | None = None, crop=None) -> np.ndarray:
    c = av.open(str(mp4))
    out = []
    for i, f in enumerate(c.decode(video=0)):
        if i >= b:
            break
        if i >= a:
            im = f.to_image()
            if crop:
                im = im.crop(crop)
            if size:
                im = im.resize(size, Image.BICUBIC)
            out.append(np.asarray(im.convert("RGB")))
    c.close()
    return np.stack(out)


def prep_assets() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "DONE_v3"
    if done.is_file():
        return
    print("preparing footage cache ...", flush=True)
    R = ds_path("OG7407/baxter-cube-pick-left-rollouts")
    D = ds_path("OG7407/baxter-cube-pick-left-v0")
    rep = pd.read_parquet(R / "meta/episodes/chunk-000/file-000.parquet")
    dep = pd.read_parquet(D / "meta/episodes/chunk-000/file-000.parquet")

    def rng(ep, e):
        return int(ep.dataset_from_index[e]), int(ep.dataset_to_index[e])

    for cam in ("head", "wrist"):
        mp4 = R / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
        a, b = rng(rep, 0)
        np.save(CACHE / f"roll_{cam}_ep0.npy", decode_range(mp4, a, b))
    mp4 = R / "videos/observation.images.head/chunk-000/file-000.mp4"
    for e in (2, 3, 8, 9):
        a, b = rng(rep, e)
        np.save(CACHE / f"roll_head_ep{e}.npy", decode_range(mp4, max(a, b - 45), b, size=(560, 315)))
    runlog = []
    for e in range(len(rep)):
        meta = json.loads((R / f"raw/ep{e:04d}/meta.json").read_text())
        runlog.append({"ep": e, "stop": meta.get("stop_reason", ""), "label": meta.get("label", ""),
                       "frames": meta.get("frames"), "duration_s": meta.get("duration_s")})
    (CACHE / "runlog.json").write_text(json.dumps(runlog))
    tac = pd.read_csv(R / "raw/ep0000/tactile_200hz.csv")
    np.save(CACHE / "tactile_ep0.npy", tac[["t", "counts_L_a", "counts_L_b"]].to_numpy(np.float64))
    rdf = pd.read_parquet(R / "data/chunk-000/file-000.parquet")
    x0 = rdf[rdf.episode_index == 0]
    closed = x0["timestamp"].to_numpy()[np.argmax(x0["gripper_closed"].to_numpy() > 0.5)]
    (CACHE / "ep0.json").write_text(json.dumps({"fps": 15, "gripper_close_s": float(closed)}))

    for cam in ("head", "wrist"):
        mp4 = D / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
        for e in (17,):
            a, b = rng(dep, e)
            np.save(CACHE / f"demo_{cam}_ep{e}.npy", decode_range(mp4, a, b))
    mp4 = D / "videos/observation.images.head/chunk-000/file-000.mp4"
    for e in (5, 14):
        a, b = rng(dep, e)
        np.save(CACHE / f"demo_head_ep{e}.npy", decode_range(mp4, a, b, size=(640, 360)))

    for name, out, kw in (
        ("test_standing_frame_4_success", "oren_frame.npy", {"size": (1120, 630)}),
        ("dev_servingbowl_small_3_failure", "hrdex_bowl.npy", {"crop": (58, 58, 595, 461)}),
    ):
        mp4 = CACHE / f"{name}.mp4"
        mp4.write_bytes(subprocess.run(["git", "-C", str(REPO), "show", f"origin/main:writeup/videos/{name}.mp4"],
                                       check=True, capture_output=True).stdout)
        frames = decode_range(mp4, 0, 10_000, **kw)
        np.save(CACHE / out, frames)
    done.write_text("ok")
    print("footage cache ready", flush=True)


# ----------------------------------------------------------------------------------------------
# Charts (matplotlib, rendered once to transparent PNG layers)
# ----------------------------------------------------------------------------------------------


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in (HERE / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 15, "text.color": "#c9ccd2", "axes.labelcolor": "#c9ccd2",
        "xtick.color": "#9aa0a8", "ytick.color": "#9aa0a8", "axes.edgecolor": "#3a3f48",
        "axes.facecolor": (0, 0, 0, 0), "figure.facecolor": (0, 0, 0, 0), "savefig.transparent": True,
        "axes.grid": True, "grid.color": "#2a2e35", "grid.linewidth": 0.8,
    })
    return plt


def fig_to_rgba(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def roc_points(es: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fail = truth <= 2
    risk = -es
    thr = np.unique(risk)[::-1]
    tpr = [0.0]
    fpr = [0.0]
    for t in thr:
        flag = risk >= t
        tpr.append((flag & fail).sum() / fail.sum())
        fpr.append((flag & ~fail).sum() / (~fail).sum())
    return np.array(fpr), np.array(tpr)


def prep_charts(facts: Facts) -> None:
    plt = mpl()
    out = CACHE / "charts"
    out.mkdir(parents=True, exist_ok=True)

    # ROC on the locked HRDexDB test, from the per-clip export (E[s] rounded to 0.01).
    df = pd.read_csv(REPO / "docs/v4/results/per_clip_scores.csv")
    df = df[df.bench == "hrdexdb_v4_test"]
    curves = [
        ("released", "released", "omitted", f"Released FLORA  {facts.au_rel}"),
        ("force", "v4_force", "real", f"Tactile FLORA (force)  {facts.au_force}"),
        ("control", "v4_control", "omitted", f"Fine-tuned, no force  {facts.au_ctrl}"),
        ("classifier", "force_classifier", "none", f"Force only, no video  {facts.fo_hrdex}"),
    ]
    size = (940, 700)

    def axes():
        fig = plt.figure(figsize=(size[0] / 100, size[1] / 100))
        ax = fig.add_axes([0.11, 0.12, 0.85, 0.83])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("successes wrongly flagged as failures")
        ax.set_ylabel("failures caught")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return fig, ax

    fig, ax = axes()
    ax.plot([0, 1], [0, 1], ls=(0, (2, 4)), color="#4a4f58", lw=1.2)
    ax.text(0.62, 0.55, "chance", color="#6a707a", rotation=36, fontsize=13)
    fig_to_rgba(fig).save(out / "roc_base.png")
    plt.close(fig)
    meta = {}
    for key, ck, mode, label in curves:
        x = df[(df.checkpoint == ck) & (df["mode"] == mode)]
        fpr, tpr = roc_points(x.es.to_numpy(), x.truth.to_numpy())
        fig, ax = axes()
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.plot(fpr, tpr, color=SERIES[key], lw=3.2, solid_capstyle="round", drawstyle="steps-post")
        fig_to_rgba(fig).save(out / f"roc_{key}.png")
        x0 = ax.transData.transform((0, 0))[0]
        x1 = ax.transData.transform((1, 0))[0]
        meta[key] = {"label": label, "x0": float(x0), "x1": float(x1)}
        plt.close(fig)
    (out / "roc_meta.json").write_text(json.dumps(meta))

    # Fingertip pads during the successful grasp (rollout ep 0).
    tac = np.load(CACHE / "tactile_ep0.npy")
    t, a, b = tac[:, 0], tac[:, 1], tac[:, 2]
    ep0 = json.loads((CACHE / "ep0.json").read_text())
    dur = 84 / ep0["fps"]
    tsize = (1760, 210)
    fig = plt.figure(figsize=(tsize[0] / 100, tsize[1] / 100))
    ax = fig.add_axes([0.055, 0.2, 0.93, 0.72])
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 520)
    ax.set_yticks([0, 250, 500])
    ax.set_ylabel("ADC counts", fontsize=13)
    ax.set_xlabel("time (s)", fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig_to_rgba(fig).save(out / "pads_base.png")
    xpix = (float(ax.transData.transform((0, 0))[0]), float(ax.transData.transform((dur, 0))[0]))
    ypix = (float(ax.transData.transform((0, 0))[1]), float(ax.transData.transform((0, 520))[1]))
    plt.close(fig)
    fig = plt.figure(figsize=(tsize[0] / 100, tsize[1] / 100))
    ax = fig.add_axes([0.055, 0.2, 0.93, 0.72])
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 520)
    ax.axis("off")
    ax.plot(t, b, color=PAD_B, lw=2.4)
    ax.plot(t, a, color=PAD_A, lw=2.4)
    fig_to_rgba(fig).save(out / "pads_lines.png")
    plt.close(fig)
    (out / "pads_meta.json").write_text(json.dumps({"xpix": xpix, "ypix": ypix, "dur": dur, "h": tsize[1],
                                                    "ymax": 520, "gripper_close_s": ep0["gripper_close_s"]}))

    # Held-out Pi0.5 predictions, if the evaluation has dumped them.
    if HELDOUT_NPZ.is_file():
        z = np.load(HELDOUT_NPZ)
        ep = z["episode"]
        pick = int(facts.pi_val_b) if (ep == int(facts.pi_val_b)).any() else int(ep[0])
        sel = np.where(ep == pick)[0]
        gt, pred, fr = z["gt"][sel], z["pred"][sel], z["frame"][sel]
        order = np.argsort(fr)
        gt, pred, fr = gt[order], pred[order], fr[order]
        demo = gt[:, 0, :7]
        motion = demo.max(0) - demo.min(0)
        joints = list(np.argsort(motion)[::-1][:3])
        names = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
        psize = (1080, 640)
        fig = plt.figure(figsize=(psize[0] / 100, psize[1] / 100))
        axs = [fig.add_axes([0.09, 0.69 - 0.29 * k, 0.88, 0.25]) for k in range(3)]
        n = len(fr)
        for k, j in enumerate(joints):
            axx = axs[k]
            for i in range(0, n, 3):
                tt = (fr[i] + np.arange(pred.shape[1])) / 10.0
                axx.plot(tt, pred[i, :, j], color=PI_COLOR, lw=1.6, alpha=0.9, zorder=2)
            axx.plot(fr / 10.0, demo[:, j], color="#f1efea", lw=3.0, zorder=3, label="demonstrated")
            axx.set_ylabel(f"{names[j]} (rad)", fontsize=13)
            axx.set_xlim(0, (fr.max() + pred.shape[1]) / 10.0)
            for s in ("top", "right"):
                axx.spines[s].set_visible(False)
            if k < 2:
                axx.set_xticklabels([])
        axs[-1].set_xlabel("time in held-out demo (s)", fontsize=13)
        fig_to_rgba(fig).save(out / "heldout.png")
        xp = (float(axs[0].transData.transform((0, 0))[0]), float(axs[0].transData.transform((axs[0].get_xlim()[1], 0))[0]))
        plt.close(fig)
        (out / "heldout_meta.json").write_text(json.dumps({"episode": pick, "joints": [names[j] for j in joints],
                                                           "xpix": xp, "n": int(n)}))
    elif (out / "heldout.png").exists():
        (out / "heldout.png").unlink()


# ----------------------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------------------

FONT_FILES = {"inter": "Inter-var.ttf", "mono": "JetBrainsMono-var.ttf", "grotesk": "SpaceGrotesk-var.ttf"}


@lru_cache(maxsize=256)
def font(family: str, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    path = HERE / "fonts" / FONT_FILES[family]
    try:
        f = ImageFont.truetype(str(path), size)
        if family == "inter":
            f.set_variation_by_axes([min(32, max(14, size)), weight])
        else:
            lo, hi = (300, 700) if family == "grotesk" else (100, 800)
            f.set_variation_by_axes([min(hi, max(lo, weight))])
        return f
    except Exception:
        return ImageFont.truetype("DejaVuSans.ttf", size)


@lru_cache(maxsize=2048)
def text_img(text: str, family: str, size: int, weight: int, color: tuple, tracking: float = 0.0) -> Image.Image:
    f = font(family, size, weight)
    if tracking == 0:
        l, t, r, b = f.getbbox(text)
        im = Image.new("RGBA", (r - min(0, l) + 4, b + 6), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((-min(0, l) + 1, 0), text, font=f, fill=color + (255,))
        return im
    widths = [f.getlength(c) + tracking for c in text]
    asc, desc = f.getmetrics()
    im = Image.new("RGBA", (int(sum(widths)) + 4, asc + desc + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 1.0
    for c, w_ in zip(text, widths):
        d.text((x, 0), c, font=f, fill=color + (255,))
        x += w_
    return im


def fade(layer: Image.Image, op: float) -> Image.Image:
    if op >= 0.999:
        return layer
    a = np.asarray(layer.getchannel("A"), dtype=np.float32) * max(0.0, op)
    out = layer.copy()
    out.putalpha(Image.fromarray(a.astype(np.uint8)))
    return out


def put(canvas: Image.Image, layer: Image.Image, x: float, y: float, op: float = 1.0, anchor: str = "lt") -> None:
    if op <= 0.003:
        return
    w_, h_ = layer.size
    if anchor[0] == "m":
        x -= w_ / 2
    elif anchor[0] == "r":
        x -= w_
    if anchor[1] == "m":
        y -= h_ / 2
    elif anchor[1] == "b":
        y -= h_
    canvas.alpha_composite(fade(layer, op), (int(round(x)), int(round(y))))


def txt(canvas, s, x, y, size=30, weight=400, color=FG, family="inter", op=1.0, anchor="lt", tracking=0.0):
    im = text_img(s, family, size, weight, color, tracking)
    put(canvas, im, x, y, op, anchor)
    return im.size


@lru_cache(maxsize=64)
def round_mask(w_: int, h_: int, r: int) -> Image.Image:
    m = Image.new("L", (w_ * 2, h_ * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w_ * 2 - 1, h_ * 2 - 1], radius=r * 2, fill=255)
    return m.resize((w_, h_), Image.LANCZOS)


def footage(arr: np.ndarray, w_: int, h_: int, r: int = 16, crop=None, dim: float = 0.0) -> Image.Image:
    im = Image.fromarray(arr)
    if crop:
        im = im.crop(crop)
    im = im.resize((w_, h_), Image.BICUBIC).convert("RGBA")
    if dim:
        im = Image.blend(im, Image.new("RGBA", im.size, BG + (255,)), dim)
    im.putalpha(round_mask(w_, h_, r))
    return im


def rrect(canvas, box, r=16, fill=PANEL, outline=None, width=1, op=1.0):
    x0, y0, x1, y1 = [int(v) for v in box]
    layer = Image.new("RGBA", (x1 - x0 + 1, y1 - y0 + 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=r, fill=None if fill is None else fill + (255,),
                        outline=None if outline is None else outline + (255,), width=width)
    put(canvas, layer, x0, y0, op)


def border(canvas, box, r=16, color=(255, 255, 255), alpha=34):
    x0, y0, x1, y1 = [int(v) for v in box]
    layer = Image.new("RGBA", (x1 - x0 + 1, y1 - y0 + 1), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=r, outline=color + (alpha,), width=2)
    canvas.alpha_composite(layer, (x0, y0))


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ease_io(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def appear(t: float, t0: float, d: float = 0.5) -> float:
    return ease((t - t0) / d)


def section(canvas, num: str, label: str, op=1.0):
    txt(canvas, num, 80, 44, 22, 600, ACCENT, "mono", op, tracking=1.5)
    txt(canvas, label.upper(), 128, 44, 22, 600, MUTED, "mono", op, tracking=3.0)


def tag(canvas, s, x, y, color=ACCENT, size=22, op=1.0, anchor="lt", text_color=FG):
    im = text_img(s, "mono", size, 600, text_color, 0.5)
    pad_x, pad_y = 14, 8
    box = Image.new("RGBA", (im.size[0] + 2 * pad_x, im.size[1] + 2 * pad_y - 4), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle([0, 0, box.size[0] - 1, box.size[1] - 1], radius=8, fill=color + (235,))
    box.alpha_composite(im, (pad_x, pad_y - 2))
    put(canvas, box, x, y, op, anchor)
    return box.size


def frame_at(arr: np.ndarray, t: float, fps: float, loop: bool = False) -> np.ndarray:
    i = int(t * fps)
    n = len(arr)
    return arr[i % n] if loop else arr[min(max(i, 0), n - 1)]


@lru_cache(maxsize=None)
def npy(name: str) -> np.ndarray:
    return np.load(CACHE / name, mmap_mode="r")


@lru_cache(maxsize=None)
def png(name: str) -> Image.Image:
    return Image.open(CACHE / "charts" / name).convert("RGBA")


@lru_cache(maxsize=None)
def jload(name: str):
    return json.loads((CACHE / name).read_text())


class Overlay:
    """Draw lines and shapes with real alpha blending (ImageDraw writes RGBA ink unblended)."""

    def __init__(self, c: Image.Image):
        self.c = c
        self.im = Image.new("RGBA", c.size, (0, 0, 0, 0))

    def __enter__(self) -> ImageDraw.ImageDraw:
        return ImageDraw.Draw(self.im)

    def __exit__(self, *exc) -> None:
        self.c.alpha_composite(self.im)


def wipe(layer: Image.Image, x_px: float) -> Image.Image:
    x_px = int(max(0, min(layer.size[0], x_px)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    if x_px > 0:
        out.paste(layer.crop((0, 0, x_px, layer.size[1])), (0, 0))
    return out


# ----------------------------------------------------------------------------------------------
# Scenes. Each draws onto an RGBA canvas at local time t (s) of a segment.
# ----------------------------------------------------------------------------------------------


class Ctx:
    def __init__(self, facts: Facts, results: dict | None, segs: list[Seg]):
        self.f = facts
        self.r = results
        self.segs = {s.id: s for s in segs}


def scene_cold_open(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    head, wrist = npy("roll_head_ep0.npy"), npy("roll_wrist_ep0.npy")
    pm = jload("charts/pads_meta.json")
    speed = 0.5
    vt = min(max(0.0, (t - 0.3) * speed), pm["dur"] - 1e-3)
    fi = min(int(vt * 15), len(head) - 1)
    op = appear(t, 0.0, 0.8)
    put(c, footage(head[fi], 1184, 666, crop=(0, 30, 480, 300)), 80, 96, op)
    border(c, (80, 96, 1264, 762))
    put(c, footage(wrist[fi], 520, 293), 1320, 96, op)
    border(c, (1320, 96, 1840, 389))
    txt(c, "HEAD CAMERA", 100, 112, 18, 600, FG, "mono", op * 0.9, tracking=2)
    txt(c, "WRIST CAMERA", 1340, 112, 18, 600, FG, "mono", op * 0.9, tracking=2)
    tag(c, f"ACT ROLLOUT 1 OF {f.n_runs} · 0.5x", 100, 712, (40, 44, 52), 18, op)
    # live readout
    tac = npy("tactile_ep0.npy")
    k = int(np.searchsorted(tac[:, 0], vt))
    k = min(max(k, 0), len(tac) - 1)
    va, vb = tac[k, 1], tac[k, 2]
    y = 430
    txt(c, f"FINGERTIP PADS · {f.pad_hz} Hz", 1320, y, 18, 600, MUTED, "mono", op, tracking=2)
    for i, (name, v, col) in enumerate((("pad A", va, PAD_A), ("pad B", vb, PAD_B))):
        yy = y + 44 + i * 108
        rrect(c, (1320, yy, 1840, yy + 92), 14, PANEL, op=op)
        rrect(c, (1320, yy, 1328, yy + 92), 3, hexrgb(col), op=op)
        txt(c, name, 1348, yy + 14, 22, 500, MUTED, "inter", op)
        txt(c, f"{int(v)}", 1820, yy + 44, 46, 600, FG, "mono", op, anchor="rm")
        txt(c, "counts", 1348, yy + 52, 18, 400, FAINT, "inter", op)
    closed = vt >= pm["gripper_close_s"]
    tag(c, "GRIPPER CLOSED" if closed else "GRIPPER OPEN", 1320, y + 270,
        ACCENT if closed else (40, 44, 52), 20, op)
    # trace
    base, lines = png("pads_base.png"), png("pads_lines.png")
    ox, oy = 80, 770
    put(c, base, ox, oy, op)
    xp = pm["xpix"][0] + (pm["xpix"][1] - pm["xpix"][0]) * vt / pm["dur"]
    put(c, wipe(lines, xp + 1), ox, oy, op)
    # resting line label and playhead
    yrest = pm["h"] - (pm["ypix"][0] + (pm["ypix"][1] - pm["ypix"][0]) * float(f.pad_rest) / pm["ymax"])
    txt(c, f"resting {f.pad_rest}", ox + pm["xpix"][1] - 6, oy + yrest - 30, 16, 500, MUTED, "mono", op, anchor="rt")
    px = ox + xp
    with Overlay(c) as d:
        d.line([(px, oy + 14), (px, oy + pm["h"] - pm["ypix"][0])], fill=(255, 255, 255, int(110 * op)), width=2)
    if vt >= pm["gripper_close_s"]:
        gx = ox + pm["xpix"][0] + (pm["xpix"][1] - pm["xpix"][0]) * pm["gripper_close_s"] / pm["dur"]
        txt(c, "contact", gx + 8, oy + 8, 16, 600, hexrgb(PAD_A), "mono", appear(vt, pm["gripper_close_s"], 0.3))


def scene_title(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    head = npy("roll_head_ep0.npy")
    bg = Image.fromarray(head[-1]).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(10)).convert("RGBA")
    bg = Image.blend(bg, Image.new("RGBA", (W, H), BG + (255,)), 0.72)
    c.alpha_composite(bg)
    op = appear(t, 0.1, 0.8)
    rise = (1 - ease((t - 0.1) / 0.9)) * 30
    txt(c, "Tactile FLORA", W / 2, 420 + rise, 150, 700, FG, "inter", op, anchor="mm")
    txt(c, "Teaching robots to tell when they failed", W / 2, 540 + rise, 50, 400, (214, 212, 207), "inter",
        appear(t, 0.5, 0.8), anchor="mm")
    txt(c, f"{f.event} · {f.demo_date}", W / 2, 640, 24, 500, MUTED, "mono", appear(t, 0.9, 0.8), anchor="mm",
        tracking=1)


def scene_problem(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    section(c, "01", "The problem")
    tiles = (2, 3, 8, 9)
    log = jload("runlog.json")
    for i, e in enumerate(tiles):
        col, row = i % 2, i // 2
        x0, y0 = 80 + col * 584, 110 + row * 343
        arr = npy(f"roll_head_ep{e}.npy")
        op = appear(t, 0.1 + 0.15 * i, 0.5)
        n = len(arr)
        cyc = (t * 15) % (n + 20)  # play the last 3 s, hold the final frame with the abort tag
        fi = int(min(cyc, n - 1))
        put(c, footage(arr[fi], 560, 315, 14), x0, y0, op)
        border(c, (x0, y0, x0 + 560, y0 + 315), 14)
        txt(c, f"run {e + 1:02d}", x0 + 16, y0 + 14, 18, 600, FG, "mono", op)
        if cyc >= n - 1:
            stop = log[e]["stop"]
            m = re.search(r"([\d.]+) rad", stop)
            msg = f"ABORT · chunk {m.group(1)} rad from arm" if m else "ABORT · " + stop[:28]
            tag(c, msg, x0 + 16, y0 + 315 - 16, ACCENT, 18, op, anchor="lb")
    # right column: tally, run log, failure mode
    t1 = seg.t_of("succeeded once")
    o1 = appear(t, t1 - 0.2, 0.5)
    txt(c, f"{f.n_success} / {f.n_runs}", 1272, 104, 120, 700, FG, "mono", o1)
    txt(c, "ACT rollouts that picked the cube", 1276, 250, 24, 500, MUTED, "inter", o1)
    y = 300
    for i, r in enumerate(log):
        oo = appear(t, t1 + 0.12 * i, 0.3)
        ok = r["label"] == "success"
        stop = r["stop"]
        m = re.search(r"([\d.]+) rad", stop)
        short = "picked" if ok else (f"jump {m.group(1)} rad > {f.max_jump}" if m else ("time limit" if "time" in stop else stop[:22]))
        col = FG if ok else (238, 160, 150)
        txt(c, f"{i + 1:02d}", 1276, y + 4 + i * 37, 21, 500, FAINT, "mono", oo)
        txt(c, "✓" if ok else "✗", 1326, y + 2 + i * 37, 23, 700, FG if ok else ACCENT, "inter", oo)
        txt(c, short, 1356, y + 4 + i * 37, 21, 600 if ok else 500, col, "mono", oo)
    t2 = seg.t_of("Eight of the nine")
    o2 = appear(t, t2 - 0.1, 0.5)
    rrect(c, (1272, 690, 1840, 900), 16, PANEL, op=o2)
    rrect(c, (1272, 690, 1280, 900), 3, ACCENT, op=o2)
    txt(c, f"{f.n_jump} of {f.n_fail} failures", 1304, 712, 40, 700, FG, "inter", o2)
    txt(c, "a new action chunk started", 1304, 770, 25, 400, (214, 212, 207), "inter", o2)
    txt(c, f"{f.jump_lo}-{f.jump_hi} rad from the arm,", 1304, 804, 25, 400, (214, 212, 207), "inter", o2)
    txt(c, f"past the {f.max_jump} rad safety limit", 1304, 838, 25, 400, (214, 212, 207), "inter", o2)


def card(c, x0, y0, w_, h_, num, title, stat, lines, op, active, thumb):
    rrect(c, (x0, y0, x0 + w_, y0 + h_), 18, PANEL, op=op)
    if active > 0:
        layer = Image.new("RGBA", (w_ + 1, h_ + 1), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle([0, 0, w_, h_], radius=18, outline=ACCENT + (int(255 * active),), width=3)
        c.alpha_composite(fade(layer, op), (x0, y0))
    txt(c, num, x0 + 28, y0 + 26, 20, 600, ACCENT, "mono", op, tracking=1.5)
    txt(c, title, x0 + 28, y0 + 58, 40, 700, FG, "inter", op)
    put(c, thumb, x0 + 28, y0 + 124, op)
    yy = y0 + 124 + thumb.size[1] + 22
    txt(c, stat, x0 + 28, yy, 44, 700, FG, "mono", op)
    for i, ln in enumerate(lines):
        txt(c, ln, x0 + 28, yy + 66 + i * 36, 23, 400, (214, 212, 207), "inter", op)


def scene_built(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    section(c, "02", "What we built")
    cues = [seg.t_of("Force pads"), seg.t_of("Tactile FLORA"), seg.t_of("Pi zero point five")]
    w_, h_, gap = 552, 610, 52
    x0s = [80 + i * (w_ + gap) for i in range(3)]
    y0 = 150
    head = npy("roll_head_ep0.npy")
    bowl = npy("hrdex_bowl.npy")
    demo = npy("demo_head_ep17.npy")
    thumbs = [
        footage(frame_at(head, t * 0.6, 15, True), 496, 279, 12, crop=(0, 40, 448, 292)),
        footage(frame_at(bowl, t * 1.5, 10, True), 496, 279, 12),
        footage(frame_at(demo, t, 10, True), 496, 279, 12),
    ]
    specs = [
        ("SENSE", "Fingertip force", f"{f.pad_hz} Hz", [f"{f.n_pads} FSR pads on the gripper fingers", "ESP32 or USB serial, raw counts"]),
        ("JUDGE", "Tactile FLORA", f"{f.model_b}B params", ["FLORA-Rewarder video model + LoRA", "video + force summary, score 1 to 5"]),
        ("ACT", "Pi0.5 policy", f"{f.pi_chunk_s} s chunks", ["PaliGemma 3B + 300M action expert", "head + wrist camera, joint state"]),
    ]
    for i in range(3):
        op = appear(t, cues[i] - 0.3, 0.6) * 0.75 + 0.25 * appear(t, 0.0, 0.6)
        nxt = cues[i + 1] if i < 2 else seg.dur
        active = 1.0 if cues[i] - 0.2 <= t < nxt - 0.2 else 0.0
        num, title, stat, lines = specs[i]
        card(c, x0s[i], y0, w_, h_, num, title, stat, lines, op, active, thumbs[i])
    # flow arrows into the judge
    oa = appear(t, cues[1] + 0.3, 0.6)
    if oa > 0:
        with Overlay(c) as d:
            ya = y0 + 264
            col = (200, 200, 200, int(200 * oa))
            xa0, xa1 = x0s[0] + w_ + 6, x0s[1] - 6
            d.line([(xa0, ya), (xa1, ya)], fill=col, width=3)
            d.polygon([(xa1, ya), (xa1 - 14, ya - 8), (xa1 - 14, ya + 8)], fill=col)
            xb0, xb1 = x0s[2] - 6, x0s[1] + w_ + 6
            d.line([(xb0, ya), (xb1, ya)], fill=col, width=3)
            d.polygon([(xb1, ya), (xb1 + 14, ya - 8), (xb1 + 14, ya + 8)], fill=col)
    ob = appear(t, cues[2] + 1.2, 0.6)
    txt(c, "The judge scores every attempt from the video and the fingertip force.", W / 2, 818, 28, 500, MUTED,
        "inter", ob, anchor="mm")


def legend_rows(c, x0, y0, items, ops):
    for i, ((key, label), op) in enumerate(zip(items, ops)):
        yy = y0 + i * 40
        rrect(c, (x0, yy + 8, x0 + 28, yy + 14), 3, hexrgb(SERIES[key]), op=op)
        txt(c, label, x0 + 42, yy, 22, 500, FG, "inter", op)


def scene_reward(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    section(c, "03", "The judge: Tactile FLORA")
    bowl = npy("hrdex_bowl.npy")
    op = appear(t, 0.0, 0.6)
    put(c, footage(frame_at(bowl, t * 1.4, 10, True), 760, 570, 16), 80, 120, op)
    border(c, (80, 120, 840, 690))
    tag(c, "HRDexDB · DEXTEROUS HAND", 100, 136, (40, 44, 52), 18, op)
    txt(c, f"{f.n_test} locked test trials · {f.n_objects} unseen objects", 80, 712, 26, 600, FG, "inter", op)
    txt(c, f"{f.n_test_fail} failures · trained on {f.n_train_clips} clips at FLORA's native input", 80, 750, 22, 400, MUTED, "inter", op)
    # ROC
    meta = jload("charts/roc_meta.json")
    rx, ry = 900, 96
    put(c, png("roc_base.png"), rx, ry, appear(t, 0.2, 0.6))
    order = [("released", seg.t_of("0.783")), ("force", seg.t_of("0.956")), ("control", seg.t_of("0.956") + 0.8),
             ("classifier", seg.t_of("0.956") + 1.6)]
    ops = []
    for key, tc in order:
        p = ease((t - tc + 0.2) / 1.6)
        m = meta[key]
        layer = png(f"roc_{key}.png")
        put(c, wipe(layer, m["x0"] + (m["x1"] - m["x0"]) * p + 4), rx, ry, 1.0 if p > 0 else 0.0)
        ops.append(appear(t, tc - 0.2, 0.4))
    legend_rows(c, rx + 330, ry + 430, [(k, meta[k]["label"]) for k, _ in order], ops)
    txt(c, "HRDexDB locked test", rx + 104, ry + 6, 20, 600, MUTED, "mono", appear(t, 0.2, 0.6), tracking=1.5)
    # headline numbers
    cards = [
        (seg.t_of("Failure-detection AUROC"), "AUROC", f"{f.au_rel} → {f.au_force}", "failure detection"),
        (seg.t_of("Failure recall"), "FAILURE RECALL", f"{float(f.fr_rel):.0%} → {float(f.fr_force):.0%}", "at the model's own threshold"),
        (seg.t_of("passing ninety-eight"), "SUCCESSES PASSED", f"{float(f.sr_force):.0%}", "still accepted"),
        (seg.t_of("On REASSEMBLE"), "REASSEMBLE AUROC", f"{f.ra_rel} → {f.ra_force}", f"{f.ra_n} held-out segments"),
    ]
    cw, gap = 418, 22
    for i, (tc, lab, val, sub) in enumerate(cards):
        o = appear(t, tc - 0.15, 0.45)
        x0 = 80 + i * (cw + gap)
        y0 = 800
        rrect(c, (x0, y0, x0 + cw, y0 + 150), 16, PANEL, op=o)
        txt(c, lab, x0 + 24, y0 + 18, 17, 600, MUTED, "mono", o, tracking=2)
        txt(c, val, x0 + 24, y0 + 48, 46, 700, FG, "mono", o)
        txt(c, sub, x0 + 24, y0 + 108, 20, 400, FAINT, "inter", o)


def scene_force(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    section(c, "04", "What touch adds")
    clip = npy("oren_frame.npy")
    op = appear(t, 0.0, 0.6)
    fi = frame_at(clip, 11.0 + t, 10, False)  # from just before contact (about 14 s into the clip)
    put(c, footage(fi, 1120, 630, 16), 80, 110, op)
    border(c, (80, 110, 1200, 740))
    tag(c, f"EARLIER RUN (v3) · MODEL INPUT {f.v3_px} PX · FORCE FIXED THIS CALL", 100, 756, (40, 44, 52), 18, op)
    # dot chart: AUROC gained by adding force, at each input resolution
    lx, ax0, ax1 = 1270, 1556, 1820
    lo, hi = -0.03, 0.06

    def X(v):
        return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

    oc = appear(t, seg.t_of("native video resolution") - 0.3, 0.6)
    txt(c, "AUROC gained by adding force", lx, 112, 26, 700, FG, "inter", oc)
    txt(c, "same model, with vs without the force text", lx, 150, 19, 400, MUTED, "inter", oc)
    ya, yb = 250, 360
    o_nat = appear(t, seg.t_of("nothing measurable") - 0.2, 0.5)
    o_v3 = appear(t, seg.t_of("ninety-six pixel") - 0.2, 0.5)
    with Overlay(c) as d:
        if oc > 0:
            zx = X(0.0)
            d.line([(zx, 200), (zx, 410)], fill=(120, 124, 132, int(200 * oc)), width=2)
            for v in (-0.02, 0.02, 0.04):
                d.line([(X(v), 402), (X(v), 410)], fill=(90, 94, 102, int(220 * oc)), width=2)
        if o_v3 > 0:
            col = hexrgb(SERIES["force"]) + (int(255 * o_v3),)
            g = float(f.v3_gain)
            d.ellipse([X(g) - 11, ya - 11, X(g) + 11, ya + 11], fill=col)
        if o_nat > 0:
            col = hexrgb(SERIES["force"]) + (int(255 * o_nat),)
            d.line([(X(float(f.fnull_lo)), yb), (X(float(f.fnull_hi)), yb)], fill=col, width=4)
            g = float(f.fnull)
            d.ellipse([X(g) - 11, yb - 11, X(g) + 11, yb + 11], fill=col)
    if oc > 0:
        txt(c, "0", X(0.0), 418, 17, 500, MUTED, "mono", oc, anchor="mt")
        for v in (-0.02, 0.02, 0.04):
            txt(c, f"{v:+.2f}", X(v), 418, 15, 400, FAINT, "mono", oc, anchor="mt")
    txt(c, f"{f.v3_px} px thumbnails", lx, ya - 30, 22, 600, FG, "inter", o_v3)
    txt(c, f"v3: {f.v3_gain}", lx, ya + 2, 20, 500, MUTED, "mono", o_v3)
    txt(c, f"native {f.view_w}x{f.view_h}", lx, yb - 30, 22, 600, FG, "inter", o_nat)
    txt(c, f"v4: {f.fnull} [{f.fnull_lo}, {f.fnull_hi}]", lx, yb + 2, 18, 500, MUTED, "mono", o_nat)
    o_msg = appear(t, seg.t_of("Touch matters") - 0.1, 0.5)
    rrect(c, (1270, 490, 1840, 620), 16, PANEL, op=o_msg)
    rrect(c, (1270, 490, 1278, 620), 3, ACCENT, op=o_msg)
    txt(c, "Touch matters where", 1300, 510, 34, 700, FG, "inter", o_msg)
    txt(c, "vision is starved.", 1300, 556, 34, 700, FG, "inter", o_msg)
    o_fo = appear(t, seg.t_of("force alone") - 0.1, 0.5)
    rrect(c, (1270, 644, 1840, 840), 16, PANEL, op=o_fo)
    txt(c, "FORCE ONLY, NO VIDEO", 1300, 666, 17, 600, MUTED, "mono", o_fo, tracking=2)
    txt(c, f"{f.fo_hrdex} · {f.fo_ra}", 1300, 696, 46, 700, hexrgb(SERIES["classifier"]), "mono", o_fo)
    txt(c, "AUROC on HRDexDB · REASSEMBLE", 1300, 762, 20, 400, MUTED, "inter", o_fo)
    txt(c, f"fine-tuned video alone: {f.au_ctrl}", 1300, 794, 20, 400, FAINT, "inter", o_fo)


def scene_policy(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    r = x.r
    section(c, "05", "The policy: Pi0.5 on Baxter")
    t_frozen = seg.t_of("The vision-language backbone")
    t_rel = seg.t_of("Actions are relative")
    t_res = seg.t_of("On two held-out")
    # phase 1: kinesthetic demos
    p1 = 1 - appear(t, t_frozen + 2.0, 0.6)
    if p1 > 0:
        head, wrist = npy("demo_head_ep17.npy"), npy("demo_wrist_ep17.npy")
        op = appear(t, 0.0, 0.6) * p1
        put(c, footage(frame_at(head, t, 10, True), 1040, 585, 16), 80, 110, op)
        put(c, footage(frame_at(wrist, t, 10, True), 680, 383, 16), 1160, 110, op)
        tag(c, "KINESTHETIC DEMONSTRATION · A PERSON GUIDES THE ARM", 100, 126, (40, 44, 52), 18, op)
        txt(c, f"{f.n_demos} demos", 1160, 530, 64, 700, FG, "inter", op)
        txt(c, f"{f.demo_frames} frames at {f.demo_hz} Hz · \"pick up the red cube\"", 1162, 612, 24, 400, MUTED, "inter", op)
        txt(c, f"laptop RTX 5090 ({f.gpu_gb} GB) · overnight", 1162, 650, 24, 400, MUTED, "inter", op)
    # phase 2: recipe
    p2 = appear(t, t_frozen + 2.0, 0.6) * (1 - appear(t, t_res - 0.2, 0.6))
    if p2 > 0:
        boxes = [
            ("INPUTS", ["head + wrist camera", "joint state (as text)", "\"pick up the red cube\""], (40, 44, 52), MUTED),
            ("PaliGemma 3B", ["SigLIP + Gemma 2B", "frozen: the action loss", "never reaches it"], (40, 44, 52), MUTED),
            ("ACTION EXPERT", ["300M flow matching", f"LoRA r{f.pi_rank}, every linear layer", "+ action and time projections"], ACCENT, FG),
            ("OUTPUT", [f"{f.pi_chunk} x 8 joint targets", f"{f.pi_chunk_s} s at {f.pi_hz} Hz", "relative to the arm"], (40, 44, 52), MUTED),
        ]
        bw, gap = 400, 52
        for i, (title, lines, accent, _) in enumerate(boxes):
            o = p2 * appear(t, t_frozen + 2.0 + 0.25 * i, 0.5)
            x0, y0 = 80 + i * (bw + gap), 170
            rrect(c, (x0, y0, x0 + bw, y0 + 250), 16, PANEL, op=o)
            rrect(c, (x0, y0, x0 + bw, y0 + 8), 3, accent, op=o)
            txt(c, title, x0 + 24, y0 + 30, 28, 700, FG, "inter", o)
            for k, ln in enumerate(lines):
                txt(c, ln, x0 + 24, y0 + 90 + k * 40, 23, 400, (214, 212, 207), "inter", o)
            if i < 3:
                with Overlay(c) as d:
                    ax_ = x0 + bw + 8
                    col = (200, 200, 200, int(200 * o))
                    d.line([(ax_, y0 + 125), (ax_ + gap - 16, y0 + 125)], fill=col, width=3)
                    d.polygon([(ax_ + gap - 16, y0 + 125), (ax_ + gap - 30, y0 + 117), (ax_ + gap - 30, y0 + 133)], fill=col)
        # absolute vs relative chunk sketch
        o3 = p2 * appear(t, t_rel - 0.2, 0.6)
        if o3 > 0:
            yb = 480
            rrect(c, (80, yb, 1840, yb + 330), 16, PANEL, op=o3)
            txt(c, "WHY RELATIVE ACTIONS", 112, yb + 24, 18, 600, MUTED, "mono", o3, tracking=2)
            # schematic, not data: arm position and the first target of a new chunk
            rows = ((f"absolute targets (ACT): a new chunk can start {f.jump_hi} rad away", 1, False),
                    ("relative targets (Pi0.5 here): chunk = arm + offsets", 0, True))
            with Overlay(c) as d:
                for k, (label, gap_rad, ok) in enumerate(rows):
                    yy = yb + 110 + k * 110
                    arm_x = 200
                    col = hexrgb(PI_COLOR) if ok else ACCENT
                    d.ellipse([arm_x - 14, yy - 14, arm_x + 14, yy + 14], fill=FG + (int(255 * o3),))
                    start = arm_x + (300 if gap_rad else 8)
                    if gap_rad:
                        d.line([(arm_x + 18, yy), (start - 10, yy)], fill=ACCENT + (int(160 * o3),), width=2)
                    pts = [(start + 52 * j, yy - 10 * math.sin(j / 2.2)) for j in range(9)]
                    d.line(pts, fill=col + (int(255 * o3),), width=4)
                    for p in pts:
                        d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=col + (int(255 * o3),))
            for k, (label, gap_rad, ok) in enumerate(rows):
                yy = yb + 110 + k * 110
                if gap_rad:
                    txt(c, "jump", 200 + 150, yy - 34, 18, 600, ACCENT, "mono", o3, anchor="mt")
                txt(c, label, 1000, yy - 14, 23, 500, FG if ok else (238, 160, 150), "inter", o3)
            txt(c, "arm", 200, yb + 290, 17, 500, MUTED, "mono", o3, anchor="mt")
            txt(c, "schematic", 1820, yb + 24, 16, 500, FAINT, "mono", o3, anchor="rt")
    # phase 3: held-out results
    p3 = appear(t, t_res - 0.2, 0.6)
    if p3 > 0:
        hp = CACHE / "charts/heldout.png"
        if hp.is_file():
            hm = jload("charts/heldout_meta.json")
            layer = png("heldout.png")
            prog = ease((t - t_res) / 4.0)
            put(c, wipe(layer, hm["xpix"][0] + (hm["xpix"][1] - hm["xpix"][0]) * prog + 4), 60, 120, p3)
            txt(c, f"HELD-OUT DEMO {hm['episode']} · white: demonstrated · blue: Pi0.5 chunks every 0.3 s",
                90, 108, 18, 600, MUTED, "mono", p3, tracking=1)
        else:
            rrect(c, (80, 120, 1140, 760), 16, PANEL, op=p3)
            layer = Image.new("RGBA", (1061, 641), (0, 0, 0, 0))
            ImageDraw.Draw(layer).rounded_rectangle([0, 0, 1060, 640], radius=16, outline=(238, 160, 150, 255), width=2)
            put(c, layer, 80, 120, p3)
            txt(c, "PLACEHOLDER", 610, 400, 40, 700, (238, 160, 150), "mono", p3, anchor="mm", tracking=4)
            txt(c, "held-out trajectory plot: waiting for runs/heldout_pred.npz", 610, 460, 22, 400, MUTED, "inter", p3, anchor="mm")
        rows = []
        if r:
            n = r.get("n_chunks")
            k = int(round(r["jump_rate"] * n)) if n else None
            rows = [
                ("JOINT ERROR, 1 S CHUNK", f"{r['joint_mae_rad']:.3f} rad", f"holding still: {r['hold_joint_mae_rad']:.3f} rad"),
                ("FIRST STEP ERROR", f"{r['step0_mae_rad']:.3f} rad", "what the arm is sent next"),
                ("JUMP-LIMIT TRIPS", f"{k} / {n}" if n else f"{r['jump_rate']:.0%}", f"chunks starting > {f.max_jump} rad from the arm"),
            ]
        else:
            rows = [("JOINT ERROR, 1 S CHUNK", "PENDING", "placeholder until training ends"),
                    ("FIRST STEP ERROR", "PENDING", "placeholder"),
                    ("JUMP-LIMIT TRIPS", "PENDING", "placeholder")]
        for i, (lab, val, sub) in enumerate(rows):
            o = appear(t, t_res + 0.4 + 0.5 * i, 0.5)
            x0, y0 = 1180, 120 + i * 216
            rrect(c, (x0, y0, 1840, y0 + 196), 16, PANEL, op=o)
            if not r:
                layer = Image.new("RGBA", (661, 197), (0, 0, 0, 0))
                ImageDraw.Draw(layer).rounded_rectangle([0, 0, 660, 196], radius=16, outline=(238, 160, 150, 255), width=2)
                put(c, layer, x0, y0, o)
            txt(c, lab, x0 + 28, y0 + 24, 18, 600, MUTED, "mono", o, tracking=2)
            txt(c, val, x0 + 28, y0 + 60, 58, 700, hexrgb(PI_COLOR) if r else (238, 160, 150), "mono", o)
            txt(c, sub, x0 + 28, y0 + 140, 21, 400, FAINT, "inter", o)
        txt(c, f"Open loop on demos {f.pi_val_a} and {f.pi_val_b}, never trained on. Not yet a robot run.", 80, 790, 22, 400,
            MUTED, "inter", p3)


def scene_close(c: Image.Image, t: float, seg: Seg, x: Ctx) -> None:
    f = x.f
    r = x.r
    head = npy("roll_head_ep0.npy")
    bg = Image.fromarray(head[-1]).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(14)).convert("RGBA")
    c.alpha_composite(Image.blend(bg, Image.new("RGBA", (W, H), BG + (255,)), 0.82))
    op = appear(t, 0.0, 0.7)
    txt(c, "Tactile FLORA", 120, 150, 110, 700, FG, "inter", op)
    rows = [
        ("JUDGE", f"failure AUROC {f.au_rel} → {f.au_force} on {f.n_test} locked trials, {f.n_objects} unseen objects"),
        ("TOUCH", f"helps where vision is starved: {f.v3_gain} at {f.v3_px} px, no measurable gain at native video"),
        ("POLICY", "Pi0.5 for Baxter, relative 1 s chunks anchored to the arm" + (
            f" · held-out error {r['joint_mae_rad']:.3f} rad" if r else "")),
    ]
    for i, (k, v) in enumerate(rows):
        o = appear(t, 0.5 + 0.35 * i, 0.5)
        y = 340 + i * 92
        txt(c, k, 124, y + 6, 22, 600, ACCENT, "mono", o, tracking=2)
        txt(c, v, 300, y, 32, 500, FG, "inter", o)
    o = appear(t, 1.8, 0.6)
    txt(c, f.repo.replace("https://", ""), 120, 700, 44, 600, FG, "mono", o)
    txt(c, f"{f.team[0]} · {f.team[1]} · {f.event}", 120, 780, 24, 400, MUTED, "inter", o)


SCENES = {"cold_open": scene_cold_open, "title": scene_title, "problem": scene_problem, "built": scene_built,
          "reward": scene_reward, "force": scene_force, "policy": scene_policy, "close": scene_close}


# ----------------------------------------------------------------------------------------------
# Captions and frame assembly
# ----------------------------------------------------------------------------------------------


CAPTION_FORMS = {"github dot com slash ogershony slash tactile flora": "github.com/ogershony/tactile-flora",
                 "Pi zero point five": "Pi0.5", "Flora": "FLORA", "Reassemble": "REASSEMBLE"}


def caption_chunks(seg: Seg) -> list[tuple[float, float, str]]:
    """Split narration into caption lines (<= 64 chars) at punctuation, timed from the speech."""
    text = seg.text
    parts, cur, start = [], "", 0
    for m in re.finditer(r"(?:[^,.;:]|[.,](?=\d))+[,.;:]?\s*", text):
        piece = m.group(0)
        if cur and len(cur) + len(piece) > 64:
            parts.append((start, cur))
            start, cur = m.start(), piece
        else:
            if not cur:
                start = m.start()
            cur += piece
    if cur:
        parts.append((start, cur))
    out = []
    base = seg.pre + LEAD
    for i, (s0, p) in enumerate(parts):
        e0 = parts[i + 1][0] if i + 1 < len(parts) else len(text)
        if seg.char_t and len(seg.char_t) == len(text):
            ts = base + seg.char_t[s0]
            te = base + (seg.char_t[e0] if e0 < len(text) else seg.speech)
        else:
            ts = base + seg.speech * s0 / len(text)
            te = base + seg.speech * e0 / len(text)
        shown = p.strip()
        for a, b in CAPTION_FORMS.items():
            shown = shown.replace(a, b)
        out.append((ts, te, shown))
    return out


def draw_caption(c: Image.Image, seg: Seg, t: float) -> None:
    for ts, te, s in caption_chunks(seg):
        if ts - 0.05 <= t < te + 0.05:
            im = text_img(s, "inter", 30, 500, FG, 0.0)
            pad = 18
            bw, bh = im.size[0] + 2 * pad, im.size[1] + 14
            box = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            ImageDraw.Draw(box).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=10, fill=(8, 9, 11, 205))
            box.alpha_composite(im, (pad, 6))
            op = min(1.0, (t - ts + 0.05) / 0.15, (te + 0.05 - t) / 0.15)
            put(c, box, W / 2, 1040, max(0.0, op), anchor="mb")
            return


class Renderer:
    def __init__(self, segs: list[Seg], ctx: Ctx):
        self.segs = segs
        self.ctx = ctx
        self.total = segs[-1].start + segs[-1].dur

    def seg_frame(self, seg: Seg, t_local: float) -> Image.Image:
        c = Image.new("RGBA", (W, H), BG + (255,))
        SCENES[seg.id](c, t_local, seg, self.ctx)
        draw_caption(c, seg, t_local)
        return c

    def frame(self, t: float) -> Image.Image:
        i = max(k for k, s in enumerate(self.segs) if s.start <= t + 1e-9) if t >= 0 else 0
        seg = self.segs[i]
        tl = t - seg.start
        im = self.seg_frame(seg, tl)
        # crossfade into the next segment over its first XFADE seconds
        if i > 0 and tl < XFADE:
            prev = self.segs[i - 1]
            a = ease_io(tl / XFADE)
            im = Image.blend(self.seg_frame(prev, prev.dur + tl), im, a)
        if i == 0 and tl < 0.6:
            im = Image.blend(Image.new("RGBA", (W, H), BG + (255,)), im, ease(tl / 0.6))
        end = self.total - t
        if end < 0.8:
            im = Image.blend(Image.new("RGBA", (W, H), BG + (255,)), im, ease(end / 0.8))
        return im.convert("RGB")


_R: Renderer | None = None


def _init(state: bytes) -> None:
    global _R
    import pickle

    segs, facts_path, results = pickle.loads(state)
    _R = Renderer(segs, Ctx(Facts(facts_path), results, segs))


def _render(i: int) -> bytes:
    return _R.frame(i / FPS).tobytes()


def mix_audio(segs: list[Seg], total: float) -> np.ndarray:
    out = np.zeros(int(math.ceil(total * AUDIO_RATE)) + AUDIO_RATE, dtype=np.float32)
    for s in segs:
        if s.audio is None:
            continue
        a = int((seg_speech_start(s)) * AUDIO_RATE)
        out[a : a + len(s.audio)] += s.audio
    peak = np.abs(out).max()
    if peak > 0.98:
        out *= 0.98 / peak
    return out[: int(total * AUDIO_RATE)]


def encode(segs: list[Seg], renderer: Renderer, state: bytes, workers: int, out: Path, has_audio: bool, crf: int) -> None:
    total = renderer.total
    n = int(math.ceil(total * FPS))
    tmp = out.with_suffix(".part.mp4")
    cont = av.open(str(tmp), "w", options={"movflags": "+faststart"})
    vs = cont.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"crf": str(crf), "preset": "medium"}
    asr = None
    if has_audio:
        asr = cont.add_stream("aac", rate=AUDIO_RATE, layout="stereo")
        asr.bit_rate = 160_000
        audio = mix_audio(segs, total)
        # encode audio up front, then interleave packets by time
        apk = []
        stereo = np.stack([audio, audio])
        step = 1024
        for k in range(0, stereo.shape[1], step):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(stereo[:, k : k + step]), format="fltp", layout="stereo")
            fr.sample_rate = AUDIO_RATE
            fr.pts = k
            apk.extend(asr.encode(fr))
        apk.extend(asr.encode(None))
        apk_t = [float(p.pts * p.time_base) if p.pts is not None else 0.0 for p in apk]
    ai = 0
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(state,)) as pool:
        for i, buf in enumerate(pool.imap(_render, range(n), chunksize=4)):
            fr = av.VideoFrame.from_ndarray(np.frombuffer(buf, np.uint8).reshape(H, W, 3), format="rgb24")
            fr.pts = i
            for p in vs.encode(fr):
                cont.mux(p)
            if has_audio:
                while ai < len(apk) and apk_t[ai] <= i / FPS:
                    cont.mux(apk[ai])
                    ai += 1
            if i % 300 == 0:
                print(f"  frame {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    for p in vs.encode(None):
        cont.mux(p)
    if has_audio:
        while ai < len(apk):
            cont.mux(apk[ai])
            ai += 1
    cont.close()
    tmp.replace(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(FACTS_DEFAULT if FACTS_DEFAULT.is_file() else HERE / "facts.md"))
    ap.add_argument("--stills", default="")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--silent", action="store_true", help="skip ElevenLabs even if a key exists")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    facts_path = Path(args.facts)
    facts = Facts(facts_path)
    if facts_path.resolve() != (HERE / "facts.md").resolve():
        (HERE / "facts.md").write_text(facts.text)  # snapshot of the facts this render used
    results = load_results()
    segs_txt = narration(results)
    check_numbers(segs_txt, facts, results)
    segs, voice = build_timeline(segs_txt, use_voice=not args.silent)
    total = segs[-1].start + segs[-1].dur
    words = sum(len(s.text.split()) for s in segs)
    print(f"voice: {voice}; {words} words; {total:.1f} s; results: {'yes' if results else 'PENDING'}", flush=True)
    for s in segs:
        print(f"  {s.id:10s} {s.start:6.1f} +{s.dur:5.1f}  speech {s.speech:5.1f}  {'aligned' if s.char_t else 'estimated'}")
    if args.plan:
        return
    prep_assets()
    prep_charts(facts)
    ctx = Ctx(facts, results, segs)
    renderer = Renderer(segs, ctx)
    if args.stills:
        outd = BUILD / "stills"
        outd.mkdir(parents=True, exist_ok=True)
        for ts in args.stills.split(","):
            p = outd / f"still_{float(ts):06.1f}.png"
            renderer.frame(float(ts)).save(p)
            print(p)
        return
    import pickle

    # audio arrays are only needed in the parent; workers get the timing
    light = []
    for s in segs:
        light.append(Seg(s.id, s.text, s.start, s.dur, s.speech, None, list(s.char_t), s.pre))
    state = pickle.dumps((light, facts_path, results))
    has_audio = any(s.audio is not None for s in segs)
    for crf in CRF_STEPS:  # the slide deck embeds the file under a 20 MB cap
        encode(segs, renderer, state, args.workers, OUT_MP4, has_audio, crf)
        if OUT_MP4.stat().st_size <= MAX_BYTES:
            break
        print(f"  {OUT_MP4.stat().st_size / 1e6:.1f} MB at crf {crf}, over the cap; re-encoding", flush=True)
    # poster: the title card
    title = next(s for s in segs if s.id == "title")
    renderer.frame(title.start + min(2.2, title.dur - 0.5)).save(OUT_POSTER)
    meta = {"duration_s": round(total, 2), "crf": crf, "voice": voice, "words": words, "results": bool(results),
            "heldout_plot": (CACHE / "charts/heldout.png").is_file(), "size_mb": round(OUT_MP4.stat().st_size / 1e6, 1)}
    (BUILD / "last_render.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    sys.exit(main())
