"""Build the silent project video: the slide deck with the narration as on-screen text.

Each card shows one slide (or one of the two side-by-side clips) above a caption band that
carries one or two sentences of narration. Card length follows the caption's word count. No
audio. Every number in the captions is copied from ``writeup/tactile-flora-report.md``.

Needs ``pymupdf`` (renders the slide PDF), ``pillow`` (composes the frames) and
``imageio-ffmpeg`` (a bundled ffmpeg that encodes and concatenates the segments):

    uv venv /tmp/video-venv --python 3.11
    uv pip install --python /tmp/video-venv/bin/python pymupdf pillow imageio-ffmpeg
    /tmp/video-venv/bin/python writeup/video/build_video.py

Writes ``writeup/video/tactile-flora.mp4`` (1920x1080, H.264, 24 fps).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SLIDES_PDF = ROOT / "writeup/slides/tactile-flora.pdf"
CLIPS = {
    "hrdexdb": ROOT / "writeup/hrdexdb/videos/test_standing_frame_4_success.mp4",
    "baxter": ROOT / "writeup/baxter-rollouts/videos/ep0000_success.mp4",
}
OUT = ROOT / "writeup/video/tactile-flora.mp4"

W, H = 1920, 1080
STAGE = (160, 30, 1600, 900)  # x, y, w, h of the slide or clip
BAND_TOP = STAGE[1] + STAGE[3] + 18
BG = (23, 33, 43)  # the deck's dark slate
INK = (240, 240, 236)
MUTED = (150, 160, 172)
FONT = "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf"
FPS = 24


@dataclass(frozen=True)
class Card:
    slide: int  # 1-based slide number shown (also the stage for clip cards' progress label)
    text: str
    clip: str | None = None  # key in CLIPS: play the clip on the stage instead of the slide
    clip_offset: float = 0.0  # seconds into the clip to start


# fmt: off
CARDS = [
    Card(1, "Tactile FLORA: fingertip force as text for a vision-language reward model. Oren Gershony, Aaryan Patel, Kenan Blair and Parth Trivedi, Triton Droids, UC San Diego."),
    Card(2, "Video reward models such as NVIDIA's FLORA-Rewarder judge a grasp from video alone. But a grasp is decided at the fingertips, where the camera cannot see whether contact was made, held or lost."),
    Card(3, "The idea: keep the model as it is. Render the fingertip force as a short text block after the video frames, and fine-tune with LoRA on 350 HRDexDB grasp clips."),
    Card(4, "The block: one row per frame with grip force, pad asymmetry, closure and a contact flag, plus a summary line computed from the force alone. Everything is tared per trial and scaled to the median grasp."),
    Card(5, "Every claim is checked against a matched fine-tune with the block omitted, against shuffled force, and against force removed or video blanked."),
    Card(6, "It took four stages. A naive rendering gave no gain; an audit found the block nearly uninformative. Correcting the representation, not the model, produced the result."),
    Card(7, "On 251 held-out clips over 32 unseen objects: 79.3% accuracy and 0.873 AUROC, against 76.5% / 0.830 for the video-only fine-tune and 66.5% / 0.600 for released FLORA."),
    Card(8, "The ranking gain over the matched control is +0.043 AUROC, object-level bootstrap 95% interval +0.001 to +0.080, p about 0.02."),
    Card(9, "Shuffle the force blocks between clips and dev accuracy drops from 85.3% to 73.5% (p = 0.009). Blank the video and AUROC falls to 0.775. The model reads both channels."),
    Card(10, "The gain lands where the video is unsure: on those 101 clips, accuracy rises from 66% to 72%. Confident clips are left alone."),
    Card(11, "A win: standing frame #4, a success the video-only model scored 1.27 of 5. The force shows contact held through the lift; the tactile model scores it 4.19.", clip="hrdexdb", clip_offset=8.0),
    Card(12, "Then our own robot: a Baxter with home-built force-sensing fingertips, 50 hand-guided demos, an ACT policy, and ten live rollouts with two successes. Zero-shot: no Baxter data in training."),
    Card(13, "Released FLORA ranks the two successes below chance, AUROC 0.375; the video-only fine-tune 0.313. The tactile model ranks them perfectly: 1.0. Shuffled force 0.25, removed 0.44."),
    Card(14, "Episode 0: neither video model can tell this lift from a miss. The force block shows a clean hold, and the tactile model ranks the clip first.", clip="baxter"),
    Card(15, "Caveats: one dataset and one hand; ten Baxter clips with two successes; uncalibrated fingertips; low-resolution video."),
    Card(16, "Tactile FLORA outperforms regular FLORA at judging grasps. Force as text is a cheap, inspectable way to give a video reward model the contact information it lacks. github.com/ogershony/tactile-flora"),
]
# fmt: on


def seconds(text: str) -> float:
    """Reading time: about 160 words per minute plus a second to settle, never under 4 s."""
    return max(4.0, len(text.split()) / 2.7 + 1.0)


def wrap(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    """Greedy word wrap to the band width."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def frame(card: Card, slide_png: Path | None) -> Image.Image:
    """Compose one 1920x1080 frame: slide (or empty stage) above the caption band."""
    im = Image.new("RGB", (W, H), BG)
    if slide_png is not None:
        slide = Image.open(slide_png).convert("RGB").resize((STAGE[2], STAGE[3]), Image.LANCZOS)
        im.paste(slide, (STAGE[0], STAGE[1]))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 36)
    lines = wrap(draw, card.text, font, 1500)
    # Balance a multi-line caption: narrow the wrap while the line count holds.
    for width in range(1450, 900, -50):
        trial = wrap(draw, card.text, font, width)
        if len(trial) != len(lines):
            break
        lines = trial
    line_h = 46
    y = BAND_TOP + (H - BAND_TOP - line_h * len(lines)) // 2 - 4
    for ln in lines:
        w = draw.textlength(ln, font=font)
        draw.text(((W - w) / 2, y), ln, font=font, fill=INK)
        y += line_h
    small = ImageFont.truetype(FONT, 20)
    label = f"{card.slide} / 16"
    draw.text((W - 60 - draw.textlength(label, font=small), H - 40), label, font=small, fill=MUTED)
    return im


def main() -> None:
    """Render the slides, compose every card, encode each as a segment, concatenate."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        doc = pymupdf.open(SLIDES_PDF)
        slides: dict[int, Path] = {}
        for i, page in enumerate(doc, 1):
            scale = pymupdf.Matrix(STAGE[2] / page.rect.width, STAGE[3] / page.rect.height)
            png = tmp / f"slide{i:02d}.png"
            page.get_pixmap(matrix=scale).save(png)
            slides[i] = png

        segments: list[Path] = []
        total = 0.0
        common = [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-preset",
            "medium",
            "-crf",
            "20",
            "-an",
        ]
        for n, card in enumerate(CARDS):
            dur = seconds(card.text)
            total += dur
            png = tmp / f"card{n:02d}.png"
            seg = tmp / f"seg{n:02d}.mp4"
            if card.clip is None:
                frame(card, slides[card.slide]).save(png)
                cmd = [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-loop",
                    "1",
                    "-framerate",
                    str(FPS),
                    "-i",
                    str(png),
                    "-t",
                    f"{dur:.2f}",
                    *common,
                    str(seg),
                ]
            else:
                frame(card, None).save(png)
                cmd = [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-loop",
                    "1",
                    "-framerate",
                    str(FPS),
                    "-i",
                    str(png),
                    "-ss",
                    f"{card.clip_offset:.2f}",
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(CLIPS[card.clip]),
                    "-filter_complex",
                    f"[1:v]scale={STAGE[2]}:{STAGE[3]}[c];[0:v][c]overlay={STAGE[0]}:{STAGE[1]}:shortest=1",
                    "-t",
                    f"{dur:.2f}",
                    *common,
                    str(seg),
                ]
            subprocess.run(cmd, check=True)
            segments.append(seg)

        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in segments))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(OUT),
            ],
            check=True,
        )
    print(f"{OUT}: {len(CARDS)} cards, {total / 60:.1f} min")


if __name__ == "__main__":
    main()
