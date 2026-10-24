"""Build writeup/slides/tactile-flora.pptx from the numbers in writeup/tactile-flora-report.md.

Run from the repository root with a venv that has python-pptx, matplotlib and pillow:

    python writeup/slides/build_deck.py

Charts are rendered with matplotlib into writeup/slides/figures/ and placed as images; the
diagram and timeline are native shapes. Every number is copied from the report (and from
comparison.md / baxter-rollouts.md, which the report cites); nothing is recomputed here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "writeup" / "slides"
FIG = HERE / "figures"
OUT = HERE / "tactile-flora.pptx"
VIDEOS = ROOT / "writeup" / "videos"
BAXTER = ROOT / "writeup" / "baxter-rollouts"

# Palette: ink on white, one blue accent (tactile), one warm accent (force), dark slate ends.
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x17, 0x21, 0x2B)
BLUE = RGBColor(0x2A, 0x6F, 0xB0)
ORANGE = RGBColor(0xD9, 0x48, 0x0F)
CARD = RGBColor(0xEE, 0xF2, 0xF6)
GREY = RGBColor(0xB8, 0xBC, 0xC2)
GREEN = RGBColor(0x1E, 0x84, 0x49)
RED = RGBColor(0xC0, 0x39, 0x2B)
LIGHT = RGBColor(0xD6, 0xDD, 0xE5)
HEAD_FONT = "Cambria"
BODY_FONT = "Calibri"
MONO_FONT = "Courier New"
W, H = Inches(13.333), Inches(7.5)
DATE = "2026-10-24"


# ----------------------------------------------------------------------------- figures
def hex_(c: RGBColor) -> str:
    return "#" + str(c)


def fig_hrdexdb() -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "hrdexdb_auroc.png"
    splits = ["dev (136 clips)", "test (115 clips)", "pooled (251)"]
    models = ["Released FLORA", "Fine-tuned, video only", "Fine-tuned tactile"]
    auroc = {
        "Released FLORA": [0.614, 0.593, 0.600],
        "Fine-tuned, video only": [0.899, 0.733, 0.830],
        "Fine-tuned tactile": [0.926, 0.800, 0.873],
    }
    colors = [hex_(GREY), "#8C9199", hex_(BLUE)]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    bw = 0.26
    for i, m in enumerate(models):
        xs = [j + (i - 1) * bw for j in range(len(splits))]
        bars = ax.bar(xs, auroc[m], bw, color=colors[i], label=m, zorder=3)
        for b, v in zip(bars, auroc[m], strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, color=hex_(INK), fontfamily="DejaVu Sans",
            )
    ax.axhline(0.5, color="#9AA0A6", lw=0.9, ls="--", zorder=2)
    ax.text(-0.42, 0.507, "chance", fontsize=9, color="#6B7280", ha="left", va="bottom")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits, fontsize=11, color=hex_(INK))
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC (success ranked above failure)", fontsize=10, color=hex_(MUTED))
    ax.tick_params(axis="y", labelsize=10, colors=hex_(MUTED))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D6DDE5")
    ax.yaxis.grid(True, color="#E6E8EB", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.12))
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def fig_baxter() -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "baxter_auroc.png"
    rows = [
        ("Fine-tuned tactile FLORA, real force", 1.000, True),
        ("Released FLORA + force block", 1.000, True),
        ("Tactile adapter, force removed", 0.438, False),
        ("Released FLORA, no force", 0.375, False),
        ("Fine-tuned FLORA, video only", 0.313, False),
        ("Tactile adapter, shuffled force", 0.250, False),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=200)
    fig.patch.set_facecolor("white")
    ys = list(range(len(rows)))[::-1]
    for y, (name, v, force) in zip(ys, rows, strict=True):
        ax.barh(y, v, 0.58, color=hex_(BLUE) if force else hex_(GREY), zorder=3)
        lx = v + 0.015 if not (0.40 < v + 0.015 < 0.50) else 0.515
        ax.text(lx, y, f"{v:.3f}", va="center", ha="left", fontsize=11, color=hex_(INK))
    ax.axvline(0.5, color="#9AA0A6", lw=0.9, ls="--", zorder=2)
    ax.text(0.5, len(rows) - 0.35, "chance 0.50", fontsize=9, color="#6B7280", ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=hex_(INK))
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D6DDE5")
    ax.set_xlabel("AUROC over the 2 successes x 8 failures (ten Baxter rollouts)", fontsize=10, color=hex_(MUTED))
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- helpers
class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    # -- primitives
    def slide(self, dark: bool = False):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        if dark:
            bg = s.background.fill
            bg.solid()
            bg.fore_color.rgb = DARK
        else:
            self.text(s, f"Tactile FLORA  ·  {DATE}", 0.6, 7.0, 6, 0.3, size=10, color=MUTED)
            self.text(s, str(self.n), 12.2, 7.0, 0.53, 0.3, size=10, color=MUTED, align=PP_ALIGN.RIGHT)
        return s

    def text(
        self, s, txt, x, y, w, h, *, size=16, bold=False, color=INK, font=BODY_FONT,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, name=None,
    ):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tb.name = name or f"text {txt[:20]}"
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Emu(0)
        tf.margin_top = tf.margin_bottom = Emu(0)
        tf.vertical_anchor = anchor
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line
            f = r.font
            f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
            f.color.rgb = color
        return tb

    def bullets(self, s, items, x, y, w, h, *, size=16, color=INK, gap=8, bold_lead=True):
        """Items are strings or (lead, rest) tuples; a tuple bolds the lead."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tb.name = "bullets"
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Emu(0)
        tf.margin_top = tf.margin_bottom = Emu(0)
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            lead, rest = (it if isinstance(it, tuple) else ("", it))
            r0 = p.add_run()
            r0.text = "•  "
            r0.font.size, r0.font.name, r0.font.color.rgb = Pt(size), BODY_FONT, BLUE
            if lead:
                r1 = p.add_run()
                r1.text = lead + " "
                r1.font.size, r1.font.name, r1.font.bold = Pt(size), BODY_FONT, bold_lead
                r1.font.color.rgb = color
            r2 = p.add_run()
            r2.text = rest
            r2.font.size, r2.font.name, r2.font.color.rgb = Pt(size), BODY_FONT, color
        return tb

    def title(self, s, txt, *, dark=False, size=32):
        return self.text(
            s, txt, 0.6, 0.45, 12.1, 0.95, size=size, bold=True, font=HEAD_FONT,
            color=WHITE if dark else INK, anchor=MSO_ANCHOR.MIDDLE, name="title",
        )

    def card(self, s, x, y, w, h, fill=CARD, name="card"):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.name = name
        sh.adjustments[0] = 0.08
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        sh.line.fill.background()
        sh.shadow.inherit = False
        return sh

    def box(self, s, txt, x, y, w, h, *, fill=CARD, color=INK, size=14, bold=False, name="box"):
        sh = self.card(s, x, y, w, h, fill=fill, name=name)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            r.font.size, r.font.name, r.font.bold = Pt(size), BODY_FONT, bold if i == 0 else False
            r.font.color.rgb = color
        return sh

    def arrow(self, s, x1, y1, x2, y2, color=MUTED):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = color
        c.line.width = Pt(1.75)
        ln = c.line._get_or_add_ln()
        tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
        return c

    def stat(self, s, big, label, x, y, w, *, color=BLUE, big_size=48, dark=False):
        self.text(s, big, x, y, w, 0.9, size=big_size, bold=True, color=color, font=HEAD_FONT, anchor=MSO_ANCHOR.BOTTOM)
        self.text(s, label, x, y + 0.95, w, 0.7, size=13, color=LIGHT if dark else MUTED)

    def table(self, s, rows, x, y, w, col_w, *, size=13, header=True, row_h=0.36, bold_col=None):
        n_r, n_c = len(rows), len(rows[0])
        t = s.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r)).table
        t.first_row = header
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
        for i, row in enumerate(rows):
            t.rows[i].height = Inches(row_h)
            for j, val in enumerate(row):
                c = t.cell(i, j)
                c.margin_left = c.margin_right = Inches(0.06)
                c.margin_top = c.margin_bottom = Inches(0.02)
                c.fill.solid()
                c.fill.fore_color.rgb = CARD if (header and i == 0) else WHITE
                tf = c.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
                r = p.add_run()
                r.text = str(val)
                r.font.size, r.font.name = Pt(size), BODY_FONT
                r.font.bold = (header and i == 0) or (bold_col is not None and j == bold_col and i > 0)
                r.font.color.rgb = BLUE if (bold_col is not None and j == bold_col and i > 0) else INK
        return t

    def image(self, s, path, x, y, w=None, h=None, name=None):
        pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
        pic.name = name or Path(path).name
        return pic

    def notes(self, s, txt):
        s.notes_slide.notes_text_frame.text = txt

    def save(self):
        self.prs.save(OUT)


# ----------------------------------------------------------------------------- slides
def build() -> None:
    f_hrdex, f_bax = fig_hrdexdb(), fig_baxter()
    d = Deck()

    # 1 title (dark)
    s = d.slide(dark=True)
    d.text(s, "Tactile FLORA", 0.9, 1.9, 9.3, 1.2, size=54, bold=True, color=WHITE, font=HEAD_FONT)
    d.text(s, "Fingertip force as text for a vision-language reward model", 0.9, 3.1, 9.3, 0.8, size=26, color=LIGHT)
    d.text(s, ["Oren Gershony and collaborators, Triton Droids / SDSC", DATE], 0.9, 5.4, 11.5, 0.9, size=16, color=LIGHT)
    d.card(s, 10.4, 1.6, 2.0, 2.0, fill=BLUE, name="motif")
    d.text(s, "F", 10.4, 1.6, 2.0, 2.0, size=72, bold=True, color=WHITE, font=HEAD_FONT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    d.notes(s, "Deck built from writeup/tactile-flora-report.md on 2026-10-24.")

    # 2 problem
    s = d.slide()
    d.title(s, "A vision-only reward model cannot see what the fingers feel")
    d.text(s, "FLORA-Rewarder (NVIDIA, Qwen3.5-27B) watches a video mosaic and scores task success 1 to 5. It has no contact channel.", 0.6, 1.5, 7.2, 1.0, size=16)
    items = [
        ("Grasp or no grasp.", "An empty gripper closes, lifts and moves exactly like a loaded one."),
        ("Hold or touch.", "A brief contact looks like a grasp on camera; the force says it ended early."),
        ("Weak grip, re-grasp, slip.", "Visually unremarkable, obvious in the force trace."),
        ("Small views.", "At 96 px per camera a cube is a few pixels wide; the lift is invisible."),
    ]
    d.bullets(s, items, 0.6, 2.6, 7.2, 3.8, size=16, gap=12)
    d.card(s, 8.3, 1.5, 4.45, 5.2)
    d.text(s, "Released FLORA on a Baxter success", 8.55, 1.7, 4.0, 0.4, size=13, bold=True, color=MUTED)
    d.text(s, "“the cube remained on the table in the final frames”", 8.55, 2.2, 4.0, 1.3, size=20, italic=True, font=HEAD_FONT)
    d.text(s, "The cube was 20 cm in the air, held for 2.9 s.", 8.55, 3.7, 4.0, 0.6, size=14, color=MUTED)
    d.text(s, "Rationale from the released model on rollout ep0000 (writeup/baxter-rollouts.md).", 8.55, 5.9, 4.0, 0.6, size=11, color=MUTED)

    # 3 idea: diagram
    s = d.slide()
    d.title(s, "The idea: render force as text, put it in the prompt, fine-tune with LoRA")
    y0 = 2.0
    d.box(s, ["Cameras", "scene video"], 0.6, y0, 2.2, 1.0, name="cams")
    d.box(s, ["64-frame mosaic", "released FLORA sampling, 96 px cells"], 3.5, y0, 2.9, 1.0, name="mosaic")
    d.arrow(s, 2.8, y0 + 0.5, 3.5, y0 + 0.5)
    y1 = 4.1
    d.box(s, ["Fingertip force", "FSR pads / Inspire hand, 50 to 200 Hz"], 0.6, y1, 2.2, 1.0, fill=RGBColor(0xFB, 0xE9, 0xDF), name="sensor")
    d.box(s, ["Resample", "to the 64 frame times"], 3.5, y1, 1.75, 1.0, fill=RGBColor(0xFB, 0xE9, 0xDF), name="resample")
    d.box(s, ["Normalize", "0 free space, 1 median grasp"], 5.45, y1, 1.75, 1.0, fill=RGBColor(0xFB, 0xE9, 0xDF), name="normalize")
    d.box(s, ["Text block", "per-frame lines + summary + classifier P"], 7.4, y1, 2.3, 1.0, fill=RGBColor(0xFB, 0xE9, 0xDF), name="block")
    d.arrow(s, 2.8, y1 + 0.5, 3.5, y1 + 0.5)
    d.arrow(s, 5.25, y1 + 0.5, 5.45, y1 + 0.5)
    d.arrow(s, 7.2, y1 + 0.5, 7.4, y1 + 0.5)
    d.box(s, ["FLORA-Rewarder", "Qwen3.5-27B + LoRA r32", "vision encoder frozen"], 10.3, 2.9, 2.45, 1.4, fill=BLUE, color=WHITE, bold=True, name="flora")
    d.arrow(s, 6.4, y0 + 0.5, 10.3, 3.35)
    d.arrow(s, 9.7, y1 + 0.5, 10.3, 3.85)
    d.box(s, ["Score 1 to 5", "E[s], success at 4.0"], 10.3, 5.0, 2.45, 0.9, name="score")
    d.arrow(s, 11.52, 4.3, 11.52, 5.0)
    d.text(s, "Everything released stays: weights, prompts, frame sampling. One addition: a force block after the video, introduced by one sentence that names its columns. Matched control: same training with the block omitted.", 0.6, 5.7, 9.2, 1.1, size=14, color=MUTED)

    # 4 force block
    s = d.slide()
    d.title(s, "The force block the model reads")
    block = [
        "Force (normalized: 0 free space, 1 median grasp, max 3):",
        "summary R: contacts=1 longest=6.00s total=6.00s peak=1.83 hold=5.67s",
        "   holdF=1.55 late/early=0.70 low=0% force-only classifier P(success)=0.76",
        "t=0.00 L none | pad R F=0.00 D=0.00 dF=0.00 g=0.00 c=0",
        "t=0.25 L none | pad R F=0.00 D=0.00 dF=0.00 g=0.00 c=0",
        "...",
        "t=12.50 L none | pad R F=1.62 D=0.21 dF=0.03 g=0.71 c=1",
    ]
    d.card(s, 0.6, 1.5, 7.6, 3.1, fill=RGBColor(0xF4, 0xF6, 0xF8))
    d.text(s, block, 0.85, 1.7, 7.2, 2.8, size=12, font=MONO_FONT)
    items = [
        ("F", "total grip force; D pad asymmetry; dF its rate"),
        ("g", "hand closure, 0 open to 1 closed; c contact flag (F above 5% of grasp)"),
        ("summary", "label-free: contact segments, longest and total contact, peak, hold time and force, late/early hold ratio, low-force fraction"),
        ("P(success)", "a 120-tree, depth-2 gradient-boosted classifier over 31 force statistics, fitted on the 350 training clips; training clips see out-of-fold values"),
        ("budget", "at most 3,000 tokens per clip (largest seen: 2,700)"),
    ]
    d.bullets(s, items, 8.5, 1.5, 4.3, 5.0, size=13, gap=8)
    d.text(s, "HRDexDB training clip inspire_dftp/cactus2/0, abbreviated. Per-trial tare, two most loaded pads, sign-corrected closure.", 0.6, 4.8, 7.6, 0.6, size=12, color=MUTED)
    d.card(s, 0.6, 5.6, 7.6, 1.1)
    d.text(s, "Why text: no new encoder, no change to the released model; the resampler keeps the raw slice so an encoder can be added later.", 0.85, 5.75, 7.1, 0.85, size=14, anchor=MSO_ANCHOR.MIDDLE)

    # 5 controls
    s = d.slide()
    d.title(s, "Every claim is checked against two controls and two diagnostics")
    cards = [
        ("Matched no-force fine-tune", "Same base, same 350 clips, same seed, epochs, learning rate, batch and LoRA; block omitted. A gain over it is a force gain, not a fine-tuning gain.", BLUE),
        ("Shuffled force", "The tactile model with each clip's block replaced by another clip's. A gain that survives shuffling is not a force gain.", BLUE),
        ("Force removed", "The tactile adapter with no block at all: does it still rank by what it learned from video?", MUTED),
        ("Video blanked", "Frames mid-gray, force only: what the force carries on its own.", MUTED),
    ]
    for i, (h, body, col) in enumerate(cards):
        x = 0.6 + i * 3.1
        d.card(s, x, 1.6, 2.9, 4.4)
        d.card(s, x + 0.25, 1.85, 0.5, 0.5, fill=col, name=f"dot{i}")
        d.text(s, str(i + 1), x + 0.25, 1.85, 0.5, 0.5, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        d.text(s, h, x + 0.25, 2.5, 2.45, 0.9, size=18, bold=True, font=HEAD_FONT)
        d.text(s, body, x + 0.25, 3.45, 2.45, 2.4, size=14, color=INK)
    d.text(s, "Held-out sets are object-disjoint from training, split by object hash before any model ran, and scored once with fixed settings (threshold 4.0, no checkpoint selection).", 0.6, 6.25, 12.1, 0.6, size=13, color=MUTED)

    # 6 timeline
    s = d.slide()
    d.title(s, "Four stages: each one fixed what the previous one showed was wrong")
    steps = [
        ("Pilot", "32 / 16 HRDexDB clips", "No force benefit. Audit: tiny sample, unaligned sensor and video clocks.", MUTED),
        ("Expanded v2", "618 clips, aligned clocks, frozen object-disjoint splits 350 / 136 / 115", "Still no benefit: 82.4% vs 81.6%. Block found nearly uninformative.", MUTED),
        ("Bugs found", "Inverted closure sign, force saturated at 37.5 units, contact flag always on", "Classifier on rendered lines: AUROC 0.57; on raw tared force: 0.78.", ORANGE),
        ("v3 (primary)", "Tared, peak-ranked, sign-corrected, summary line, classifier P", "85.3% / AUROC 0.926 on dev; 72.2% / 0.800 on test.", BLUE),
        ("Baxter", "FSR fingertips, 50 demos, ACT policy, 10 rollouts", "Zero-shot: force takes AUROC from 0.31 to 1.0.", BLUE),
    ]
    xs = [0.6 + i * 2.5 for i in range(5)]
    for i, (h, what, res, col) in enumerate(steps):
        x = xs[i]
        d.card(s, x, 2.3, 0.5, 0.5, fill=col, name=f"step{i}")
        d.text(s, str(i + 1), x, 2.3, 0.5, 0.5, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        if i < 4:
            d.arrow(s, x + 0.6, 2.55, x + 2.4, 2.55, color=LIGHT)
        d.text(s, h, x, 3.0, 2.3, 0.5, size=18, bold=True, font=HEAD_FONT)
        d.text(s, what, x, 3.5, 2.3, 1.3, size=13, color=MUTED)
        d.text(s, res, x, 4.9, 2.3, 1.3, size=13)
    d.text(s, "Training: released FLORA base, four epochs, seed 17, LoRA r32/a64, lr 2e-5, batch 16, 88 updates, two AMD MI210s. Only the force representation changed between v2 and v3.", 0.6, 6.3, 12.1, 0.6, size=12, color=MUTED)

    # 7 HRDexDB headline
    s = d.slide()
    d.title(s, "HRDexDB: the tactile fine-tune ranks best on every held-out split")
    d.image(s, f_hrdex, 0.6, 1.45, w=8.0, name="hrdexdb auroc chart")
    d.text(s, "Accuracy at the 4.0 threshold", 8.9, 1.55, 3.9, 0.4, size=13, bold=True, color=MUTED)
    rows = [
        ["Model", "dev", "test", "pooled"],
        ["Released", "66.2%", "67.0%", "66.5%"],
        ["Video only", "82.4%", "69.6%", "76.5%"],
        ["Tactile", "85.3%", "72.2%", "79.3%"],
    ]
    d.table(s, rows, 8.9, 2.0, 3.9, [1.4, 0.83, 0.83, 0.84], size=13, bold_col=None)
    d.text(s, "MAE on dev (0 to 100): 150 released, 71 video only, 53 tactile.", 8.9, 3.6, 3.9, 0.8, size=13, color=INK)
    d.text(s, "251 clips over 32 objects never seen in training. Pooled: +12.8 accuracy points and +0.27 AUROC over released FLORA; +2.8 points and +0.043 AUROC over the matched video-only fine-tune.", 8.9, 4.5, 3.9, 2.0, size=13, color=MUTED)

    # 8 statistics
    s = d.slide()
    d.title(s, "The ranking gain over the video-only fine-tune is significant; the accuracy gain is not")
    d.stat(s, "+0.043", "AUROC, tactile minus video-only, pooled over 251 clips", 0.6, 1.6, 3.6)
    d.stat(s, "+0.001 to +0.080", "object-level bootstrap 95% interval, p about 0.02", 4.5, 1.6, 4.4, big_size=36)
    d.stat(s, "26 vs 19", "clips fixed vs clips broken relative to the video-only model", 9.2, 1.6, 3.6, color=INK)
    d.card(s, 0.6, 3.9, 5.9, 2.6)
    d.text(s, "Per split", 0.85, 4.05, 5.4, 0.4, size=14, bold=True, color=MUTED)
    d.bullets(s, [
        "dev: +0.026 AUROC (not significant); 12 wins, 8 losses",
        "test: +0.067 AUROC (p about 0.04); 14 wins, 11 losses",
        "accuracy +2.8 points overall, exact McNemar p = 0.37",
    ], 0.85, 4.5, 5.4, 1.9, size=14, gap=6)
    d.card(s, 6.8, 3.9, 5.9, 2.6)
    d.text(s, "Against released FLORA", 7.05, 4.05, 5.4, 0.4, size=14, bold=True, color=MUTED)
    d.bullets(s, [
        "+12.8 accuracy points and +0.27 AUROC pooled",
        "dev MAE 150 to 53 on the 0 to 100 scale",
        "251 clips cannot certify a 3-point accuracy gain; about 1,000 would",
    ], 7.05, 4.5, 5.4, 1.9, size=14, gap=6)

    # 9 force is read
    s = d.slide()
    d.title(s, "The model reads the force: shuffle it and the gain disappears")
    rows = [
        ["Tactile adapter on dev (136)", "Accuracy", "AUROC", "MAE"],
        ["Real force", "85.3%", "0.926", "53"],
        ["Shuffled force (another clip's block)", "73.5%", "0.824", "109"],
        ["Video blanked (force only)", "53.7%", "0.775", "121"],
        ["Video-only fine-tune, for reference", "82.4%", "0.899", "71"],
    ]
    d.table(s, rows, 0.6, 1.6, 7.6, [4.0, 1.2, 1.2, 1.2], size=14, row_h=0.42)
    d.stat(s, "25 : 9", "paired clips lost : won when the force is shuffled, p = 0.009", 8.7, 1.5, 4.0, color=ORANGE)
    d.card(s, 0.6, 4.2, 12.1, 2.3)
    d.bullets(s, [
        ("Shuffling costs 12 points.", "A gain that survived shuffling would not be a force gain; this one does not survive it."),
        ("Force alone is weak.", "With the video blanked the model still ranks above chance (0.775) but calls almost everything a failure."),
        ("Both channels are used.", "Neither video nor force alone reaches 0.926; the fine-tuned model combines them."),
    ], 0.85, 4.4, 11.6, 2.0, size=14, gap=8)

    # 10 where it helps
    s = d.slide()
    d.title(s, "The force settles the clips the video cannot call, and leaves the rest alone")
    rows = [
        ["Clips, split by the video-only model's confidence", "Video only", "Tactile", "Gain"],
        ["Uncertain: E[s] within 0.75 of the threshold (101 clips)", "66.3%", "72.3%", "+6.0"],
        ["Confident: the other 150 clips", "83.3%", "84.0%", "+0.7"],
        ["dev only, uncertain band (46 clips)", "71.7%", "82.6%", "+10.9"],
    ]
    d.table(s, rows, 0.6, 1.6, 12.1, [6.7, 1.8, 1.8, 1.8], size=14, row_h=0.45, bold_col=2)
    d.card(s, 0.6, 4.0, 5.9, 2.5)
    d.text(s, "What the wins look like", 0.85, 4.15, 5.4, 0.4, size=14, bold=True, color=MUTED)
    d.text(s, "The video-only model under-calls successes whose hold is visually unremarkable (long, steady contact above the median force) and over-calls failures where the hand closes convincingly but the force shows a weak grip, a re-grasp, or contact that ends early.", 0.85, 4.55, 5.4, 1.9, size=13)
    d.card(s, 6.8, 4.0, 5.9, 2.5)
    d.text(s, "Stacking check", 7.05, 4.15, 5.4, 0.4, size=14, bold=True, color=MUTED)
    d.text(s, "A force-only classifier alone scores 76.5% / 0.777 on test, level with both VLMs on accuracy; a logistic stack of video score, tactile score and classifier reaches 88.2% / 0.928 on dev. The VLM, not the sensor, is the bottleneck.", 7.05, 4.55, 5.4, 1.9, size=13)

    # 11 a win
    s = d.slide()
    d.title(s, "A win: standing frame #4, a success the camera scored 1.27")
    d.image(s, VIDEOS / "test_standing_frame_4_success.png", 0.6, 1.5, w=8.4, name="standing frame poster")
    d.card(s, 9.3, 1.5, 3.45, 4.75)
    d.text(s, "HRDexDB test clip", 9.55, 1.65, 3.0, 0.4, size=13, bold=True, color=MUTED)
    d.text(s, "1.27", 9.55, 2.05, 1.4, 0.7, size=34, bold=True, color=RED, font=HEAD_FONT)
    d.text(s, "video only: failure", 9.55, 2.75, 3.0, 0.4, size=13, color=MUTED)
    d.text(s, "4.19", 9.55, 3.2, 1.4, 0.7, size=34, bold=True, color=GREEN, font=HEAD_FONT)
    d.text(s, "tactile: success (truth)", 9.55, 3.9, 3.0, 0.4, size=13, color=MUTED)
    d.text(s, "Force: peak 2.5x the median grasp, 16.4 s of contact, last contact at 23.7 s of 28.2 s.", 9.55, 4.4, 3.0, 1.7, size=13)
    d.text(s, "All 26 such clips are rendered as side-by-side videos in writeup/videos/; the 19 the video-only model wins are listed in writeup/comparison.md.", 0.6, 6.35, 8.4, 0.6, size=12, color=MUTED)

    # 12 Baxter setup
    s = d.slide()
    d.title(s, "Baxter: our own fingertips, our own policy, zero-shot evaluation")
    cards = [
        ("Sensor", "Four thin-film FSR pads in a 3.3 V divider, Teensy 4.0 at 200 Hz, 12-bit ADC, clock-fitted into ROS. Left gripper only; raw counts, resting offset 92."),
        ("Demos", "50 hand-guided left-arm cube picks, 15 fps head and wrist cameras at 640 x 360, no tactile. Cut at the first sign of the gripper re-opening."),
        ("Policy", "ACT (LeRobot 0.6.1): chunk 30, 15 action steps, ResNet-18, VAE; 50k steps on five RTX 2080 Ti. Rollouts use the 30k checkpoint (held-out L1 0.028)."),
        ("Rollouts", "Ten live runs at 30 Hz with joint-jump guard, fingertip box and speed cap, tactile recorded. Operator labels: ep0000 and ep0001 succeeded, eight failed."),
    ]
    for i, (h, body) in enumerate(cards):
        x = 0.6 + i * 3.1
        d.card(s, x, 1.45, 2.9, 2.55)
        d.text(s, h, x + 0.2, 1.6, 2.5, 0.4, size=16, bold=True, font=HEAD_FONT)
        d.text(s, body, x + 0.2, 2.05, 2.5, 1.9, size=12)
    d.image(s, BAXTER / "force_traces.png", 0.6, 4.2, w=6.4, name="force traces")
    d.text(s, "Force and closure the tactile model read, per rollout", 7.3, 4.3, 5.4, 0.4, size=14, bold=True, color=MUTED)
    d.text(s, "Only ep0000 shows a clean grasp signature: closure stops at 0.64 on the cube and force holds at the grasp level for 2.9 s. ep0001 has one contact frame at the very end. The eight failures closed on nothing (closure 1.0, force 0), and five of them then lift the empty gripper exactly as a success would.", 7.3, 4.75, 5.4, 1.9, size=13)

    # 13 Baxter result
    s = d.slide()
    d.title(s, "Baxter: the force block takes the ranking from below chance to perfect")
    d.image(s, f_bax, 0.6, 1.45, w=8.2, name="baxter auroc chart")
    d.stat(s, "0.31 → 1.00", "AUROC, fine-tuned FLORA without and with the force block", 9.1, 1.5, 3.7, big_size=36)
    d.stat(s, "0.375 → 1.00", "released FLORA without and with the block, no training at all", 9.1, 3.3, 3.7, big_size=36)
    d.text(s, "Every condition has accuracy 0.80, the majority class: no model scores a success at or above 4.0, so the signal is the ranking. Shuffled force 0.25, force removed 0.44, force-only classifier 0.50.", 9.1, 5.1, 3.7, 1.6, size=12, color=MUTED)
    d.text(s, "Ten clips, two successes: 16 ordered pairs. A direction, not a significance result. Slurm job 6490, two MI210s, 8 min.", 0.6, 6.5, 8.2, 0.5, size=11, color=MUTED)

    # 14 Baxter per-episode
    s = d.slide()
    d.title(s, "ep0000: the video models cannot see the lift, the force block can")
    d.image(s, BAXTER / "videos" / "ep0000_success.png", 0.6, 1.45, w=7.9, name="ep0000 poster")
    rows = [
        ["Episode", "Video only", "Tactile"],
        ["ep0000 success", "1.34, 6th of 10", "2.41, 1st"],
        ["ep0001 success", "1.05, 8th", "1.60, 2nd"],
        ["ep0003 empty lift", "1.99, 1st", "1.09, 3rd"],
        ["ep0008 empty lift", "1.75, 2nd", "1.04, 6th"],
    ]
    d.table(s, rows, 8.8, 1.45, 3.95, [1.55, 1.3, 1.1], size=12, row_h=0.38, bold_col=2)
    d.text(s, "Expected score and rank among the ten clips. The tactile model's lowest success (1.60) sits 0.51 above its highest failure (1.09).", 8.8, 3.5, 3.95, 1.0, size=12, color=MUTED)
    d.text(s, "ep0003 and ep0008 are the video-only model's two top scores: the arm closes, lifts and moves exactly as in a success. With its force block removed the tactile adapter also puts them on top (2.97, 2.31).", 8.8, 4.6, 3.95, 1.6, size=12)
    d.text(s, "Side-by-side videos for the seven clips where the tactile model wins by at least 0.1: writeup/baxter-rollouts/videos/.", 0.6, 6.4, 7.9, 0.5, size=11, color=MUTED)

    # 15 caveats
    s = d.slide()
    d.title(s, "Caveats")
    items = [
        ("Sample size.", "251 held-out clips cannot certify a 3-point accuracy gain; the AUROC gain is the significant one. Baxter: ten clips, two positives, no significance claim."),
        ("Control provenance.", "The video-only comparison is the v2 control adapter; the v3 control retrain was preempted. Inputs are identical, so only training noise is unmeasured."),
        ("A small classifier rides in the block.", "Its probability is one line the VLM reads; the raw force lines stay. It cannot explain Baxter (its own AUROC there is 0.5)."),
        ("Undertrained.", "Loss floors by update 20 of 88: the target is a template plus one digit. Both fine-tunes are data-limited."),
        ("96 px views.", "All video models saw tiny thumbnails; higher resolution would help every one of them and may shrink the force gain."),
        ("Baxter specifics.", "Normalization fitted on the ten test clips (label-free); ep0001 rests on one line of the block; uncalibrated FSR counts; left pads shown as the right-hand pads the adapter knew."),
    ]
    d.bullets(s, items, 0.6, 1.5, 12.1, 5.3, size=14, gap=10)

    # 16 conclusion (dark)
    s = d.slide(dark=True)
    d.text(s, "Tactile FLORA outperforms regular FLORA at judging grasps", 0.9, 0.7, 11.5, 1.0, size=34, bold=True, color=WHITE, font=HEAD_FONT)
    cols = [
        ("vs released FLORA", "+12.8 pts, +0.27 AUROC on HRDexDB; 0.375 to 1.0 on Baxter"),
        ("vs video-only fine-tune", "+2.8 pts, +0.043 AUROC (p about 0.02); +6 pts where video is unsure; 0.31 to 1.0 on Baxter"),
        ("the force is causal", "shuffled: 25 lost / 9 won (p = 0.009), Baxter 0.25; removed: 0.44"),
    ]
    for i, (h, body) in enumerate(cols):
        x = 0.9 + i * 4.0
        d.text(s, h, x, 2.0, 3.7, 0.5, size=18, bold=True, color=BLUE if i < 2 else ORANGE, font=HEAD_FONT)
        d.text(s, body, x, 2.55, 3.7, 1.6, size=15, color=LIGHT)
    d.text(s, "What's next", 0.9, 4.5, 11.5, 0.5, size=18, bold=True, color=WHITE, font=HEAD_FONT)
    d.bullets(s, [
        "More held-out clips (about 1,000) to certify the accuracy gain, and a second training seed",
        "Higher-resolution video input, then re-measure how much force still adds",
        "Calibrated fingertip sensors on both Baxter grippers and more rollouts, including induced slips",
        "More embodiments: the same text block works for any force stream",
    ], 0.9, 5.05, 11.5, 2.0, size=15, color=LIGHT, gap=4)

    d.save()
    print(OUT, d.n, "slides")


if __name__ == "__main__":
    build()
