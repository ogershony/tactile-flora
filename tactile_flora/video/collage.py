"""FLORA-native multi-view collage plan: the released model's input geometry.

Owns: plan_flora_collage, TARGET_PIXELS_PER_VIEW, ALIGN, MAX_VISUAL_TOKENS.
Must never: decode video or build prompts.

The released Flora-Rewarder reads a collage in which every camera view is resized (bilinear, no
crop, padding or gutter) to a 32-pixel-aligned cell of about 100k pixels: one view as is, two or
three side by side, four as a 2x2 grid, all inside 14,336 visual tokens for a 64-frame clip. This
module reproduces the cell sizes of the planner shipped with the released weights
(``batch_utils/core.py``, ``plan_collage``) with the same objective: per view, squared aspect
distortion in percent over 4 plus squared log-area error over 0.32, a hard penalty beyond 4%
distortion, heights 128-512 px, widths within four 32-px steps of the aspect-preserving width.
tests/unit/test_collage.py checks it against the shipped planner whenever a snapshot is present.
On HRDexDB's 2048x1536 cameras every cell is 384x288 for one to four views; the legacy repo
mosaic (video/mosaic.py, ``MOSAIC_CELL_PX`` = 96) gave each view about 1/19 of those pixels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

ALIGN = 32
TARGET_PIXELS_PER_VIEW = 100_000
# The released engine's per-clip pixel budget (batch_utils/engine.py COLLAGE_VIDEO_SIZE, equal to
# the video_processor size in processor_config.json). video_preprocessor_config.json says
# 25,165,824 instead, which would shrink a 4-view 64-frame clip (28.3 M pixels) by ~6%, so every
# processor call (training, HF and vLLM inference) passes this explicitly.
VIDEO_SIZE = {"shortest_edge": 4096, "longest_edge": 29_360_128}
MAX_VISUAL_TOKENS = 14_336
MAX_VIEWS = 4

Size = tuple[int, int]  # (width, height)
Rect = tuple[int, int, int, int]  # (x, y, width, height)


def _distortion(src: Size, w: int, h: int) -> float:
    return 100.0 * abs((w / h) / (src[0] / src[1]) - 1.0)


def _cell_loss(src: Size, w: int, h: int) -> float:
    d = _distortion(src, w, h)
    area = abs(math.log((w * h) / TARGET_PIXELS_PER_VIEW))
    return (d / 4.0) ** 2 + (area / 0.32) ** 2 + 10_000.0 * max(0.0, d - 4.0) ** 2


def _widths(src: Size, h: int) -> list[int]:
    center = round(src[0] / src[1] * h / ALIGN) * ALIGN
    return sorted(
        {center + k * ALIGN for k in range(-4, 5) if 2 * ALIGN <= center + k * ALIGN <= 32 * ALIGN}
    )


def _heights() -> range:
    return range(4 * ALIGN, 17 * ALIGN, ALIGN)


def _tokens(w: int, h: int) -> int:
    # 64 frames -> 32 temporal groups of (w / 32) x (h / 32) merged patches
    return 32 * (w // ALIGN) * (h // ALIGN)


def _single(src: Size) -> Size:
    best = min((_cell_loss(src, w, h), w, h) for h in _heights() for w in _widths(src, h))
    return best[1], best[2]


def _row(sources: Sequence[Size]) -> tuple[list[Size], int, int]:
    cands = []
    for h in _heights():
        ws = [min(_widths(s, h), key=lambda w, s=s: _cell_loss(s, w, h)) for s in sources]
        loss = sum(_cell_loss(s, w, h) for s, w in zip(sources, ws, strict=True))
        if _tokens(sum(ws), h) <= MAX_VISUAL_TOKENS:
            cands.append((loss, [(w, h) for w in ws], sum(ws), h))
    if not cands:
        raise ValueError("no side-by-side layout fits the visual token budget")
    _, cells, width, height = min(cands)
    return cells, width, height


def _row_options(a: Size, b: Size) -> dict[int, tuple[float, int, int, int]]:
    opts: dict[int, tuple[float, int, int, int]] = {}
    for h in _heights():
        for w1 in _widths(a, h):
            for w2 in _widths(b, h):
                loss = _cell_loss(a, w1, h) + _cell_loss(b, w2, h)
                cur = opts.get(w1 + w2)
                if cur is None or loss < cur[0]:
                    opts[w1 + w2] = (loss, h, w1, w2)
    return opts


def _grid(sources: Sequence[Size]) -> tuple[list[Rect], int, int]:
    top, bottom = _row_options(sources[0], sources[1]), _row_options(sources[2], sources[3])
    cands = []
    for width in top.keys() & bottom.keys():
        t, b = top[width], bottom[width]
        if _tokens(width, t[1] + b[1]) <= MAX_VISUAL_TOKENS:
            cands.append((t[0] + b[0], width, t, b))
    if not cands:
        raise ValueError("no 2 x 2 layout fits the visual token budget")
    _, width, (_, th, tl, tr), (_, bh, bl, br) = min(cands)
    rects = [(0, 0, tl, th), (tl, 0, tr, th), (0, th, bl, bh), (bl, th, br, bh)]
    return rects, width, th + bh


@cache
def plan_flora_collage(sources: tuple[Size, ...]) -> tuple[tuple[Rect, ...], int, int]:
    """Return ``(rects, width, height)``: one ``(x, y, w, h)`` cell per view, in view order."""
    if not 1 <= len(sources) <= MAX_VIEWS:
        raise ValueError(f"1 to {MAX_VIEWS} camera views are supported, got {len(sources)}")
    if len(sources) == 1:
        w, h = _single(sources[0])
        return ((0, 0, w, h),), w, h
    if len(sources) in (2, 3):
        cells, width, height = _row(sources)
        rects, x = [], 0
        for w, h in cells:
            rects.append((x, 0, w, h))
            x += w
        return tuple(rects), width, height
    rects, width, height = _grid(sources)
    return tuple(rects), width, height
