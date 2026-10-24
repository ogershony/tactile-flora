"""Force trace rendered as an image frame appended to the clip (the "chart" force modes).

Owns: the numpy line-chart renderer and the shared clip assembly used by training and eval.
A chart is one temporal group (``CHART_FRAMES`` identical frames) appended after the mosaic
frames, so the processor sees ``n + CHART_FRAMES`` frames and the prompt's frame times /
indices are extended by repeating the last entry (see ``prompts.flora.build_prompt``).

Drawing (per present arm, one horizontal panel, L above R): white background; x is the frame
position across the clip; grip force F (red, 0..``f_max`` where 1 is the median training
grasp) and hand closure g (blue, 0..1) as 2 px polylines; a dashed gray line at F = 1; the
contact flag c as a light gray band along the bottom; 1 px black ticks at every 8th frame.
Pure numpy, deterministic, no matplotlib. Requires normalized features (``c`` set).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tactile_flora.schema import Episode, ResampledForce
from tactile_flora.video.mosaic import clip_frames

CHART_FRAMES = 2  # one temporal group of the Qwen3-VL video encoder
CHART_F_MAX = 3.0  # matches RenderConfig.clip_max used by the text block
RED = (230, 60, 60)
BLUE = (60, 100, 230)
GRAY = (150, 150, 150)
BAND = (215, 215, 215)
BLACK = (0, 0, 0)


def _segment(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, colour: tuple) -> None:
    """Rasterize a 2 px thick segment (clipped to the image)."""
    h, w = img.shape[:2]
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    for x, y in zip(xs, ys, strict=True):
        xi, yi = int(round(x)), int(round(y))
        for dy in (0, 1):
            if 0 <= xi < w and 0 <= yi + dy < h:
                img[yi + dy, xi] = colour


def _polyline(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, colour: tuple) -> None:
    for i in range(len(xs) - 1):
        _segment(img, xs[i], ys[i], xs[i + 1], ys[i + 1], colour)


def chart_image(
    features: ResampledForce, shape: tuple[int, int], *, f_max: float = CHART_F_MAX
) -> np.ndarray:
    """Return an (H, W, 3) uint8 chart of ``features`` with one panel per present arm."""
    h, w = shape
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    arms = [(arm, af) for arm in ("L", "R") if (af := features.arms.get(arm)) is not None]
    if not arms:
        raise ValueError("chart needs at least one arm")
    panel_h = h // len(arms)
    margin = 3
    for p, (_arm, af) in enumerate(arms):
        if af.c is None:
            raise ValueError("chart needs normalized features (contact flags missing)")
        top, bottom = p * panel_h + margin, (p + 1) * panel_h - margin - 1
        band_top = bottom - 3
        plot_bottom = band_top - 1
        n = len(af.F)
        xs = np.linspace(margin, w - margin - 1, n) if n > 1 else np.array([w / 2])
        F = np.nan_to_num(np.asarray(af.F, dtype=np.float64))
        g = np.nan_to_num(np.asarray(af.g, dtype=np.float64)) if af.g is not None else None
        c = np.asarray(af.c, dtype=np.int64)

        def y_of(v: np.ndarray, vmax: float, lo: int = plot_bottom, hi: int = top) -> np.ndarray:
            return lo - np.clip(v / vmax, 0.0, 1.0) * (lo - hi)

        # contact band
        for i in range(n):
            if c[i]:
                x0 = int(round(xs[i] - (xs[1] - xs[0]) / 2)) if n > 1 else margin
                x1 = int(round(xs[i] + (xs[1] - xs[0]) / 2)) if n > 1 else w - margin
                img[band_top : bottom + 1, max(margin, x0) : min(w - margin, x1 + 1)] = BAND
        # dashed reference at F = 1
        y1 = int(round(y_of(np.array([1.0]), f_max)[0]))
        for x in range(margin, w - margin, 6):
            img[y1, x : min(x + 3, w - margin)] = GRAY
        # frame ticks
        for i in range(0, n, 8):
            xi = int(round(xs[i]))
            img[band_top - 2 : band_top, xi] = BLACK
        if g is not None:
            _polyline(img, xs, y_of(g, 1.0), BLUE)
        _polyline(img, xs, y_of(F, f_max), RED)
    return img


def append_chart(
    frames: Sequence[np.ndarray], chart: np.ndarray, n: int = CHART_FRAMES
) -> list[np.ndarray]:
    """Return ``frames`` followed by ``n`` copies of ``chart`` (shapes must match)."""
    if frames and frames[0].shape != chart.shape:
        raise ValueError(f"chart shape {chart.shape} differs from frames {frames[0].shape}")
    return [*frames, *([chart] * n)]


def clip_array(episode: Episode, video: Any, reader: Any) -> np.ndarray:
    """(T, H, W, 3) uint8 clip for a ``VideoSpec``: mosaic frames, then the chart group if set.

    ``video`` is a ``prompts.flora.VideoSpec`` (typed loosely to avoid an import cycle). With
    ``video.chart`` the last ``CHART_FRAMES`` entries of ``frame_times`` are the chart's
    placeholders and are not decoded.
    """
    chart = getattr(video, "chart", None)
    times = list(video.frame_times)
    if chart is not None:
        times = times[: len(times) - CHART_FRAMES]
    frames = clip_frames(
        episode,
        reader,
        times,
        video.roles,
        blank=video.blank,
        cell_px=video.cell_px,
    )
    if chart is not None:
        frames = append_chart(frames, chart_image(chart, frames[0].shape[:2]))
    return np.stack(frames)
