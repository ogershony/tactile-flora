"""Mosaic layout, composition and clip sampling: the only place frames become a model input.

Owns: mosaic_layout, compose_mosaic, blank_frames, fit_cell, frame_indices, clip_frames,
video_metadata, visual_token_estimate and the mosaic constants.
Must never: decode video (video/readers.py does) or build prompts.

Rules (FLORA collage): 1 view as is; 2 or 3 side by side (left, center, right); 4 as 2x2 in the
order top, front/bottom, left, right. The scene view is always first, i.e. upper left. Two cell
geometries (``Collage``):

- ``cell`` (legacy, every run before 2026-10-23): each view is fitted (aspect kept, black
  padding) into a square ``cell_px`` cell, 96 px by default. A 4:3 camera reaches the model as
  96x72 pixels, about 1/19 of what the released model was trained on.
- ``flora``: the released input geometry (video/collage.py): each view resized bilinearly, with
  no crop or padding, to its planned cell of about 100k pixels (384x288 for a 4:3 camera), all
  inside 14,336 visual tokens for 64 frames. Views are planned from their source sizes
  (``VideoRef.width``/``height`` when known, else the decoded frame size).

Episode boundaries repeat the nearest available frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from PIL import Image

from tactile_flora.schema import Episode, FrameReader
from tactile_flora.video.collage import plan_flora_collage

Collage = Literal["cell", "flora"]

MOSAIC_CELL_PX = 96  # multiple of 32 nearest the paper's ~105 px per view
MAX_VIEWS = 4
BLANK_GRAY = 128
VISUAL_TOKEN_CAP = 14_336  # paper: a 64-frame clip is capped at 14,336 visual tokens
TOKEN_PX = 32  # 16 px patch x 2x2 spatial merge
TEMPORAL_PATCH = 2
_LAYOUTS: dict[int, tuple[int, int]] = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}


def mosaic_layout(roles: Sequence[str]) -> tuple[tuple[int, int], list[str]]:
    """Pick 1x1/1x2/1x3/2x2 by number of views (max 4), scene first so it is upper left."""
    ordered = sorted(dict.fromkeys(roles), key=lambda r: r != "scene")[:MAX_VIEWS]
    if not ordered:
        raise ValueError("at least one camera role is required")
    return _LAYOUTS[len(ordered)], ordered


def _check_cell(cell_px: int) -> None:
    if cell_px <= 0 or cell_px % TOKEN_PX:
        raise ValueError(f"cell_px must be a positive multiple of {TOKEN_PX}, got {cell_px}")


def fit_cell(frame: np.ndarray, cell_px: int = MOSAIC_CELL_PX) -> np.ndarray:
    """Fit ``frame`` into a (cell, cell, 3) uint8 cell: aspect kept, black padding, top-left."""
    _check_cell(cell_px)
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    img.thumbnail((cell_px, cell_px))
    cell = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)
    cell[: img.height, : img.width] = np.asarray(img)
    return cell


def compose_mosaic(
    frames_by_role: dict[str, np.ndarray],
    layout: tuple[int, int] | None = None,
    *,
    cell_px: int = MOSAIC_CELL_PX,
) -> np.ndarray:
    """Tile one frame per role into a (rows*cell, cols*cell, 3) uint8 mosaic, row-major.

    Cell order is ``mosaic_layout``'s (scene first = upper left). ``layout`` defaults to the one
    ``mosaic_layout`` picks and, when given, must be that grid.
    """
    auto, roles = mosaic_layout(list(frames_by_role))
    rows, cols = layout or auto
    if (rows, cols) != auto:
        raise ValueError(f"layout {(rows, cols)} does not fit {len(roles)} views")
    out = np.zeros((rows * cell_px, cols * cell_px, 3), dtype=np.uint8)
    for k, role in enumerate(roles):
        r, c = divmod(k, cols)
        out[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px] = fit_cell(
            frames_by_role[role], cell_px
        )
    return out


def blank_frames(like: np.ndarray) -> np.ndarray:
    """Mid-gray (128) array shaped like ``like`` (the video_blank ablation)."""
    return np.full_like(np.asarray(like, dtype=np.uint8), BLANK_GRAY)


def frame_indices(episode: Episode, role: str, frame_times: Sequence[float]) -> list[int]:
    """Return the episode frame index of each time: round(t * fps).

    Times are clamped to [0, duration_s] and indices to the declared frames, so boundaries
    repeat the nearest available frame.
    """
    ref = episode.cameras[role]
    last = max(ref.n_frames - 1, 0)
    return [
        min(max(round(min(max(float(t), 0.0), episode.duration_s) * ref.fps), 0), last)
        for t in frame_times
    ]


def _flora_sources(episode: Episode, roles: Sequence[str]) -> tuple[tuple[int, int], ...] | None:
    sizes = []
    for r in roles:
        ref = episode.cameras[r]
        if ref.width is None or ref.height is None:
            return None
        sizes.append((int(ref.width), int(ref.height)))
    return tuple(sizes)


def flora_clip_frames(
    episode: Episode,
    reader: FrameReader,
    frame_times: Sequence[float],
    roles: Sequence[str],
    *,
    blank: bool = False,
) -> list[np.ndarray]:
    """Return the released collage (module docstring, ``flora``) at each of ``frame_times``."""
    _, ordered = mosaic_layout(roles)
    per_view = (
        None
        if blank
        else [reader.read_many(episode, r, frame_indices(episode, r, frame_times)) for r in ordered]
    )
    sources = _flora_sources(episode, ordered)
    if sources is None:
        # sizes from decoded frames; a blank clip (sizes not stored, e.g. REASSEMBLE) decodes the
        # first frame of each view, so its canvas matches the real clip's exactly
        views = per_view or [
            reader.read_many(episode, r, frame_indices(episode, r, frame_times[:1])) for r in ordered
        ]
        sources = tuple((int(v[0].shape[1]), int(v[0].shape[0])) for v in views)
    rects, width, height = plan_flora_collage(sources)
    if per_view is None:
        return [np.full((height, width, 3), BLANK_GRAY, dtype=np.uint8) for _ in frame_times]
    out: list[np.ndarray] = []
    for t in range(len(frame_times)):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for frames, (x, y, w, h) in zip(per_view, rects, strict=True):
            f = np.asarray(frames[t], dtype=np.uint8)
            if f.shape[:2] != (h, w):
                img = Image.fromarray(f).resize((w, h), resample=Image.Resampling.BILINEAR)
                f = np.asarray(img)
            canvas[y : y + h, x : x + w] = f
        out.append(canvas)
    return out


def clip_frames(
    episode: Episode,
    reader: FrameReader,
    frame_times: Sequence[float],
    roles: Sequence[str],
    *,
    blank: bool = False,
    cell_px: int = MOSAIC_CELL_PX,
    collage: Collage = "cell",
) -> list[np.ndarray]:
    """One mosaic frame per time in ``frame_times`` (64 for a dense clip, m for outcome).

    Times outside [0, duration] repeat the nearest available frame. ``blank`` skips decoding and
    returns mid-gray frames of the same shape. ``collage`` picks the geometry (module docstring).
    """
    if collage == "flora":
        return flora_clip_frames(episode, reader, frame_times, roles, blank=blank)
    layout, ordered = mosaic_layout(roles)
    shape = (layout[0] * cell_px, layout[1] * cell_px, 3)
    _check_cell(cell_px)
    if blank:
        return [np.full(shape, BLANK_GRAY, dtype=np.uint8) for _ in frame_times]
    cells: dict[str, list[np.ndarray]] = {}
    for role in ordered:
        idx = frame_indices(episode, role, frame_times)
        cells[role] = [fit_cell(f, cell_px) for f in reader.read_many(episode, role, idx)]
    return [
        compose_mosaic({r: cells[r][t] for r in ordered}, layout, cell_px=cell_px)
        for t in range(len(frame_times))
    ]


def source_frame_count(episode: Episode) -> int:
    """Frames of the scene (or first) camera, the clock for video metadata."""
    cam = episode.cameras.get("scene") or next(iter(episode.cameras.values()))
    return cam.n_frames


class VideoMetadataDict(TypedDict):
    """Frame-clock fields accepted by the Transformers video processor."""

    total_num_frames: int
    fps: float
    frames_indices: list[int]
    duration: float


def video_metadata(frame_indices_: Sequence[int], fps: float, n_frames: int) -> VideoMetadataDict:
    """Frame-clock metadata so the processor's timestamp tokens are absolute episode seconds."""
    return {
        "total_num_frames": int(n_frames),
        "fps": float(fps),
        "frames_indices": [int(i) for i in frame_indices_],
        "duration": float(n_frames) / float(fps),
    }


def visual_token_estimate(n_frames: int, mosaic_hw: tuple[int, int]) -> int:
    """Visual tokens of a clip: ceil(n/2) temporal groups x (H/32) x (W/32), capped at 14,336."""
    h, w = mosaic_hw
    tokens = (
        math.ceil(n_frames / TEMPORAL_PATCH) * math.ceil(h / TOKEN_PX) * math.ceil(w / TOKEN_PX)
    )
    return min(tokens, VISUAL_TOKEN_CAP)
