"""Windows sidecar for benchmarks that cannot be rebuilt from a source adapter.

Owns: CachedWindow, write_windows_cache, read_windows_cache and the sidecar path convention
``<manifest>.windows.parquet`` (next to ``<manifest>.parquet``).
Must never: hold force data (cached windows are video + task + target only, so they serve the
force-free benchmarks such as roboreward_500) or decide which windows are accepted.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from tactile_flora.schema import Episode, ForceStream, VideoRef, Window


class CachedWindow(BaseModel):
    """One eval window with everything needed to prompt a model, minus force."""

    window_id: str
    source: str
    episode_id: str
    session: str = ""
    task: str
    prompt_kind: str
    anchor_s: float
    frame_times: list[float]
    target: str
    split: str = "benchmark"
    video_path: str
    fps: float
    n_frames: int
    object: str = ""
    license: str = "unknown"


def cache_path(manifest_path: Path | str) -> Path:
    """Return the sidecar path ``<manifest stem>.windows.parquet`` next to the manifest."""
    p = Path(manifest_path)
    return p.with_name(p.stem + ".windows.parquet")


def write_windows_cache(path: Path | str, items: list[CachedWindow]) -> Path:
    """Write ``items`` to ``path`` (parent created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {f: [getattr(i, f) for i in items] for f in CachedWindow.model_fields}
    schema = pa.schema(
        [
            (f, pa.list_(pa.float64()) if f == "frame_times" else _ARROW[f])
            for f in CachedWindow.model_fields
        ]
    )
    pq.write_table(pa.table(cols, schema=schema), out)
    return out


_ARROW: dict[str, pa.DataType] = {
    "anchor_s": pa.float64(),
    "fps": pa.float64(),
    "n_frames": pa.int64(),
}
for _f in CachedWindow.model_fields:
    _ARROW.setdefault(_f, pa.string())


def read_windows_cache(path: Path | str) -> list[CachedWindow]:
    """Read a windows sidecar."""
    return [CachedWindow.model_validate(r) for r in pq.read_table(path).to_pylist()]


def to_episode_window(item: CachedWindow) -> tuple[Episode, Window]:
    """Rebuild a force-free (Episode, Window) pair from a cached row."""
    ep = Episode(
        source=item.source,
        episode_id=item.episode_id,
        session=item.session,
        task=item.task,
        duration_s=item.n_frames / item.fps,
        cameras={"scene": VideoRef(path=item.video_path, fps=item.fps, n_frames=item.n_frames)},
        force=ForceStream(channel="pad", arms={}, rate_hz=1.0, t=[]),
        license=item.license,
        meta={"object": item.object},
    )
    win = Window(
        episode_ref=f"{item.source}/{item.episode_id}",
        prompt_kind=item.prompt_kind,  # type: ignore[arg-type]
        anchor_s=item.anchor_s,
        frame_times=tuple(item.frame_times),
        force_lines=(),
        target=item.target,
        split=item.split,  # type: ignore[arg-type]
    )
    return ep, win
