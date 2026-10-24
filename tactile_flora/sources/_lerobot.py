"""Minimal reader for LeRobot v3 datasets (parquet + concatenated mp4) without the lerobot package.

Owns: parsing meta/info.json, meta/episodes/*.parquet and data/chunk-*/file-*.parquet, and
resolving per-episode video files. Must never: decode video, write to disk, or know which
dataset it is reading (field choices belong to the adapters).

Video note: v3 concatenates many episodes into one mp4; ``video()`` therefore returns the file
path plus the episode's ``from_timestamp`` inside it (adapters store that in ``meta``).

Data files are read whole and the last ``TABLE_CACHE`` of them kept in memory (an RH20T file is
120-150 MB in Arrow and holds 500-800 episodes; filtering it in memory costs 10 ms where a
filtered parquet read costs 250 ms, and verify loads every episode four times).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tactile_flora.sources._common import py

DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TABLE_CACHE = 2  # data files kept in memory per dataset (most recently read)
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


class LeRobotV3:
    """Read-only view of one LeRobot v3 dataset directory (the dir holding meta/ and data/)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.info: dict[str, Any] = json.loads((root / "meta" / "info.json").read_text())
        self.fps = float(self.info.get("fps", 30))
        self._records = self._read_episode_records()
        self._tables: dict[Path, pa.Table] = {}

    # -- metadata ---------------------------------------------------------------------
    def _read_episode_records(self) -> dict[int, dict[str, Any]]:
        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        records: dict[int, dict[str, Any]] = {}
        if files:
            for f in files:
                df = pq.read_table(f).to_pandas()
                for row in df.to_dict("records"):
                    records[int(row["episode_index"])] = {k: py(v) for k, v in row.items()}
            return records
        # Fallback: no episode table; scan the data files for episode indices.
        for f in sorted((self.root / "data").rglob("*.parquet")):
            col = pq.read_table(f, columns=["episode_index"]).column("episode_index")
            for idx in sorted({int(v) for v in col.to_pylist()}):
                records[idx] = {"episode_index": idx, "_data_file": str(f)}
        return records

    def episode_indices(self) -> list[int]:
        """Return all episode indices in ascending order."""
        return sorted(self._records)

    def record(self, idx: int) -> dict[str, Any]:
        """Return the episode-table row for ``idx`` as a plain dict."""
        return self._records[idx]

    def feature_names(self, key: str) -> list[str]:
        """Return the declared per-dimension names of feature ``key`` (empty if unknown)."""
        names = self.info.get("features", {}).get(key, {}).get("names")
        if isinstance(names, dict):  # some exports nest {"motors": [...]}
            names = next(iter(names.values()), None)
        return [str(n) for n in names] if isinstance(names, list) else []

    def video_keys(self) -> list[str]:
        """Return all feature keys whose dtype is video, in declaration order."""
        feats: dict[str, Any] = self.info.get("features", {})
        return [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "video"]

    def stats(self) -> dict[str, Any]:
        """Return meta/stats.json (empty dict if absent)."""
        p = self.root / "meta" / "stats.json"
        return json.loads(p.read_text()) if p.exists() else {}

    # -- frames and videos ------------------------------------------------------------
    def frames(self, idx: int) -> pd.DataFrame:
        """Return the per-frame rows of episode ``idx``, sorted by frame_index if present."""
        rec = self._records[idx]
        if "_data_file" in rec:
            path = Path(rec["_data_file"])
        else:
            tmpl = self.info.get("data_path") or DEFAULT_DATA_PATH
            path = self.root / tmpl.format(
                chunk_index=int(rec["data/chunk_index"]), file_index=int(rec["data/file_index"])
            )
        df = self._table(path).filter(pc.field("episode_index") == idx).to_pandas()
        if "frame_index" in df.columns:
            df = df.sort_values("frame_index").reset_index(drop=True)
        return df

    def _table(self, path: Path) -> pa.Table:
        """Return the whole data file at ``path`` (cached, most recent TABLE_CACHE files)."""
        if path not in self._tables:
            if len(self._tables) >= TABLE_CACHE:
                del self._tables[next(iter(self._tables))]
            self._tables[path] = pq.read_table(path)
        return self._tables[path]

    def video(self, idx: int, key: str) -> tuple[Path, float]:
        """Return (mp4 path, from_timestamp seconds inside that file) for one camera."""
        rec = self._records[idx]
        tmpl = self.info.get("video_path") or DEFAULT_VIDEO_PATH
        path = self.root / tmpl.format(
            video_key=key,
            chunk_index=int(rec.get(f"videos/{key}/chunk_index", 0)),
            file_index=int(rec.get(f"videos/{key}/file_index", 0)),
        )
        return path, float(rec.get(f"videos/{key}/from_timestamp", 0.0))


def column_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return column ``col`` as an (N, D) float64 array (scalars become D=1)."""
    arr = np.asarray([np.asarray(v, dtype=np.float64) for v in df[col]], dtype=np.float64)
    return arr.reshape(len(df), -1)
