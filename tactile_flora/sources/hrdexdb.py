"""Source adapter for HRDexDB (`HRDexDB/HRDexDB`, eval only).

Owns: how HRDexDB trials are laid out on disk and how one trial becomes an Episode. Maps
fingertip tactile of the allegro_v5 and inspire_dftp hands to the pad channel (the two
most-loaded fingertip channels, by mean absolute value over the trial) on arm "R".
Must never: filter episodes in discover(), write to disk, be used for training (eval only), or
leak layout details to other modules.

Resolved open question ("HRDexDB tactile layout per hand type and which hands to include"),
verified 2026-10-21 from the HF API and file headers (docs/research/datasets.md section 6):
- Layout: ``<hand>/<object>/<scene_id>/{grasp_result.json, C2R.npy, cam_param/,
  vid/<camera_serial>.mp4 (about 23 cameras), raw/{arm,hand,timestamps}/*.npy}``
  plus ``metadata/episodes.parquet``
  (episode_id, embodiment, object_id, sequence_id, split, success, has_tactile, num_frames,
  fps=30). Hand dirs: human, allegro_v5, inspire_dftp, inspire_f1, robotiq_2f85.
- Episode id used here: ``"<hand>/<object>/<scene_id>"`` (the trial directory relative to root).
- allegro_v5: ``raw/hand/tactile.npy`` int32 (T, 4), one scalar per fingertip; time base
  ``raw/hand/time.npy``. INCLUDED.
- inspire_dftp: ``raw/hand/force.npy`` int64 (T, 6), one scalar per DOF/finger. INCLUDED.
- inspire_f1: ``raw/hand/right_tactile.npy`` is a pickled per-frame object array whose internal
  layout is NOT verified; load() raises NotImplementedError. robotiq_2f85: no tactile or force
  file at all; load() raises NotImplementedError. human: no robot; load() raises. All three are
  still returned by discover() (the verifier, not the adapter, decides what to keep).
- Units and per-finger semantics are undocumented ("sensor specifications vary across
  platforms"); uint16-range samples are decoded as signed int16 before abs() is taken. Values
  remain in raw sensor units and are NOT comparable across hands.
- Time bases: every stream has its own time file (``<stem>_time.npy`` or ``time.npy`` next to
  the data, else ``raw/timestamps/<stem>.npy``); if none is found a uniform ``fps`` grid
  (episodes.parquet fps, default 30) is assumed. Streams are aligned by nearest sample.
- Gripper: no gripper scalar exists; the hand closure proxy is the mean over columns of
  ``raw/hand/position.npy``, min/max-normalized per trial to 0 (open) .. 1 (closed). This is a
  PROXY: it assumes larger joint values mean more closed, and is flagged in
  ``meta["gripper_is_proxy"]``.
- Cameras: the first camera serial (sorted) in ``vid/`` is used as "scene". UNVERIFIED which
  serial is a sensible scene view; ``cam_param/`` or cameras.parquet may carry roles.
- Clock: legacy runs assume video starts with the earliest used sensor sample. Opt in with
  ``hrdexdb_use_video_timestamps`` to map sensor time to encoded video time using the recorded
  ``raw/timestamps/{timestamp,frame_id}.npy``. The upstream viewer uses one-based frame IDs;
  both start offset and capture/encoded clock-rate differences are accounted for. Missing or
  incomplete timelines raise rather than silently reverting to the legacy assumption.
- Native label: outcome 5 if ``grasp_result.json`` grasp_success is true, else 1 (binary;
  criterion in the paper: lifted and held 10 s). No label when success is null/absent (human).
- Session / split key: hand type. Every episode goes to split "benchmark".
- License: CC-BY-NC-4.0. Metadata may predate the robotiq upload (2026-09-20).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Episode,
    ForceStream,
    GripperStream,
    Label,
    PadArmForce,
)
from tactile_flora.sources._eval_common import nearest_index, probe_video, rate_from_times
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "CC-BY-NC-4.0"
DEFAULT_FPS = 30.0
# hand -> (tactile file, time-file stem)
_TACTILE_FILES = {"allegro_v5": "tactile", "inspire_dftp": "force"}
_UNSUPPORTED = {
    "inspire_f1": "raw/hand/right_tactile.npy is a pickled per-frame object array with an "
    "unverified internal layout",
    "robotiq_2f85": "it has no tactile or force file",
    "human": "there is no robot, hence no tactile or force stream",
}


def _read_time(trial: Path, stem: str, n: int, fps: float) -> np.ndarray:
    """Return the time base (seconds) of raw/hand/<stem>.npy, or a uniform grid."""
    candidates = [
        trial / "raw" / "hand" / f"{stem}_time.npy",
        trial / "raw" / "hand" / "time.npy",
        trial / "raw" / "timestamps" / f"{stem}.npy",
        trial / "raw" / "timestamps" / "hand.npy",
    ]
    for path in candidates:
        if path.is_file():
            t = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
            if len(t) == n:
                return t
    return np.arange(n, dtype=np.float64) / fps


def _signed_force(values: np.ndarray) -> np.ndarray:
    """Decode unsigned 16-bit bit patterns used for signed force samples.

    HRDexDB's ``force.npy`` files are often serialized as int64 despite containing values in
    the uint16 range. Values above 32767 encode negative int16 readings (for example 65517 is
    -19); taking abs before decoding turns those small readings into artificial full-scale
    forces. Preserve already-signed arrays and values outside the uint16 range.
    """
    raw = np.asarray(values)
    out = np.asarray(raw, dtype=np.float64)
    if np.issubdtype(raw.dtype, np.unsignedinteger) or (
        out.size and float(np.nanmin(out)) >= 0 and float(np.nanmax(out)) <= 65535
    ):
        out = np.where(out >= 32768, out - 65536, out)
    return out


def video_clock(trial: Path, sensor_time: np.ndarray, fps: float, n_frames: int):
    """Map recorded sensor time to encoded video seconds using upstream one-based frame IDs.

    Only sensor samples inside the recorded video interval are retained. Missing internal
    frame timestamps are interpolated; offsets and clock-rate differences are both handled.
    The upstream viewer uses frame_id - 1 as the zero-based video frame position.
    """
    directory = trial / "raw/timestamps"
    timestamps = np.load(directory / "timestamp.npy", allow_pickle=False).reshape(-1)
    frames = np.load(directory / "frame_id.npy", allow_pickle=False).reshape(-1)
    if (
        len(timestamps) < 2
        or len(timestamps) != len(frames)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(frames).all()
        or not (np.diff(timestamps) > 0).all()
        or not (np.diff(frames) > 0).all()
        or not np.equal(frames, np.floor(frames)).all()
        or frames[0] != 1
        or frames[-1] != n_frames
        or fps <= 0
    ):
        raise ValueError(f"invalid or incomplete recorded video timeline: {trial}")
    mask = (sensor_time >= timestamps[0]) & (sensor_time <= timestamps[-1])
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"sensor stream does not overlap recorded video: {trial}")
    mapped = np.interp(sensor_time[mask], timestamps, (frames - 1) / fps)
    if not (np.diff(mapped) > 0).all():
        raise ValueError(f"non-increasing sensor/video clock mapping: {trial}")
    return (
        mask,
        mapped,
        {
            "method": "recorded_video_timestamps",
            "video_start_epoch_s": float(timestamps[0]),
            "video_start_minus_sensor_start_s": float(timestamps[0] - sensor_time[0]),
            "capture_span_s": float(timestamps[-1] - timestamps[0]),
            "encoded_span_s": float((frames[-1] - 1) / fps),
            "timestamp_rows": len(timestamps),
            "missing_frame_timestamps": int(n_frames - len(frames)),
            "max_capture_gap_s": float(np.diff(timestamps).max()),
            "samples_outside_video": int(len(sensor_time) - mask.sum()),
        },
    )


@register_source("hrdexdb")
class HrdexdbAdapter:
    """Adapter for HRDexDB (eval only)."""

    name = "hrdexdb"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._ids: list[str] | None = None
        self._fps: dict[str, float] = {}

    def discover(self, root: Path) -> list[str]:
        """Return every trial id ``<hand>/<object>/<scene_id>`` under ``root`` (no filtering)."""
        self._root = Path(root)
        ids: list[str] = []
        for hand in sorted(p for p in self._root.iterdir() if p.is_dir()):
            for obj in sorted(p for p in hand.iterdir() if p.is_dir()):
                for scene in sorted(p for p in obj.iterdir() if p.is_dir()):
                    if (scene / "grasp_result.json").is_file() or (scene / "raw").is_dir():
                        ids.append(f"{hand.name}/{obj.name}/{scene.name}")
        self._ids = ids
        self._load_fps()
        return list(ids)

    def _load_fps(self) -> None:
        """Read per-trial fps from metadata/episodes.parquet when present (best effort)."""
        path = self._root / "metadata" / "episodes.parquet"
        self._fps = {}
        if not path.is_file():
            return
        import pandas as pd

        df = pd.read_parquet(path)
        needed = {"embodiment", "object_id", "sequence_id", "fps"}
        if not needed <= set(df.columns):
            return
        keys = zip(df["embodiment"], df["object_id"], df["sequence_id"], strict=True)
        for key, fps in zip(keys, df["fps"].astype(float).tolist(), strict=True):
            self._fps["/".join(str(k) for k in key)] = fps

    def _trial(self, episode_id: str) -> Path:
        return self._root / episode_id

    def _ensure(self) -> None:
        if self._ids is None:
            self.discover(self._root)

    def _result(self, episode_id: str) -> dict[str, Any]:
        path = self._trial(episode_id) / "grasp_result.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}

    def load(self, episode_id: str) -> Episode:
        """Load one trial on the episode clock (allegro_v5 and inspire_dftp only)."""
        self._ensure()
        hand, obj, scene = episode_id.split("/")
        if hand in _UNSUPPORTED:
            raise NotImplementedError(
                f"hrdexdb hand {hand!r} is not loadable: {_UNSUPPORTED[hand]}; "
                "verify/ decides what to keep, so discover() still lists it"
            )
        if hand not in _TACTILE_FILES:
            raise NotImplementedError(f"hrdexdb hand {hand!r} has no known tactile layout")
        trial = self._trial(episode_id)
        fps = self._fps.get(episode_id, DEFAULT_FPS)
        stem = _TACTILE_FILES[hand]
        raw = np.abs(_signed_force(np.load(trial / "raw" / "hand" / f"{stem}.npy")))
        raw = raw.reshape(len(raw), -1)
        t_raw = _read_time(trial, stem, len(raw), fps)

        pos = np.asarray(np.load(trial / "raw" / "hand" / "position.npy"), dtype=np.float64)
        pos = pos.reshape(len(pos), -1)
        t_pos = _read_time(trial, "position", len(pos), fps)

        offset = float(min(t_raw[0], t_pos[0]))
        t = t_raw - offset
        top = np.argsort(-raw.mean(axis=0), kind="stable")[:2]
        top = np.concatenate([top, top[:1]]) if len(top) == 1 else top
        f_a, f_b = raw[:, top[0]], raw[:, top[1]]

        closure = pos.mean(axis=1)
        span = float(closure.max() - closure.min())
        closure = (closure - closure.min()) / span if span > 0 else np.zeros_like(closure)
        g = closure[nearest_index(t_pos - offset, t)]

        vids = sorted((trial / "vid").glob("*.mp4"))
        if not vids:
            raise FileNotFoundError(f"no videos under {trial / 'vid'}")
        # SourceRootReader prepends the configured root to relative VideoRef paths. Resolve here
        # because ``trial`` already includes that root (which may itself be a relative path).
        scene_ref = probe_video(vids[0].resolve())
        alignment = {"method": "legacy_sensor_origin"}
        if self.cfg.hrdexdb_use_video_timestamps:
            mask, t, alignment = video_clock(trial, t_raw, scene_ref.fps, scene_ref.n_frames or 0)
            f_a, f_b, g = f_a[mask], f_b[mask], g[mask]
            offset = alignment["video_start_epoch_s"]
        duration = scene_ref.n_frames / scene_ref.fps if scene_ref.n_frames else float(t[-1])
        rate = rate_from_times(t, fps)
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=hand,
            task=strip_outcome_words(f"grasp and lift the {obj.replace('_', ' ')}"),
            duration_s=duration,
            cameras={"scene": scene_ref},
            force=ForceStream(
                channel="pad",
                arms={"R": PadArmForce(f_a=f_a.tolist(), f_b=f_b.tolist())},
                rate_hz=rate,
                t=t.tolist(),
            ),
            gripper=GripperStream(arms={"R": g.tolist()}, rate_hz=rate, t=t.tolist()),
            labels=self.native_labels(episode_id),
            license=LICENSE,
            meta={
                "clock_offset_s": offset,
                "hand": hand,
                "object": obj,
                "scene_id": scene,
                "camera_serial": vids[0].stem,
                "tactile_file": f"raw/hand/{stem}.npy",
                "tactile_channels": [int(top[0]), int(top[1])],
                "force_units": "raw sensor units (undocumented), signed-int16 decoded then abs",
                "gripper_is_proxy": True,
                "fps": fps,
                "clock_alignment": alignment,
            },
        )

    def splits(self) -> dict[str, list[str]]:
        """Eval only: every episode is in split "benchmark"."""
        self._ensure()
        return {"benchmark": list(self._ids or [])}

    def native_labels(self, episode_id: str) -> list[Label]:
        """Binary outcome (5 if grasp_success else 1); empty when success is unknown."""
        self._ensure()
        success = self._result(episode_id).get("grasp_success")
        if success is None:
            return []
        return [Label(kind="outcome", origin="source", outcome=5 if success else 1)]
