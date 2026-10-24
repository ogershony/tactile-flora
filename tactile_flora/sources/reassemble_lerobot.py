"""Source adapter for the LeRobot v3 port of REASSEMBLE (Hugging Face ``robot-lev/reassemble``).

Owns: the port's layout (parquet frames + concatenated mp4) and its conversion to the episode
clock, including the segment labels the raw adapter gets from labels/mapping.py.
Must never: filter episodes beyond "present on disk" in discover(), write to disk, or invent
anomaly spans that the dataset does not have.

Why the labels live here: ``verify.windowing.assemble_labels`` only runs ``labels/mapping.py``
for ``mapping.MAPPED_SOURCES`` (``reassemble`` is listed, this source is not), so load() emits
the same labels the raw adapter ends up with after mapping (outcome, dense failure spans) plus
the raw adapter's native event label; ``native_labels()`` returns that same list.

Resolved facts (verified on the local copy, 2026-10-22):
- Layout: one LeRobot v3 dataset under ``<root>/`` (meta/info.json, meta/episodes/*.parquet,
  meta/tasks.parquet, meta/splits.json, meta/stats.json, data/chunk-*/file-*.parquet,
  videos/<key>/chunk-*/file-*.mp4). 149 episodes = the 149 TU Wien recordings, 1,433,406
  frames at 30 fps. Episode id = ``str(episode_index)``. License CC-BY-4.0 (as the archive).
- Present: discover() lists an episode when its row, its data parquet AND its hama1 mp4 exist,
  so a partial download yields a usable subset (the count is visibly below 149).
- Clock: ``timestamp`` restarts at 0.0 for every episode and steps exactly 1/30 s; the first
  frame time is subtracted anyway and kept in ``meta["clock_offset_s"]``.
- Force: ``observation.force.base`` (3) + ``observation.torque`` (3) -> WristArmForce arm "R",
  channel wrist, raw units, 30 Hz on the frame clock (the port nearest-neighbour downsampled
  the ~1.7 kHz F/T); ``observation.force`` and ``observation.torque.base`` are the fallbacks and
  ``meta["force_keys"]`` records the pair used. Measured over all 149 recordings
  (scripts/measure_reassemble_force_keys.py, 2026-10-22): ``observation.force`` is the raw
  sensor reading with the tool weight in it (|mean| over idle frames 13.6 N median, IQR
  12.9-14.3 N, along sensor -z), ``observation.force.base`` is gravity-compensated (0.43 N
  median, IQR 0.33-0.56 N; 7 recordings above 1 N, mostly in the last idle segment of the
  README's ft_invalid_last_action files). Neither key separates failed from successful segments
  by magnitude (per-segment p99 AUROC 0.60 raw vs 0.58 base, p50 0.52 vs 0.55); ``.base`` is
  preferred because |F| then measures contact rather than |gravity + contact| and the
  normalization scale (grasp-frame median |F| minus free-space median) stays positive: 0.26 N
  median, <= 0 in 7 of 149 recordings, against -0.08 N and <= 0 in 77 of 149 with the raw
  key, which ``normalize.fit`` could not use. Torque keeps the
  sensor-frame column: ``observation.torque.base`` has standing offsets of 5-7.4 N m in 26
  recordings (21 on 2025-01-09) where the sensor torque is 0.15 N m, so it is not a compensated
  wrist torque (meaning UNVERIFIED). Torque is not used by any feature (resample takes the force
  magnitude), only by the sensor-health check. This differs from the raw adapter, which prefers
  ``measured_force`` over ``compensated_base_force``.
- Gripper: ``observation.state.gripper_position`` (2, both Franka fingers, identical, 0..0.04)
  averaged over the fingers; closure = 1 - w / max(w) with max(w) from meta/stats.json (episode
  max if absent), clipped to 0..1. Larger width = more open is assumed, as in the raw adapter.
- Cameras: ``observation.images.hama1`` = scene, ``hama2`` = scene_2, ``hand`` = wrist_right
  (role names as the raw adapter); a camera whose mp4 is not on disk is omitted, a missing
  scene file raises FileNotFoundError. ``event_cam`` is not exposed. v3 mp4 files are
  concatenated: the in-file start of each camera is ``meta["video_from_s"][role]``. Missing
  hand frames are black-filled by the port, so every camera has ``len(frames)`` frames.
- Segments: ``segment.index`` is constant over a run of frames and ``segment.success`` /
  ``task_index`` are constant within a segment (checked over all 149 episodes; one episode has a
  one-frame segment out of index order, so segments are runs of equal index in frame order,
  not groups). Segment ``t0`` is its first frame time, ``t1`` the next segment's first frame
  time (last segment: last frame + 1/fps), so segments tile the episode half-open.
  Text = meta/tasks.parquet row of ``task_index`` (69 tasks). ``No action.`` segments are idle:
  kept in ``meta["segments"]`` and in time, excluded from labels and ``action_success``
  (145/149 recordings start and end with one; one idle segment is flagged failed).
- Labels (mirror of ``labels.mapping.map_reassemble`` + the raw ``native_labels``): outcome 5
  if every non-idle segment succeeded else 1; dense quality-1 spans with the segment text for
  the failed segments; one event label with a span per non-idle segment (3 success / 1 fail).
  131 of 149 recordings have a failed segment, hence ``expected_label_rate`` 0.88 in the config.
- Splits: meta/splits.json maps ``str(episode_index)`` to {"recording": <h5 stem>, "split":
  "train" | "test" | "unassigned"} (111 / 37 / 1; the unassigned one is 2025-01-10-16-17-40, a
  hand-cam-missing recording). "train" -> train; "test" is held out and divided into val/test
  with the raw adapter's session hash (``session_split`` over the recording date, the seed and
  the configured val:test ratio; never train); anything else -> train (the raw adapter trains
  on every recording). Both port splits contain recordings of every date, so a date (session)
  does straddle train and held-out here; val and test stay date-disjoint.
- Session = recording date (``recording[:10]``) as in the raw adapter; ``KNOWN_ISSUES`` of the
  raw adapter are looked up by recording stem.

Segment mode (``SourceConfig.episode_unit: segment``, the ``reassemble_seg`` source; added
2026-10-22 for the failure judge): every qualifying segment is its own Episode on its own clock
(zero at the segment's first frame). A segment qualifies when its task is not idle and its last
frame is at least ``segment_select.min_duration_s`` (1.0 s) after its first; per port split every
failed segment is kept plus up to ``success_multiple`` (2) times as many successes, drawn after a
seeded shuffle of the successes in recording-then-segment order, capped at ``max_total`` for the
split (450 on test). That is exactly the clip set of data/colab/e0b_reassemble_force.py (seed 0;
37 test recordings, 1,110 qualifying segments, 133 failures, 399 clips; measured on the local port
2026-10-22). Episode id ``<episode_index>_seg<segment.index:03d>``; task = the segment text;
cameras point at the recording's mp4 with ``video_from_s`` advanced by the segment's first frame
and ``n_frames`` = the segment's frame count, so video/readers.py reads the segment's span; force,
gripper and times are the recording's slices; the single label is outcome 5 (success) / 1
(failure) and ``meta["prompt_kinds"] = ["outcome"]`` suppresses dense windows. Splits follow the
port directly ("test" -> test, anything else -> train; no val hashing). discover() lists only the
SELECTED segments (a documented exception to "present on disk": the unselected ones belong to
recordings the recording-level source verifies in full). ``norm_group`` stays "all": the
recording-level fit is one pooled group (data/verified/reassemble_lerobot/norm.json) and a
segment's first 0.5 s is not free space, so segments cannot be fitted on their own; with
``verify_from: reassemble_lerobot`` the segment source reuses those constants and the parent's
recording-level checks (see verify.run_verify).

meta keys: segments (list of {index, task_index, text, idle, t0, t1, frame0, frame1, success}),
segment_boundaries_s (list[float], the t0 of every segment plus the last t1), action_success
(list[bool], non-idle segments in order), known_issues (list[str]), recording (str), port_split
(str | None), episode_index (int), clock_offset_s, video_from_s, force_keys (list[str], the
force and torque columns read), hf_repo.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tactile_flora.config import SegmentSelect, SourceConfig
from tactile_flora.schema import (
    Episode,
    ForceStream,
    GripperStream,
    Label,
    Span,
    VideoRef,
    WristArmForce,
)
from tactile_flora.sources._common import (
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
    session_split,
)
from tactile_flora.sources._lerobot import DEFAULT_DATA_PATH, LeRobotV3, column_array
from tactile_flora.sources.base import register_source, strip_outcome_words
from tactile_flora.sources.reassemble import KNOWN_ISSUES, LICENSE

HF_REPO = "robot-lev/reassemble"
# Preference order per channel; the first column present is used (see the docstring).
FORCE_KEYS = ("observation.force.base", "observation.force")
TORQUE_KEYS = ("observation.torque", "observation.torque.base")
GRIPPER_KEY = "observation.state.gripper_position"
SEGMENT_KEY = "segment.index"
SUCCESS_KEY = "segment.success"
TASK_KEY = "task_index"
SCENE_KEY = "observation.images.hama1"
CAMERA_ROLES = {
    SCENE_KEY: "scene",
    "observation.images.hama2": "scene_2",
    "observation.images.hand": "wrist_right",
}
IDLE_TASK = "no action"
HELD_OUT_SPLIT = "test"
DOWNLOAD_HINT = "download the port first: hf download {repo} --repo-type dataset --local-dir {root}"
SEGMENT_COLUMNS = ("episode_index", "frame_index", "timestamp", SEGMENT_KEY, SUCCESS_KEY, TASK_KEY)
RECORDING_CACHE = 2  # parsed recordings kept in memory (segments of one recording load in a row)


def is_idle(text: str) -> bool:
    """Return True for the port's idle task text ("No action.", any case or punctuation)."""
    return text.strip().rstrip(".!").strip().lower() == IDLE_TASK


def read_tasks(root: Path) -> dict[int, str]:
    """Return task_index -> task text from meta/tasks.parquet (task as the index or a column)."""
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return {}
    df = pq.read_table(path).to_pandas()
    texts = df["task"].tolist() if "task" in df.columns else list(df.index)
    return {int(i): str(t) for i, t in zip(df["task_index"].tolist(), texts, strict=True)}


def read_splits(root: Path) -> dict[int, dict[str, Any]]:
    """Return episode_index -> {"recording", "split"} from meta/splits.json (empty if absent)."""
    path = root / "meta" / "splits.json"
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text())
    out: dict[int, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and str(k).lstrip("-").isdigit():
                out[int(k)] = {"recording": v.get("recording"), "split": v.get("split")}
    return out


def frame_segments(
    seg_index: np.ndarray, success: np.ndarray, task_index: np.ndarray, t: np.ndarray, fps: float
) -> list[dict[str, Any]]:
    """Split frames into runs of equal ``segment.index`` (frame order), half-open in time.

    A segment succeeds only if every one of its frames is flagged successful. ``frame0`` /
    ``frame1`` are the run's first frame and one past its last frame.
    """
    n = int(t.size)
    if n == 0:
        return []
    change = np.flatnonzero(seg_index[1:] != seg_index[:-1]) + 1
    starts = np.concatenate([[0], change]).astype(int)
    ends = np.concatenate([change, [n]]).astype(int)
    out: list[dict[str, Any]] = []
    for a, b in zip(starts, ends, strict=True):
        t1 = float(t[b]) if b < n else float(t[n - 1]) + 1.0 / fps
        out.append(
            {
                "index": int(seg_index[a]),
                "task_index": int(task_index[a]),
                "t0": float(t[a]),
                "t1": t1,
                "frame0": int(a),
                "frame1": int(b),
                "success": bool(np.all(success[a:b])),
            }
        )
    return out


def segment_labels(segments: list[dict[str, Any]]) -> list[Label]:
    """Outcome, dense failure spans and the event label over the non-idle segments.

    Mirrors ``labels.mapping.map_reassemble`` (outcome 5 if all succeed else 1; a quality-1
    span per failed segment, rationale = segment text) plus the raw adapter's native event
    label (a span per segment, 3 if success else 1). Idle segments contribute nothing.
    """
    acts = [s for s in segments if not s.get("idle", False) and s["t1"] >= s["t0"]]
    if not acts:
        return []
    labels = [
        Label(kind="outcome", origin="source", outcome=5 if all(s["success"] for s in acts) else 1)
    ]
    failed = [
        Span(t0=s["t0"], t1=s["t1"], quality=1, rationale=str(s["text"]))
        for s in acts
        if not s["success"]
    ]
    if failed:
        labels.append(Label(kind="dense", origin="source", spans=failed))
    labels.append(
        Label(
            kind="event",
            origin="source",
            spans=[
                Span(
                    t0=s["t0"],
                    t1=s["t1"],
                    quality=3 if s["success"] else 1,
                    rationale=str(s["text"]),
                )
                for s in acts
            ],
        )
    )
    return labels


def holdout_split(session: str, cfg: SourceConfig) -> str:
    """Return val or test for a held-out session: the raw hash rule at the val:test ratio.

    ``cfg.split_rules`` may carry ``val_fraction`` and ``test_fraction``; only their ratio
    matters here because nothing held out may go to train.
    """
    rules: dict[str, Any] = cfg.split_rules
    val_f = float(rules.get("val_fraction", DEFAULT_VAL_FRACTION))
    test_f = float(rules.get("test_fraction", DEFAULT_TEST_FRACTION))
    total = val_f + test_f
    if total <= 0:
        return "test"
    pick = session_split(session, cfg.seed, val_f / total, test_f / total)
    return "val" if pick == "val" else "test"


def port_split(raw: str | None, session: str, cfg: SourceConfig) -> str:
    """Map the port's split string to train/val/test ("test" is held out, anything else trains)."""
    return holdout_split(session, cfg) if raw == HELD_OUT_SPLIT else "train"


def _data_path(ds: LeRobotV3, idx: int) -> Path:
    rec = ds.record(idx)
    if "_data_file" in rec:
        return Path(rec["_data_file"])
    tmpl = ds.info.get("data_path") or DEFAULT_DATA_PATH
    return ds.root / tmpl.format(
        chunk_index=int(rec["data/chunk_index"]), file_index=int(rec["data/file_index"])
    )


def segment_episode_id(episode_index: int, segment_index: int) -> str:
    """Segment-mode episode id ``<episode_index>_seg<segment.index:03d>``."""
    return f"{episode_index}_seg{segment_index:03d}"


def parse_segment_id(episode_id: str) -> tuple[int, int]:
    """Inverse of ``segment_episode_id``; ValueError for a recording-level id."""
    rec, sep, seg = episode_id.partition("_seg")
    if not sep or not rec.isdigit() or not seg.isdigit():
        raise ValueError(f"not a segment episode id: {episode_id!r}")
    return int(rec), int(seg)


def segment_qualifies(seg: dict[str, Any], t: np.ndarray, min_duration_s: float) -> bool:
    """E0b rule: not idle, and the last frame at least ``min_duration_s`` after the first."""
    if seg.get("idle", False):
        return False
    a, b = int(seg["frame0"]), int(seg["frame1"])
    return b > a and float(t[b - 1]) - float(t[a]) >= min_duration_s


def select_segments(items: list[tuple[str, bool]], rule: SegmentSelect, split: str) -> list[str]:
    """Balanced pick for one split: every failure plus the E0b share of shuffled successes.

    ``items`` are (episode id, success) in recording-then-segment order. The successes are
    shuffled with ``random.Random(rule.seed)`` and the first
    ``min(len(succ), max(0, max_total - len(fails)), success_multiple * len(fails))`` kept
    (the ``max_total`` term only when the split has a positive cap), exactly as
    data/colab/e0b_reassemble_force.py does with ``random.seed(0)``. Order: failures first (in
    enumeration order), then the picked successes in shuffled order.
    """
    fails = [eid for eid, ok in items if not ok]
    succ = [eid for eid, ok in items if ok]
    random.Random(rule.seed).shuffle(succ)
    n = min(len(succ), int(rule.success_multiple * len(fails)))
    cap = int(rule.max_total.get(split, 0))
    if cap > 0:
        n = min(n, max(0, cap - len(fails)))
    return fails + succ[:n]


@dataclass
class _Recording:
    """One recording parsed onto the episode clock (the cache unit of segment mode)."""

    idx: int
    fps: float
    t: np.ndarray  # frame times, zero at the first frame
    wrench: np.ndarray  # (N, 6): fx fy fz tx ty tz
    closure: np.ndarray | None
    cameras: dict[str, tuple[Path, float]]  # role -> (mp4 path, in-file start of frame 0)
    segments: list[dict[str, Any]]
    recording: str
    split: str | None
    offset: float
    force_keys: list[str]


@register_source("reassemble_lerobot")
class ReassembleLerobotAdapter:
    """Adapter for the LeRobot v3 port of REASSEMBLE (one episode per recording or segment)."""

    name = "reassemble_lerobot"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._ds: LeRobotV3 | None = None
        self._ids: list[str] = []
        self._tasks: dict[int, str] = {}
        self._splits: dict[int, dict[str, Any]] = {}
        self._cache: dict[int, _Recording] = {}
        self._seg_index: dict[int, list[dict[str, Any]]] = {}
        self._seg_split: dict[str, str] = {}

    @property
    def segment_mode(self) -> bool:
        """True when the config asks for one episode per segment."""
        return self.cfg.episode_unit == "segment"

    def discover(self, root: Path) -> list[str]:
        """Return the episode ids under ``root``.

        Recording mode: every recording present on disk (row + data parquet + scene mp4).
        Segment mode: the selected qualifying segments of those recordings.
        """
        self._root = root
        self._ds, self._ids, self._tasks, self._splits = None, [], {}, {}
        self._cache, self._seg_index, self._seg_split = {}, {}, {}
        if not (root / "meta" / "info.json").exists():
            return []
        ds = self._dataset()
        recordings = [
            idx
            for idx in ds.episode_indices()
            if _data_path(ds, idx).exists() and ds.video(idx, SCENE_KEY)[0].exists()
        ]
        if not self.segment_mode:
            self._ids = [str(i) for i in recordings]
            return list(self._ids)
        self._seg_index = self._read_segment_index(recordings)
        per_split: dict[str, list[tuple[str, bool]]] = {"train": [], "test": []}
        for idx in recordings:
            _, raw = self._recording(idx)
            split = "test" if raw == HELD_OUT_SPLIT else "train"
            for seg in self._seg_index[idx]:
                if seg["qualifies"]:
                    per_split[split].append(
                        (segment_episode_id(idx, seg["index"]), bool(seg["success"]))
                    )
        for split, items in per_split.items():
            for eid in select_segments(items, self.cfg.segment_select, split):
                self._ids.append(eid)
                self._seg_split[eid] = split
        return list(self._ids)

    def _read_segment_index(self, recordings: list[int]) -> dict[int, list[dict[str, Any]]]:
        """Segments of every recording from the label columns only (no force or video)."""
        ds = self._dataset()
        out: dict[int, list[dict[str, Any]]] = {}
        by_file: dict[Path, list[int]] = {}
        for idx in recordings:
            by_file.setdefault(_data_path(ds, idx), []).append(idx)
        for path, idxs in by_file.items():
            schema_names = set(pq.read_schema(path).names)
            cols = [c for c in SEGMENT_COLUMNS if c in schema_names]
            table = pq.read_table(path, columns=cols)
            for idx in idxs:
                df = table.filter(pc.field("episode_index") == idx).to_pandas()
                if "frame_index" in df.columns:
                    df = df.sort_values("frame_index").reset_index(drop=True)
                t = np.asarray(df["timestamp"], dtype=np.float64)
                t = t - (float(t[0]) if t.size else 0.0)
                out[idx] = self._segments(df, t, ds.fps)
        return out

    def _segments(self, df: Any, t: np.ndarray, fps: float) -> list[dict[str, Any]]:
        n = len(df)
        segments = frame_segments(
            np.asarray(df[SEGMENT_KEY]).reshape(-1) if SEGMENT_KEY in df.columns else np.zeros(n),
            np.asarray(df[SUCCESS_KEY]).reshape(-1) if SUCCESS_KEY in df.columns else np.ones(n),
            np.asarray(df[TASK_KEY]).reshape(-1) if TASK_KEY in df.columns else np.full(n, -1),
            t,
            fps,
        )
        for s in segments:
            s["text"] = self._tasks.get(s["task_index"], f"task_{s['task_index']}")
            s["idle"] = is_idle(s["text"])
            s["qualifies"] = segment_qualifies(s, t, self.cfg.segment_select.min_duration_s)
        return segments

    def _dataset(self) -> LeRobotV3:
        if self._ds is None:
            self._ds = LeRobotV3(self._root)
            self._tasks = read_tasks(self._root)
            self._splits = read_splits(self._root)
        return self._ds

    def _ensure(self) -> None:
        if not self._ids:
            self.discover(self._root)

    def _recording(self, idx: int) -> tuple[str, str | None]:
        """Return (recording stem, port split string) of an episode."""
        sp = self._splits.get(idx, {})
        rec = sp.get("recording")
        split = sp.get("split")
        return (str(rec) if rec else f"episode_{idx}", str(split) if split is not None else None)

    @staticmethod
    def _session(recording: str) -> str:
        return recording[:10]

    def _load_recording(self, idx: int) -> _Recording:
        """Parse one recording onto its episode clock (cached, last RECORDING_CACHE)."""
        if idx in self._cache:
            return self._cache[idx]
        ds = self._dataset()
        df = ds.frames(idx)
        n = len(df)
        if n == 0:
            raise ValueError(f"{self.name}/{idx}: no frames in {_data_path(ds, idx)}")
        fps = ds.fps
        t_raw = np.asarray(df["timestamp"], dtype=np.float64)
        offset = float(t_raw[0])
        t = t_raw - offset
        fkey = next((k for k in FORCE_KEYS if k in df.columns), FORCE_KEYS[0])
        tkey = next((k for k in TORQUE_KEYS if k in df.columns), TORQUE_KEYS[0])
        w = np.concatenate([column_array(df, fkey)[:, :3], column_array(df, tkey)[:, :3]], axis=1)
        closure: np.ndarray | None = None
        if GRIPPER_KEY in df.columns:
            width = column_array(df, GRIPPER_KEY).mean(axis=1)
            stat_max = ds.stats().get(GRIPPER_KEY, {}).get("max")
            hi = float(np.max(stat_max)) if stat_max is not None else float(np.max(width))
            closure = np.clip(1.0 - width / hi, 0.0, 1.0) if hi > 0 else np.zeros_like(width)
        cameras: dict[str, tuple[Path, float]] = {}
        declared = ds.video_keys()
        for key, role in CAMERA_ROLES.items():
            if key not in declared:
                continue
            path, from_ts = ds.video(idx, key)
            if not path.exists():
                if key == SCENE_KEY:
                    raise FileNotFoundError(
                        f"{path} missing; " + DOWNLOAD_HINT.format(repo=HF_REPO, root=self._root)
                    )
                continue
            cameras[role] = (path, from_ts)
        recording, split = self._recording(idx)
        rec = _Recording(
            idx=idx,
            fps=fps,
            t=t,
            wrench=w,
            closure=closure,
            cameras=cameras,
            segments=self._segments(df, t, fps),
            recording=recording,
            split=split,
            offset=offset,
            force_keys=[fkey, tkey],
        )
        if len(self._cache) >= RECORDING_CACHE:
            del self._cache[next(iter(self._cache))]
        self._cache[idx] = rec
        return rec

    @staticmethod
    def _streams(
        rec: _Recording, a: int, b: int
    ) -> tuple[np.ndarray, ForceStream, GripperStream | None]:
        """Force and gripper streams over frames ``a:b`` on a clock that is zero at frame a."""
        t = rec.t[a:b] - rec.t[a]
        w = rec.wrench[a:b]
        force = ForceStream(
            channel="wrist",
            arms={
                "R": WristArmForce(
                    fx=w[:, 0].tolist(),
                    fy=w[:, 1].tolist(),
                    fz=w[:, 2].tolist(),
                    tx=w[:, 3].tolist(),
                    ty=w[:, 4].tolist(),
                    tz=w[:, 5].tolist(),
                )  # fmt: skip
            },
            rate_hz=rec.fps,
            t=t.tolist(),
        )
        gripper: GripperStream | None = None
        if rec.closure is not None:
            gripper = GripperStream(
                arms={"R": rec.closure[a:b].tolist()}, rate_hz=rec.fps, t=t.tolist()
            )
        return t, force, gripper

    @staticmethod
    def _cameras(
        rec: _Recording, start_s: float, n: int
    ) -> tuple[dict[str, VideoRef], dict[str, float]]:
        """VideoRefs of ``n`` frames starting ``start_s`` into the recording, plus video_from_s."""
        cams = {
            r: VideoRef(path=str(p), fps=rec.fps, n_frames=n) for r, (p, _) in rec.cameras.items()
        }
        from_s = {r: from_ts + start_s for r, (_, from_ts) in rec.cameras.items()}
        return cams, from_s

    def load(self, episode_id: str) -> Episode:
        """Load one recording (or, in segment mode, one segment) on its own clock."""
        self._ensure()
        if self.segment_mode:
            idx, seg_index = parse_segment_id(episode_id)
            return self._segment_episode(episode_id, self._load_recording(idx), seg_index)
        return self._recording_episode(episode_id, self._load_recording(int(episode_id)))

    def _recording_episode(self, episode_id: str, rec: _Recording) -> Episode:
        n = rec.t.size
        t, force, gripper = self._streams(rec, 0, n)
        cameras, video_from = self._cameras(rec, 0.0, n)
        segments = [{k: v for k, v in s.items() if k != "qualifies"} for s in rec.segments]
        actions = [s for s in segments if not s["idle"]]
        task = actions[0]["text"] if actions else rec.recording
        meta: dict[str, Any] = {
            "segments": segments,
            "segment_boundaries_s": [s["t0"] for s in segments]
            + ([segments[-1]["t1"]] if segments else []),
            "action_success": [s["success"] for s in actions],
            "known_issues": list(KNOWN_ISSUES.get(rec.recording, [])),
            "recording": rec.recording,
            "port_split": rec.split,
            "episode_index": rec.idx,
            "clock_offset_s": rec.offset,
            "video_from_s": video_from,
            "force_keys": list(rec.force_keys),
            "hf_repo": HF_REPO,
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=self._session(rec.recording),
            task=strip_outcome_words(task),
            duration_s=float(t[-1]),
            cameras=cameras,
            force=force,
            gripper=gripper,
            labels=segment_labels(segments),
            license=LICENSE,
            meta=meta,
        )

    def _segment_episode(self, episode_id: str, rec: _Recording, seg_index: int) -> Episode:
        """One qualifying segment as an Episode (clock zero at its first frame)."""
        hits = [s for s in rec.segments if s["index"] == seg_index and s["qualifies"]]
        if len(hits) != 1:
            raise KeyError(f"{self.name}/{episode_id}: {len(hits)} qualifying segments of index")
        seg = hits[0]
        a, b = int(seg["frame0"]), int(seg["frame1"])
        t, force, gripper = self._streams(rec, a, b)
        cameras, video_from = self._cameras(rec, a / rec.fps, b - a)
        meta: dict[str, Any] = {
            "recording": rec.recording,
            "port_split": rec.split,
            "episode_index": rec.idx,
            "parent_episode_id": str(rec.idx),
            "segment_index": seg_index,
            "segment": {k: v for k, v in seg.items() if k != "qualifies"},
            "segment_t0_s": float(rec.t[a]),
            "known_issues": list(KNOWN_ISSUES.get(rec.recording, [])),
            "clock_offset_s": rec.offset + float(rec.t[a]),
            "video_from_s": video_from,
            "force_keys": list(rec.force_keys),
            "hf_repo": HF_REPO,
            "prompt_kinds": ["outcome"],
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=self._session(rec.recording),
            task=strip_outcome_words(str(seg["text"])),
            duration_s=float(t[-1]),
            cameras=cameras,
            force=force,
            gripper=gripper,
            labels=[Label(kind="outcome", origin="source", outcome=5 if seg["success"] else 1)],
            license=LICENSE,
            meta=meta,
        )

    def splits(self) -> dict[str, list[str]]:
        """Return train/val/test ids.

        Recording mode: the port's split, its held-out "test" hashed into val/test. Segment
        mode: the port's split as is (test -> test, anything else -> train).
        """
        self._ensure()
        self._dataset()
        out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        if self.segment_mode:
            for eid in self._ids:
                out[self._seg_split[eid]].append(eid)
            return out
        for eid in self._ids:
            recording, split = self._recording(int(eid))
            out[port_split(split, self._session(recording), self.cfg)].append(eid)
        return out

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return the labels load() puts on the Episode (outcome only for a segment)."""
        return list(self.load(episode_id).labels)


@register_source("reassemble_seg")
class ReassembleSegAdapter(ReassembleLerobotAdapter):
    """The segment-level view of the port as its own source (configs/sources/reassemble_seg.yaml).

    Same data, one Episode per selected segment; ``cfg.episode_unit`` must be ``segment``.
    """

    name = "reassemble_seg"

    def __init__(self, cfg: SourceConfig) -> None:
        if cfg.episode_unit != "segment":
            raise ValueError(f"{self.name} needs episode_unit: segment, got {cfg.episode_unit!r}")
        super().__init__(cfg)
