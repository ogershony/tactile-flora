"""Source adapter for PHANTOM, now published as HapticWAM (`armteam/hapticwam-teleop-raw`).

Owns: the loose HapticWAM layout on disk and its conversion to the episode clock.
Must never: filter episodes in discover(), write to disk, or leak layout details elsewhere.

Resolved open questions (verified 2026-10-21, docs/research/datasets.md; NOT LeRobot):
- Layout: ``<root>/tasks/<task>/ep_<task>_<epoch>_<idx>/`` with ``meta.json`` plus one zarr group
  per stream, each holding ``data`` (T,...) and ``ts`` (T,) master-clock seconds (not resampled).
  ``<root>/manifests/all.jsonl`` rows: episode, task, success, failure_demo, tactile_contact,
  split, session, operator, duration_s, peak_force_N, max_contact_mm2, path. Episode id is the
  episode directory name; the task folder is its parent directory.
- Scene video: ``camera_scene_color`` is per-frame JPEG bytes in a zarr VLenBytes array at 15 Hz,
  NOT an mp4. VideoRef has no ``kind`` field, so the convention is: a ``VideoRef.path`` that is
  a directory ending in ``camera_scene_color`` is a zarr JPEG source; the frame reader must open
  it with zarr and decode ``data[i]`` (JPEG) with ``ts`` as frame times. fps=15, n_frames=T.
- Force: ``tactile_{left,right}_wrench`` (T,6) [Fx,Fy,Fz,Mx,My,Mz] in N and N*m at ~6 Hz on two
  independent clocks. The UR3 is a single arm so the stream has arm "R" only, with
  f_a = |Fz| of the LEFT DM-Tac pad and f_b = |Fz| of the RIGHT pad, both nearest-sampled onto a
  common 10 Hz grid (``FORCE_RATE_HZ``) starting at episode time 0. Samples before a pad's first
  reading take that first reading (nearest). Pad-free deployments have no ``tactile_*`` group:
  load() then returns an empty force stream (never filtered here; verify/ decides).
  Distributed fields (fx,fy,fz) are raw uncalibrated SDK units and are not used. ``arm_ft`` is a
  current-based estimate, NOT a real F/T sensor, and is ignored.
- Gripper: ``gripper`` (T,2) at ~95 Hz, column 0 is position 0 open .. 1 closed (column 1 is
  Robotiq gOBJ, ignored). Arm "R".
- Success caveats: v4 ``*_fail`` demos (over-squeeze / induced slip) carry success=true; the
  batch_20260822 ``*_fail`` (under-grasp) carry success=false. ``failure_demo`` (manifest) is the
  reliable flag and failure demos are forced into the train split. ``tactile_contact=false`` means
  the grasp landed off the pads. Whiteboard rows have blank stats. batch_20260822 lives only in
  ``archive/`` and is not discovered.
- License: CC-BY-4.0 (all HapticWAM repos). Session = manifest ``session``.
- Native labels: none are emitted here; the raw flags live in ``meta`` for labels/mapping.py.
- Unverified: whether newer exports use zarr v2 or v3 stores (zarr>=3 reads both).

meta keys read by labels/mapping.py: success, failure_demo, task_folder, tactile_contact, object,
peak_force_N. ``exclude_from_norm_fit`` (= failure_demo) tells verify to leave the episode out
of the normalization fit (force/normalize.py). Also provided: operator, max_contact_mm2, tags,
manifest_split, clock_offset_s.
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
    VideoRef,
)
from tactile_flora.sources._common import median_rate, nearest_index, py
from tactile_flora.sources.base import register_source, strip_outcome_words
from tactile_flora.video.zarr_compat import open_group

LICENSE = "CC-BY-4.0"
SCENE_STREAM = "camera_scene_color"
SCENE_FPS = 15.0
FORCE_RATE_HZ = 10.0
LEFT_STREAM = "tactile_left_wrench"
RIGHT_STREAM = "tactile_right_wrench"
FZ_COLUMN = 2
GRIPPER_STREAM = "gripper"
MANIFEST_SPLITS = {"train": "train", "val": "val", "test": "test"}


def _stream_path(ep_dir: Path, name: str) -> Path:
    """Resolve published ``*.zarr`` streams and legacy extensionless exports."""
    published = ep_dir / f"{name}.zarr"
    return published if published.exists() else ep_dir / name


def _read_group(ep_dir: Path, name: str) -> tuple[Any, np.ndarray] | None:
    """Open one zarr stream group; return (data array handle, ts float64) or None if absent."""
    path = _stream_path(ep_dir, name)
    if not path.exists():
        return None
    group: Any = open_group(path)
    return group["data"], np.asarray(group["ts"][:], dtype=np.float64)


@register_source("phantom")
class PhantomAdapter:
    """Adapter for HapticWAM (formerly PHANTOM) loose per-episode zarr episodes."""

    name = "phantom"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._dirs: dict[str, Path] = {}
        self._manifest: dict[str, dict[str, Any]] = {}

    # -- discovery --------------------------------------------------------------------
    def discover(self, root: Path) -> list[str]:
        """Return every episode directory name under ``<root>/tasks/*/`` (no filtering)."""
        self._root = root
        self._dirs = {p.name: p for p in sorted(root.glob("tasks/*/ep_*")) if p.is_dir()}
        self._manifest = {}
        manifest = root / "manifests" / "all.jsonl"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._manifest[str(row["episode"])] = row
        return list(self._dirs)

    def _ensure(self) -> None:
        if not self._dirs:
            self.discover(self._root)

    # -- loading ----------------------------------------------------------------------
    def load(self, episode_id: str) -> Episode:
        """Load one episode on the episode clock (zero at the first scene frame)."""
        self._ensure()
        ep_dir = self._dirs[episode_id]
        meta_json: dict[str, Any] = json.loads((ep_dir / "meta.json").read_text())
        row = self._manifest.get(episode_id, {})
        task_folder = ep_dir.parent.name

        scene = _read_group(ep_dir, SCENE_STREAM)
        if scene is None:
            raise FileNotFoundError(f"{ep_dir} has no {SCENE_STREAM} stream")
        scene_data, scene_ts = scene
        offset = float(scene_ts[0])
        duration = float(scene_ts[-1] - offset)

        grid = np.arange(0.0, duration + 1e-9, 1.0 / FORCE_RATE_HZ)
        pads: list[np.ndarray] = []
        native_saturation: dict[str, float] = {}
        native_max_counts: dict[str, int] = {}
        native_timing: dict[str, dict[str, float | int]] = {}
        for name in (LEFT_STREAM, RIGHT_STREAM):
            stream = _read_group(ep_dir, name)
            if stream is None:
                pads = []
                break
            data, ts = stream
            fz = np.abs(np.asarray(data[:], dtype=np.float64)[:, FZ_COLUMN])
            key = f"R/{'f_a' if name == LEFT_STREAM else 'f_b'}"
            if ts.size > 1:
                native_timing[key] = {
                    "samples": int(ts.size),
                    "median_interval_s": float(np.median(np.diff(ts))),
                    "max_interval_s": float(np.max(np.diff(ts))),
                    "first_s": float(ts[0] - offset),
                    "last_s": float(ts[-1] - offset),
                }
            # Health statistics must use independent observations, before nearest sampling
            # duplicates a single maximum onto several points of the common force grid.
            if fz.size:
                native_saturation[key] = float(np.mean(fz == fz.max()))
                native_max_counts[key] = int(np.count_nonzero(fz == fz.max()))
            pads.append(fz[nearest_index(ts - offset, grid)])
        if pads:
            force = ForceStream(
                channel="pad",
                arms={"R": PadArmForce(f_a=pads[0].tolist(), f_b=pads[1].tolist())},
                rate_hz=FORCE_RATE_HZ,
                t=grid.tolist(),
            )
        else:
            force = ForceStream(
                channel="pad", arms={"R": PadArmForce(f_a=[], f_b=[])}, rate_hz=FORCE_RATE_HZ, t=[]
            )

        gripper: GripperStream | None = None
        g = _read_group(ep_dir, GRIPPER_STREAM)
        if g is not None:
            g_data, g_ts = g
            pos = np.clip(np.asarray(g_data[:], dtype=np.float64)[:, 0], 0.0, 1.0)
            t_g = g_ts - offset
            gripper = GripperStream(
                arms={"R": pos.tolist()}, rate_hz=median_rate(t_g), t=t_g.tolist()
            )

        task_text = str(meta_json.get("text") or task_folder)
        success = meta_json.get("success", row.get("success"))
        failure_demo = bool(row.get("failure_demo", task_folder.endswith("_fail")))
        meta: dict[str, Any] = {
            "success": success,
            "failure_demo": failure_demo,
            "task_folder": task_folder,
            "tactile_contact": row.get("tactile_contact"),
            "object": meta_json.get("object")
            or row.get("object")
            or task_folder.removesuffix("_fail"),
            "peak_force_N": row.get("peak_force_N"),
            "max_contact_mm2": row.get("max_contact_mm2"),
            "operator": meta_json.get("operator", row.get("operator")),
            "tags": py(meta_json.get("tags", [])),
            "clock_offset_s": offset,
            "manifest_split": row.get("split"),
            # failure demos (over-squeeze, under-grasp) must not define the "typical grasp"
            "exclude_from_norm_fit": failure_demo,
            "native_saturation": native_saturation,
            "native_max_counts": native_max_counts,
            "native_timing": native_timing,
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=str(row.get("session", "unknown")),
            task=strip_outcome_words(task_text),
            duration_s=duration,
            cameras={
                "scene": VideoRef(
                    path=str(_stream_path(ep_dir, SCENE_STREAM).resolve()),
                    fps=SCENE_FPS,
                    n_frames=int(scene_data.shape[0]),
                )
            },
            force=force,
            gripper=gripper,
            labels=[],
            license=LICENSE,
            meta=meta,
        )

    # -- splits and labels ------------------------------------------------------------
    def splits(self) -> dict[str, list[str]]:
        """Return manifest splits; failure demos are forced into train."""
        self._ensure()
        out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for ep in self._dirs:
            row = self._manifest.get(ep, {})
            failure = bool(row.get("failure_demo", self._dirs[ep].parent.name.endswith("_fail")))
            split = MANIFEST_SPLITS.get(str(row.get("split", "train")), "train")
            out["train" if failure else split].append(ep)
        return out

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return no labels: the raw success flags are in meta for labels/mapping.py."""
        return []
