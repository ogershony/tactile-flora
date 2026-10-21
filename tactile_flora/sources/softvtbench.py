"""Source adapter for SoftVTBench (`Arthur12137/SoftVTBench`, eval only).

Owns: how SoftVTBench HDF5 + MP4 demos are laid out and how one demo becomes an Episode. Maps
the simulated per-finger contact force to the pad channel on arm "R".
Must never: filter episodes in discover(), write to disk, be used for training (eval only), or
leak layout details to other modules.

Resolved open question ("which FEM array gives per-finger contact force, and the deformation
tolerance per object used for DSR"), verified 2026-10-21 (docs/research/datasets.md section 7):
- Force: ``obs/finger_force`` float32 (T, 2), one signed scalar per finger (observed range
  -57..49, simulator units, not newtons-calibrated). The adapter takes abs() and maps column 0 to
  f_a and column 1 to f_b. ``obs/gripper_net_force`` (T, 1, 2, 3) is clipped to +-10 and is
  NOT used.
- Gripper: ``obs/gripper_close_norm`` (T,) as is (0 open .. 1 closed).
- Layout: ``<subset>/libero_{spatial,object}/<task>/replayed_demos/<task>_<lang>_replayed_demo``
  ``.hdf5`` holding ``data/demo_<i>/{obs, soft_extras, states, actions, ...}``; subsets are
  spatial-rigid, object-rigid, object-soft, spatial-soft. Demo attrs: asset_name, demo_id,
  language, success,  soft_task_kind, task_id, task_suite. Episode id used here:
  ``"<subset>/<libero_suite>/<task>/<demo_i>"``.
- Videos: ``<subset>/video_datasets/<task>/videos/demo_<i>_{agentview,eye_in_hand}_rgb.mp4``;
  agentview is the "scene" camera and eye_in_hand is "wrist_right" (Franka, single arm "R").
- Rate: 20 Hz (Isaac Lab). The video and the HDF5 come from the same replay, so
  ``meta["clock_offset_s"] = 0.0`` (sample k is at k / 20 s). Video/HDF5 frame-count equality is
  not verified; duration comes from the scene video.
- Deformation (soft subsets only): ``fem_deformation_max``, ``fem_deformation_rms``,
  ``fem_kabsch_max_pct``, ``fem_kabsch_rms_pct`` (T,). Kept in meta as summary statistics
  (peak_fem_deformation_max, peak_fem_kabsch_max_pct, peak_fem_deformation_rms,
  peak_fem_kabsch_rms_pct); rigid subsets have none (keys absent).
- DSR tolerance: UNKNOWN (calibrated per object in the paper; the values are not in the HDF5).
  It must come from config: ``split_rules["dsr_tolerance_pct"]`` in
  configs/sources/softvtbench.yaml, either one float or a mapping
  ``{asset_name | "default": float}``, applied to
  ``fem_kabsch_max_pct``. When null/absent no event label is produced and
   ``meta["dsr_tolerance_pct"]`` is None. Safety Success vs DSR naming varies between sources.
- Native labels: outcome 5 if attr success else 1 (the published subsets keep only successes, so
  the outcome is nearly constant); plus, when a tolerance is configured and
  fem_kabsch_max_pct exceeds it, one "event" Label whose spans are the contiguous
  over-tolerance runs (quality 1, rationale "excessive force on the object").
- Session / split key: ``"<subset>/<task_id>"`` (task_id alone repeats across subsets). Every
  episode goes to split "benchmark".
- Caveats: simulation only; demo_id conventions differ per subset (the group key is used, not
  the attr); task 6 language is a placeholder; policy-hidden FEM states are present in the HDF5.
- License: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Episode,
    ForceStream,
    GripperStream,
    Label,
    PadArmForce,
    Span,
    VideoRef,
)
from tactile_flora.sources._eval_common import probe_video
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "Apache-2.0"
RATE_HZ = 20.0
OVERCOMPRESS_RATIONALE = "excessive force on the object"
_CAMERAS = {"scene": "agentview", "wrist_right": "eye_in_hand"}
_DEFORMATION = {
    "peak_fem_deformation_max": "fem_deformation_max",
    "peak_fem_deformation_rms": "fem_deformation_rms",
    "peak_fem_kabsch_max_pct": "fem_kabsch_max_pct",
    "peak_fem_kabsch_rms_pct": "fem_kabsch_rms_pct",
}


def _attr(value: Any) -> Any:
    """Decode an HDF5 attribute (bytes, numpy scalars) into a plain Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


@register_source("softvtbench")
class SoftvtbenchAdapter:
    """Adapter for SoftVTBench (eval only)."""

    name = "softvtbench"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._index: dict[str, tuple[Path, str]] | None = None

    def discover(self, root: Path) -> list[str]:
        """Return every demo id ``<subset>/<suite>/<task>/<demo_i>`` (no filtering)."""
        self._root = Path(root)
        index: dict[str, tuple[Path, str]] = {}
        for h5_path in sorted(self._root.glob("*/libero_*/*/replayed_demos/*.hdf5")):
            task_dir = h5_path.parent.parent
            prefix = f"{task_dir.parent.parent.name}/{task_dir.parent.name}/{task_dir.name}"
            with h5py.File(h5_path, "r") as f:
                data = f["data"]
                assert isinstance(data, h5py.Group)
                for key in data:
                    index[f"{prefix}/{key}"] = (h5_path, str(key))
        self._index = dict(sorted(index.items()))
        return list(self._index)

    def _locate(self, episode_id: str) -> tuple[Path, str]:
        if self._index is None:
            self.discover(self._root)
        assert self._index is not None
        return self._index[episode_id]

    def _tolerance(self, asset_name: str) -> float | None:
        raw = self.cfg.split_rules.get("dsr_tolerance_pct")
        if raw is None:
            return None
        if isinstance(raw, dict):
            val = raw.get(asset_name, raw.get("default"))  # type: ignore[reportUnknownMemberType]
            return None if val is None else float(val)  # type: ignore[reportUnknownArgumentType]
        return float(raw)

    def load(self, episode_id: str) -> Episode:
        """Load one demo on the episode clock (offset 0, 20 Hz)."""
        h5_path, key = self._locate(episode_id)
        subset, _suite, task, _demo = episode_id.split("/")
        with h5py.File(h5_path, "r") as f:
            demo = f[f"data/{key}"]
            assert isinstance(demo, h5py.Group)
            attrs = {k: _attr(v) for k, v in demo.attrs.items()}
            obs = demo["obs"]
            assert isinstance(obs, h5py.Group)
            ff = np.abs(np.asarray(obs["finger_force"], dtype=np.float64)).reshape(-1, 2)
            grip = np.asarray(obs["gripper_close_norm"], dtype=np.float64).reshape(-1)
            deform = {
                meta_key: float(np.max(np.asarray(obs[name], dtype=np.float64)))
                for meta_key, name in _DEFORMATION.items()
                if name in obs
            }
        t = np.arange(len(ff), dtype=np.float64) / RATE_HZ
        video_dir = self._root / subset / "video_datasets" / task / "videos"
        cameras: dict[str, VideoRef] = {}
        for role, cam in _CAMERAS.items():
            path = video_dir / f"{key}_{cam}_rgb.mp4"
            if path.is_file():
                cameras[role] = probe_video(path)
        if "scene" not in cameras:
            raise FileNotFoundError(f"missing scene video {video_dir / f'{key}_agentview_rgb.mp4'}")
        scene = cameras["scene"]
        duration = scene.n_frames / scene.fps if scene.n_frames else float(len(ff)) / RATE_HZ
        asset = str(attrs.get("asset_name", ""))
        meta: dict[str, Any] = {
            "clock_offset_s": 0.0,
            "subset": subset,
            "task_id": attrs.get("task_id"),
            "soft_task_kind": attrs.get("soft_task_kind"),
            "asset_name": asset,
            "demo_id": attrs.get("demo_id"),
            "task_suite": attrs.get("task_suite"),
            "force_units": "simulator units, abs of obs/finger_force",
            "dsr_tolerance_pct": self._tolerance(asset),
            **deform,
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=f"{subset}/{attrs.get('task_id', task)}",
            task=strip_outcome_words(str(attrs.get("language", task))),
            duration_s=duration,
            cameras=cameras,
            force=ForceStream(
                channel="pad",
                arms={"R": PadArmForce(f_a=ff[:, 0].tolist(), f_b=ff[:, 1].tolist())},
                rate_hz=RATE_HZ,
                t=t.tolist(),
            ),
            gripper=GripperStream(arms={"R": grip.tolist()}, rate_hz=RATE_HZ, t=t.tolist()),
            labels=self.native_labels(episode_id),
            license=LICENSE,
            meta=meta,
        )

    def splits(self) -> dict[str, list[str]]:
        """Eval only: every episode is in split "benchmark"."""
        if self._index is None:
            self.discover(self._root)
        assert self._index is not None
        return {"benchmark": list(self._index)}

    def native_labels(self, episode_id: str) -> list[Label]:
        """Outcome (5 if success else 1) plus over-compression events when a tolerance is set."""
        h5_path, key = self._locate(episode_id)
        with h5py.File(h5_path, "r") as f:
            demo = f[f"data/{key}"]
            assert isinstance(demo, h5py.Group)
            success = bool(_attr(demo.attrs.get("success", False)))
            asset = str(_attr(demo.attrs.get("asset_name", "")))
            obs = demo["obs"]
            assert isinstance(obs, h5py.Group)
            kabsch = (
                np.asarray(obs["fem_kabsch_max_pct"], dtype=np.float64).reshape(-1)
                if "fem_kabsch_max_pct" in obs
                else None
            )
        labels = [Label(kind="outcome", origin="source", outcome=5 if success else 1)]
        tol = self._tolerance(asset)
        if tol is not None and kabsch is not None:
            over = np.concatenate([[False], kabsch > tol, [False]])
            edges = np.flatnonzero(np.diff(over.astype(np.int8)))
            spans = [
                Span(
                    t0=float(s) / RATE_HZ,
                    t1=float(e) / RATE_HZ,
                    quality=1,
                    rationale=OVERCOMPRESS_RATIONALE,
                )
                for s, e in zip(edges[::2], edges[1::2], strict=True)
            ]
            if spans:
                labels.append(Label(kind="event", origin="source", spans=spans))
        return labels
