"""Source adapter for REASSEMBLE (TU Wien, NIST Task Board 1, Franka; HDF5 per recording).

Owns: the REASSEMBLE HDF5 layout and its conversion to the episode clock.
Must never: filter episodes in discover(), write to disk (including extracting video), or invent
anomaly spans that the dataset does not have.

Resolved open questions (verified 2026-10-21, docs/research/datasets.md):
- Source: https://researchdata.tuwien.ac.at/records/0ewrv-8cb44 (v1.0.0), license CC-BY-4.0.
  149 recordings named ``YYYY-MM-DD-HH-MM-SS.h5`` under ``<root>/``, 4,551 demonstrations.
  Episode id = the file stem, i.e. one episode is a whole RECORDING containing many actions;
  per-action structure is in ``meta["segments"]``. ``Episode.task`` is the first segment's text
  (e.g. "Insert USB."); windowing code must use the segment texts for finer prompts.
- HDF5: ``robot_state/{measured_force, measured_torque, compensated_base_force,
  compensated_base_torque, gripper_positions, ...}``, ``timestamps/<sensor>`` (N x 1,
  seconds on the sensor clock, own rate; F/T ~1.7 kHz), top-level mp4 bytes ``hama1``, ``hama2``,
  ``hand``, and ``segments_info/<i>/{start,end,success,text}`` with ``Low_level/<j>/...``.
- Force: ``measured_force`` + ``measured_torque`` -> WristArmForce arm "R" (channel wrist, raw
  units; compensated_base_* used only if measured_* is absent). Timestamps come from
  ``timestamps/measured_force`` (falling back to ``timestamps/<same name>`` / ``measured_torque``);
  the exact timestamps key names are UNVERIFIED beyond "one per sensor".
- Gripper: ``robot_state/gripper_positions`` is a Franka finger width; closure = 1 - w/max(w)
  per episode (assumes larger width = more open; UNVERIFIED units).
- Videos are mp4 BYTES inside the HDF5. load() writes nothing: it expects the caller to have
  extracted them to ``<root>/extracted/<file stem>/{hama1,hama2,hand}.mp4`` and raises
  FileNotFoundError with the extraction command otherwise. hama1 is the scene camera, hama2 role
  ``scene_2``, hand role ``wrist_right`` (absent for three files with the hand cam missing).
  Frame times are ``timestamps/hama1`` (fps estimated from them).
- Clock: zero is ``timestamps/hama1[0]``; the offset is stored in ``meta["clock_offset_s"]`` and
  subtracted from every stream and segment time.
- Anomaly format: there is NO span/frame anomaly annotation. Only per-segment ``success`` and
  the README known-issues table plus two measured additions (``KNOWN_ISSUES`` below, exposed
  in ``meta["known_issues"]``; the LeRobot-port adapter reads the same table by recording stem).
  native_labels() returns one ``event`` Label whose spans are the segments (quality 3 if
  success else 1, rationale = segment text); labels/mapping.py may ignore or re-map them.
- Session = recording date (file name ``YYYY-MM-DD``); splits by date (hash, see split_rules).

meta keys: segments (list of {t0,t1,success,text}), low_level_segments (list of
{t0,t1,success,text,parent}), action_success (list[bool], one per top-level segment),
known_issues (list[str]), clock_offset_s, h5_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Episode,
    ForceStream,
    GripperStream,
    Label,
    Span,
    VideoRef,
    WristArmForce,
)
from tactile_flora.sources._common import assign_splits, median_rate
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "CC-BY-4.0"
KNOWN_ISSUES: dict[str, list[str]] = {
    "2025-01-10-15-28-50": ["hand_cam_missing"],
    "2025-01-10-16-17-40": ["hand_cam_missing"],
    "2025-01-10-17-10-38": ["hand_cam_missing"],
    "2025-01-10-17-54-09": ["no_empty_action_at_beginning"],
    "2025-01-11-14-22-09": ["no_empty_action_at_beginning"],
    "2025-01-11-14-45-48": ["ft_invalid_last_action"],
    "2025-01-11-15-27-19": ["ft_invalid_last_action"],
    "2025-01-11-15-35-08": ["ft_invalid_last_action"],
    "2025-01-13-11-16-17": ["gripper_broke_last_action"],
    "2025-01-13-11-18-57": ["pose_unavailable_last_action"],
    # Not in the README table. Same signature as the three ft_invalid_last_action files above:
    # the gravity-compensated force offset is confined to the last idle segment (6.4 N and
    # 3.9 N there, below 0.6 N in the first idle segment). Measured on the LeRobot port with
    # scripts/measure_reassemble_force_keys.py (2026-10-22, all 149 recordings).
    "2025-01-13-18-06-08": ["ft_invalid_last_action"],
    "2025-01-13-18-16-38": ["ft_invalid_last_action"],
}
EXTRACTED_DIR = "extracted"
VIDEO_ROLES = {"hama1": "scene", "hama2": "scene_2", "hand": "wrist_right"}
FORCE_KEYS = ("measured_force", "measured_torque")
FALLBACK_FORCE_KEYS = ("compensated_base_force", "compensated_base_torque")
EXTRACT_HINT = (
    'python -c "import h5py,sys; f=h5py.File(sys.argv[1]); '
    "[open(sys.argv[2]+'/'+k+'.mp4','wb').write(f[k][()].tobytes()) "
    "for k in ('hama1','hama2','hand') "
    'if k in f]" {h5} {out}'
)


def _s(v: Any) -> str:
    return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)


def _scalar(ds: Any) -> Any:
    v = ds[()]
    return v.item() if isinstance(v, np.generic) else v


def _segments(group: Any, offset: float, parent: int | None = None) -> list[dict[str, Any]]:
    """Read ``<i>/{start,end,success,text}`` children in numeric order."""
    out: list[dict[str, Any]] = []
    keys: list[str] = [str(k) for k in group]
    for key in sorted((k for k in keys if k.isdigit()), key=int):
        g: Any = group[key]
        seg: dict[str, Any] = {
            "t0": float(_scalar(g["start"])) - offset,
            "t1": float(_scalar(g["end"])) - offset,
            "success": bool(_scalar(g["success"])),
            "text": _s(_scalar(g["text"])),
        }
        if parent is not None:
            seg["parent"] = parent
        out.append(seg)
    return out


@register_source("reassemble")
class ReassembleAdapter:
    """Adapter for REASSEMBLE recordings (one episode per HDF5 file)."""

    name = "reassemble"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._files: dict[str, Path] = {}

    def discover(self, root: Path) -> list[str]:
        """Return every ``*.h5`` file stem under ``root``."""
        self._root = root
        self._files = {p.stem: p for p in sorted(root.glob("*.h5"))}
        return list(self._files)

    def _ensure(self) -> None:
        if not self._files:
            self.discover(self._root)

    @staticmethod
    def _date(stem: str) -> str:
        return stem[:10]

    def load(self, episode_id: str) -> Episode:
        """Load one recording on the episode clock (zero at the first hama1 frame)."""
        self._ensure()
        path = self._files[episode_id]
        extracted = path.parent / EXTRACTED_DIR / episode_id
        if not (extracted / "hama1.mp4").exists():
            raise FileNotFoundError(
                f"{extracted}/hama1.mp4 missing; extract the embedded mp4 bytes first with: "
                + EXTRACT_HINT.format(h5=path, out=extracted)
            )
        with h5py.File(path, "r") as f:
            ts_group: Any = f["timestamps"]
            rs: Any = f["robot_state"]
            cam_t = np.asarray(ts_group["hama1"][:], dtype=np.float64).reshape(-1)
            offset = float(cam_t[0])
            duration = float(cam_t[-1] - offset)
            cam_fps = median_rate(cam_t - offset, default=1.0)
            camera_times = {
                key: np.asarray(ts_group[key][:], dtype=np.float64).reshape(-1) - offset
                for key in VIDEO_ROLES
                if key in ts_group
            }

            fkey, tkey = FORCE_KEYS if FORCE_KEYS[0] in rs else FALLBACK_FORCE_KEYS
            force_raw = np.asarray(rs[fkey][:], dtype=np.float64).reshape(-1, 3)
            torque_raw = np.asarray(rs[tkey][:], dtype=np.float64).reshape(-1, 3)
            ts_key = next(k for k in (fkey, tkey, "measured_force") if k in ts_group)
            ft_t = np.asarray(ts_group[ts_key][:], dtype=np.float64).reshape(-1)
            n = min(len(force_raw), len(torque_raw), len(ft_t))
            w = np.concatenate([force_raw[:n], torque_raw[:n]], axis=1)
            ft_t = ft_t[:n] - offset

            gripper: GripperStream | None = None
            if "gripper_positions" in rs:
                gp = np.asarray(rs["gripper_positions"][:], dtype=np.float64)
                gp = gp.reshape(len(gp), -1)[:, 0]
                gt_key = "gripper_positions" if "gripper_positions" in ts_group else None
                if gt_key is not None:
                    gt = np.asarray(ts_group[gt_key][:], dtype=np.float64).reshape(-1)
                    m = min(len(gp), len(gt))
                    gp, gt = gp[:m], gt[:m] - offset
                    hi = float(gp.max()) if m else 0.0
                    closure = np.clip(1.0 - gp / hi, 0.0, 1.0) if hi > 0 else np.zeros_like(gp)
                    gripper = GripperStream(
                        arms={"R": closure.tolist()}, rate_hz=median_rate(gt), t=gt.tolist()
                    )

            segments: list[dict[str, Any]] = []
            low: list[dict[str, Any]] = []
            if "segments_info" in f:
                seg_group: Any = f["segments_info"]
                segments = _segments(seg_group, offset)
                for key in sorted((k for k in seg_group if k.isdigit()), key=int):
                    if "Low_level" in seg_group[key]:
                        low += _segments(seg_group[key]["Low_level"], offset, parent=int(key))

        cameras: dict[str, VideoRef] = {}
        for key, role in VIDEO_ROLES.items():
            mp4 = extracted / f"{key}.mp4"
            if mp4.exists():
                # Each camera has its own timestamps and frame count. Timestamp jitter
                # biases median(dt); the encoded playback rate is authoritative here.
                rate = cam_fps
                try:
                    with av.open(str(mp4)) as container:
                        encoded_rate = container.streams.video[0].average_rate
                        if encoded_rate:
                            rate = float(encoded_rate)
                except Exception:
                    pass  # the integrity check reports undecodable video
                times = camera_times.get(key, cam_t - offset)
                cameras[role] = VideoRef(path=str(mp4), fps=rate, n_frames=len(times))

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
            rate_hz=median_rate(ft_t, default=1.0),
            t=ft_t.tolist(),
        )
        task = segments[0]["text"] if segments else episode_id
        meta: dict[str, Any] = {
            "segments": segments,
            "low_level_segments": low,
            "action_success": [s["success"] for s in segments],
            "known_issues": list(KNOWN_ISSUES.get(episode_id, [])),
            "clock_offset_s": offset,
            "h5_path": str(path),
            "camera_timing": {
                VIDEO_ROLES[key]: {
                    "samples": len(t),
                    "first_s": float(t[0]),
                    "last_s": float(t[-1]),
                    "max_interval_s": float(np.max(np.diff(t))) if len(t) > 1 else None,
                }
                for key, t in camera_times.items()
                if len(t)
            },
        }
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=self._date(episode_id),
            task=strip_outcome_words(task),
            duration_s=duration,
            cameras=cameras,
            force=force,
            gripper=gripper,
            labels=[],
            license=LICENSE,
            meta=meta,
        )

    def splits(self) -> dict[str, list[str]]:
        """Return train/val/test by recording date."""
        self._ensure()
        return assign_splits({e: self._date(e) for e in self._files}, self.cfg)

    def native_labels(self, episode_id: str) -> list[Label]:
        """Return one event Label with a span per top-level segment (success -> 3, else 1)."""
        segs: list[dict[str, Any]] = self.load(episode_id).meta["segments"]
        spans = [
            Span(
                t0=s["t0"],
                t1=s["t1"],
                quality=3 if s["success"] else 1,
                rationale=s["text"],
            )
            for s in segs
            if s["t1"] >= s["t0"]
        ]
        return [Label(kind="event", origin="source", spans=spans)] if spans else []
