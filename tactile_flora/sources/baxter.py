"""Source adapter for the Baxter capture (internal, eval only).

Owns: the on-disk contract between baxter/to_lerobot.py (writer) and this adapter (reader), and
how one Baxter episode becomes an Episode. Maps the four FSR pads to the pad channel, both arms
(L: L_a, L_b; R: R_a, R_b).
Must never: filter episodes in discover(), write to disk, be used for training (eval only), or
leak layout details to other modules.

On-disk contract (LeRobot v3 flavored; ``<idx>`` is the zero-padded 6-digit episode_index,
so the episode id is ``"episode_<idx>"``, e.g. ``episode_000003``):

- ``<root>/data/chunk-000/episode_<idx>.parquet``: one row per video frame (30 fps). Columns:
  ``timestamp`` (float s, 0 at the first scene video frame, i.e. already the episode clock),
  ``frame_index`` (int), ``episode_index`` (int), ``observation.state`` (list[float], Baxter joint
  and gripper state), ``observation.gripper_left`` and ``observation.gripper_right`` (float,
  0 open .. 1 closed), ``observation.force`` (list of 4 floats in newtons ordered
  L_a, L_b, R_a, R_b: the 200 Hz force resampled to the frame clock; a convenience copy, used
  by the adapter only if the native force file below is missing).
- ``<root>/force/episode_<idx>.parquet``: the NATIVE 200 Hz force, columns ``t`` (float s on the
  recorder (robot) clock, NOT shifted), ``L_a``, ``L_b``, ``R_a``, ``R_b`` (newtons).
- ``<root>/videos/observation.images.{scene,wrist_left,wrist_right}/episode_<idx>.mp4`` at
  30 fps; scene is required, the wrist videos are optional.
- ``<root>/meta/episodes.jsonl``: one JSON object per line with ``episode_index`` (int),
  ``object`` (str), ``firmness`` (str), ``policy`` ({grasp_height, lateral_offset,
  gripper_force}), ``occlusion`` (bool), ``operator_notes`` (str), ``recording_day``
  (YYYY-MM-DD, the session / split key), ``calibration_version`` (str), and optional ``task``
  (instruction text, default ``"pick up the <object>"``) and ``video_t0_s`` (recorder-clock time
  in seconds of the first scene video frame, default 0.0).
- ``<root>/labels.json`` (written later by ``tf baxter label``, may be absent or partial):
  ``{"<episode_index>": {"quality_1hz": [digits], "outcome": int 1..5, "object": str,
  "agreement": float}}``. Digit k covers [k, k+1) s; 1 failure, 2 suboptimal, 3 normal.

Clock: the force ``t`` minus ``video_t0_s`` puts force on the episode clock;
``meta["clock_offset_s"]`` is that ``video_t0_s``. Frame-table streams (gripper) are already on the
episode clock via ``timestamp``. Force samples before the first video frame are kept as negative
times only if the clock shift makes them so; downstream resampling decides what to use.

Native labels: outcome from labels.json; dense spans from the 1 Hz digits: runs of equal digits
are merged and only non-normal runs (1 or 2) become spans (rationale ""). Episodes without a
labels.json entry get no labels. Session: recording_day. License: "internal".

Unverified (hardware not yet captured): the real FSR units/range (sensor identity is an open
question in the spec); whether the writer can provide ``video_t0_s``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Arm,
    ArmForce,
    Episode,
    ForceStream,
    GripperStream,
    Label,
    PadArmForce,
    Span,
    VideoRef,
)
from tactile_flora.sources._eval_common import probe_video, rate_from_times
from tactile_flora.sources.base import register_source, strip_outcome_words

LICENSE = "internal"
FRAME_RATE_HZ = 30.0
FORCE_RATE_HZ = 200.0
NORMAL_DIGIT = 3
_ROLES = ("scene", "wrist_left", "wrist_right")


def _index_of(episode_id: str) -> int:
    return int(episode_id.removeprefix("episode_"))


@register_source("baxter")
class BaxterAdapter:
    """Adapter for the Baxter capture (eval only)."""

    name = "baxter"

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._root = Path(cfg.root)
        self._ids: list[str] | None = None

    def discover(self, root: Path) -> list[str]:
        """Return every episode id found under ``<root>/data/chunk-*`` (no filtering)."""
        self._root = Path(root)
        files = sorted(self._root.glob("data/chunk-*/episode_*.parquet"))
        self._ids = sorted({p.stem for p in files})
        return list(self._ids)

    def _ensure(self) -> list[str]:
        if self._ids is None:
            self.discover(self._root)
        assert self._ids is not None
        return self._ids

    def _episode_meta(self, idx: int) -> dict[str, Any]:
        path = self._root / "meta" / "episodes.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if int(row["episode_index"]) == idx:
                        return row
        return {}

    def load(self, episode_id: str) -> Episode:
        """Load one episode; force is shifted by ``video_t0_s`` onto the episode clock."""
        self._ensure()
        idx = _index_of(episode_id)
        stem = f"episode_{idx:06d}"
        meta = self._episode_meta(idx)
        frames = pd.read_parquet(next(self._root.glob(f"data/chunk-*/{stem}.parquet")))
        offset = float(meta.get("video_t0_s", 0.0))

        force_path = self._root / "force" / f"{stem}.parquet"
        if force_path.is_file():
            fdf = pd.read_parquet(force_path)
            t = fdf["t"].to_numpy(dtype=np.float64) - offset
            cols = {c: fdf[c].to_numpy(dtype=np.float64) for c in ("L_a", "L_b", "R_a", "R_b")}
            rate = rate_from_times(t, FORCE_RATE_HZ)
            force_source = "force/native_200hz"
        else:
            t = frames["timestamp"].to_numpy(dtype=np.float64)
            stacked = np.stack(
                [np.asarray(v, dtype=np.float64) for v in frames["observation.force"]]
            )
            cols = {c: stacked[:, i] for i, c in enumerate(("L_a", "L_b", "R_a", "R_b"))}
            rate = FRAME_RATE_HZ
            offset = 0.0
            force_source = "observation.force/frame_aligned"

        arms: dict[Arm, ArmForce] = {
            "L": PadArmForce(f_a=cols["L_a"].tolist(), f_b=cols["L_b"].tolist()),
            "R": PadArmForce(f_a=cols["R_a"].tolist(), f_b=cols["R_b"].tolist()),
        }
        ft = frames["timestamp"].to_numpy(dtype=np.float64)
        gripper = GripperStream(
            arms={
                "L": frames["observation.gripper_left"].to_numpy(dtype=np.float64).tolist(),
                "R": frames["observation.gripper_right"].to_numpy(dtype=np.float64).tolist(),
            },
            rate_hz=rate_from_times(ft, FRAME_RATE_HZ),
            t=ft.tolist(),
        )

        cameras: dict[str, VideoRef] = {}
        for role in _ROLES:
            path = self._root / "videos" / f"observation.images.{role}" / f"{stem}.mp4"
            if path.is_file():
                cameras[role] = probe_video(path)
        if "scene" not in cameras:
            raise FileNotFoundError(f"missing scene video for {episode_id}")
        scene = cameras["scene"]
        duration = scene.n_frames / scene.fps if scene.n_frames else float(ft[-1])

        obj = str(meta.get("object", ""))
        task = str(meta.get("task") or f"pick up the {obj}".strip())
        return Episode(
            source=self.name,
            episode_id=episode_id,
            session=str(meta.get("recording_day", "unknown")),
            task=strip_outcome_words(task),
            duration_s=duration,
            cameras=cameras,
            force=ForceStream(channel="pad", arms=arms, rate_hz=rate, t=t.tolist()),
            gripper=gripper,
            labels=self.native_labels(episode_id),
            license=LICENSE,
            meta={
                "clock_offset_s": offset,
                "force_source": force_source,
                "episode_index": idx,
                "object": obj,
                "firmness": meta.get("firmness"),
                "policy": meta.get("policy"),
                "occlusion": meta.get("occlusion"),
                "operator_notes": meta.get("operator_notes"),
                "calibration_version": meta.get("calibration_version"),
            },
        )

    def splits(self) -> dict[str, list[str]]:
        """Eval only: every episode is in split "benchmark"."""
        return {"benchmark": list(self._ensure())}

    def native_labels(self, episode_id: str) -> list[Label]:
        """Human labels from labels.json: outcome plus non-normal 1 Hz runs as dense spans."""
        path = self._root / "labels.json"
        if not path.is_file():
            return []
        entry = json.loads(path.read_text()).get(str(_index_of(episode_id)))
        if not entry:
            return []
        labels: list[Label] = []
        if entry.get("outcome") is not None:
            labels.append(Label(kind="outcome", origin="source", outcome=int(entry["outcome"])))
        digits = [int(d) for d in entry.get("quality_1hz", [])]
        spans: list[Span] = []
        start = 0
        for i in range(1, len(digits) + 1):
            if i == len(digits) or digits[i] != digits[start]:
                if digits[start] != NORMAL_DIGIT:
                    spans.append(
                        Span(t0=float(start), t1=float(i), quality=digits[start])  # type: ignore[arg-type]
                    )
                start = i
        if digits:
            labels.append(Label(kind="dense", origin="source", spans=spans))
        return labels
