"""The keystroke labeling tool for Baxter episodes (labels.json).

Owns: playing an episode's video through a ``Player``, recording 1 Hz quality digits (keys 1/2/3),
an outcome 1-5 and the object, per-labeler files, and the two-labeler merge into labels.json with
the agreement rate.
Must never: read, import or show force data, or derive labels from force thresholds. The labeler
reads only ``videos/observation.images.scene`` and writes only ``labels/`` and ``labels.json``;
it does not open the data/ or force/ parquet files, and ``assert_no_force_access`` enforces that.

Per-labeler files: ``<root>/labels/<labeler>.json``, same schema as labels.json minus agreement.
Merge: per episode both labelers must have the same number of seconds; ``agreement`` is the
fraction of seconds with equal digits; each disagreeing second (and an outcome / object
disagreement) goes to ``resolve`` (the labelers decide together), others are kept as agreed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import av

QUALITY_KEYS = "123"
OUTCOME_KEYS = "12345"
_FORBIDDEN = ("force", "resample", "parquet", "pandas")


def assert_no_force_access() -> None:
    """Fail if this module's namespace references force data (guard against future edits)."""
    bad = [n for n in globals() if any(f in n.lower() for f in _FORBIDDEN) and n != "_FORBIDDEN"]
    bad = [n for n in bad if n != "assert_no_force_access"]
    assert not bad, f"labeler must never touch force data: {bad}"


assert_no_force_access()


class Player(Protocol):
    """What the labeler needs from a UI; the real one is interactive, tests script it."""

    def show_second(self, video: Path, second: int) -> str:
        """Play second ``second`` of ``video`` and return the quality key pressed (1/2/3)."""
        ...

    def ask(self, prompt: str, valid: str) -> str:
        """Ask for one key out of ``valid`` (a string of allowed characters) or free text if ""."""
        ...


class TerminalPlayer:
    """Plain-terminal player: opens the video in the OS viewer once, then asks per second."""

    def __init__(self) -> None:
        self._opened: Path | None = None

    def show_second(self, video: Path, second: int) -> str:
        """Open the video (first call) and prompt for the quality of this second."""
        if self._opened != video:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(video)])  # noqa: S603
            self._opened = video
        return self.ask(
            f"second {second} quality (1 failure, 2 suboptimal, 3 normal)", QUALITY_KEYS
        )

    def ask(self, prompt: str, valid: str) -> str:
        """Prompt until the answer is one allowed key (free text when ``valid`` is empty)."""
        while True:
            ans = input(f"{prompt}: ").strip()
            if (not valid and ans) or (len(ans) == 1 and ans in valid):
                return ans


def scene_video(root: Path, idx: int) -> Path:
    """Path of the scene video of episode ``idx`` (the only file the labeler plays)."""
    return root / "videos" / "observation.images.scene" / f"episode_{idx:06d}.mp4"


def episode_indices(root: Path) -> list[int]:
    """Episode indices that have a scene video."""
    vids = (root / "videos" / "observation.images.scene").glob("episode_*.mp4")
    return sorted(int(p.stem.removeprefix("episode_")) for p in vids)


def video_seconds(video: Path) -> int:
    """Whole seconds (rounded up) of a video, from PyAV stream metadata / frame count."""
    container = av.open(str(video))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        n = int(stream.frames) or sum(1 for _ in container.decode(video=0))
    finally:
        container.close()
    return max(1, int(-(-n // fps)))


def label_episode(player: Player, video: Path) -> dict[str, Any]:
    """Collect 1 Hz digits, the outcome and the object for one episode."""
    digits = [int(player.show_second(video, s)) for s in range(video_seconds(video))]
    outcome = int(player.ask("episode outcome (1 worst .. 5 best)", OUTCOME_KEYS))
    obj = player.ask("object name", "")
    return {"quality_1hz": digits, "outcome": outcome, "object": obj}


def _load(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text()) if path.is_file() else {}


def run_labeler(
    root: Path, labeler: str, player: Player, only: list[int] | None = None, redo: bool = False
) -> Path:
    """Label every (or the ``only``) episode not yet labeled by ``labeler``; saves after each."""
    out = root / "labels" / f"{labeler}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = _load(out)
    for idx in only if only is not None else episode_indices(root):
        if str(idx) in labels and not redo:
            continue
        labels[str(idx)] = label_episode(player, scene_video(root, idx))
        out.write_text(json.dumps(labels, indent=2))
    out.write_text(json.dumps(labels, indent=2))
    return out


Resolve = Callable[[int, str, Any, Any], Any]


def player_resolver(player: Player) -> Resolve:
    """Build a resolver that asks the labelers (via ``player``) to settle a disagreement."""

    def _resolve(idx: int, what: str, a: Any, b: Any) -> Any:
        valid = (
            QUALITY_KEYS if what.startswith("second") else OUTCOME_KEYS if what == "outcome" else ""
        )
        ans = player.ask(f"episode {idx} {what}: A={a} B={b}; agreed value", valid)
        return int(ans) if valid else ans

    return _resolve


def merge_labels(root: Path, labeler_a: str, labeler_b: str, resolve: Resolve) -> Path:
    """Merge two labelers' files into ``<root>/labels.json`` with the agreement rate."""
    a = _load(root / "labels" / f"{labeler_a}.json")
    b = _load(root / "labels" / f"{labeler_b}.json")
    merged = _load(root / "labels.json")
    for key in sorted(set(a) & set(b), key=int):
        ea, eb = a[key], b[key]
        da, db = list(ea["quality_1hz"]), list(eb["quality_1hz"])
        if len(da) != len(db):
            raise RuntimeError(f"episode {key}: labelers saw {len(da)} vs {len(db)} seconds")
        digits: list[int] = []
        for s, (x, y) in enumerate(zip(da, db, strict=True)):
            digits.append(int(x if x == y else resolve(int(key), f"second {s}", x, y)))
        agreement = sum(x == y for x, y in zip(da, db, strict=True)) / len(da)
        outcome = (
            ea["outcome"]
            if ea["outcome"] == eb["outcome"]
            else resolve(int(key), "outcome", ea["outcome"], eb["outcome"])
        )
        obj = (
            ea["object"]
            if ea["object"] == eb["object"]
            else resolve(int(key), "object", ea["object"], eb["object"])
        )
        merged[key] = {
            "quality_1hz": digits,
            "outcome": int(outcome),
            "object": str(obj),
            "agreement": agreement,
        }
    path = root / "labels.json"
    path.write_text(json.dumps(merged, indent=2))
    return path
