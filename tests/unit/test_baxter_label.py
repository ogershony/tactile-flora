"""Labeler with a scripted headless player; labels.json merge and the no-force guarantee."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tactile_flora.baxter import label
from tactile_flora.sources.baxter import BaxterAdapter
from tests.fixtures._video import write_mp4


class ScriptedPlayer:
    """Headless Player: pops answers from a script; records what it was shown."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.shown: list[tuple[Path, int]] = []

    def show_second(self, video: Path, second: int) -> str:
        self.shown.append((video, second))
        return self.keys.pop(0)

    def ask(self, prompt: str, valid: str) -> str:
        ans = self.keys.pop(0)
        assert not valid or ans in valid
        return ans


def _root(tmp_path: Path) -> Path:
    # 3 s (12 frames at 4 fps), two episodes
    for i in (0, 1):
        write_mp4(
            tmp_path / "videos" / "observation.images.scene" / f"episode_{i:06d}.mp4", n=12, fps=4
        )
    return tmp_path


def test_single_labeler_writes_per_labeler_file(tmp_path: Path) -> None:
    root = _root(tmp_path)
    player = ScriptedPlayer(["3", "2", "1", "4", "can", "3", "3", "3", "5", "block"])
    path = label.run_labeler(root, "alice", player)
    data = json.loads(path.read_text())
    assert data["0"] == {"quality_1hz": [3, 2, 1], "outcome": 4, "object": "can"}
    assert data["1"]["outcome"] == 5 and not player.keys
    assert [s for _, s in player.shown[:3]] == [0, 1, 2]
    # rerun skips already-labeled episodes
    label.run_labeler(root, "alice", ScriptedPlayer([]))


def test_merge_two_labelers_with_agreement_and_resolution(tmp_path: Path) -> None:
    root = _root(tmp_path)
    label.run_labeler(
        root, "a", ScriptedPlayer(["3", "2", "1", "4", "can", "3", "3", "3", "5", "block"])
    )
    label.run_labeler(
        root, "b", ScriptedPlayer(["3", "3", "1", "2", "can", "3", "3", "3", "5", "block"])
    )
    resolver = label.player_resolver(ScriptedPlayer(["2", "3"]))  # second 1 -> 2, outcome -> 3
    path = label.merge_labels(root, "a", "b", resolver)
    out = json.loads(path.read_text())
    assert out["0"]["quality_1hz"] == [3, 2, 1] and out["0"]["outcome"] == 3
    assert out["0"]["agreement"] == pytest.approx(2 / 3) and out["1"]["agreement"] == 1.0
    assert out["1"]["object"] == "block"


def test_labels_load_in_the_baxter_adapter_contract(tmp_path: Path) -> None:
    root = _root(tmp_path)
    label.run_labeler(
        root, "a", ScriptedPlayer(["3", "2", "2", "4", "can", "3", "3", "3", "5", "b"])
    )
    label.run_labeler(
        root, "b", ScriptedPlayer(["3", "2", "2", "4", "can", "3", "3", "3", "5", "b"])
    )
    label.merge_labels(root, "a", "b", label.player_resolver(ScriptedPlayer([])))
    from tactile_flora.config import SourceConfig

    cfg = SourceConfig(
        name="baxter",
        root=str(root),
        license_allowed=["internal"],
        max_windows=0,
        max_episodes=0,
        eval_only=True,
        expected_label_rate=0.3,
        seed=0,
    )
    labels = BaxterAdapter(cfg).native_labels("episode_000000")
    assert [lab.kind for lab in labels] == ["outcome", "dense"]
    assert [(s.t0, s.t1, s.quality) for s in labels[1].spans] == [(1.0, 3.0, 2)]


def test_labeler_never_touches_force() -> None:
    label.assert_no_force_access()
    tree = ast.parse(Path(label.__file__).read_text())
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        for a in (n.names if isinstance(n, ast.Import | ast.ImportFrom) else [])
    } | {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not [m for m in imported if "force" in m or "resample" in m or "pandas" in m]
    assert not [n for n in dir(label) if "force" in n.lower().replace("assert_no_force_access", "")]
