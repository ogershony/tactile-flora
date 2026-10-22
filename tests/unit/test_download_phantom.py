"""Safety and selection checks for the bounded PHANTOM downloader."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import download_phantom as phantom


def _episode(path: str, *, split: str, failure: bool, episode: str) -> dict[str, Any]:
    return {
        "path": path,
        "split": split,
        "failure_demo": failure,
        "tactile_contact": True,
        "episode": episode,
    }


def test_select_episodes_picks_first_success_failure_and_validation_in_catalog_order() -> None:
    rows = [
        _episode("tasks/a/success-first", split="train", failure=False, episode="s1"),
        _episode("tasks/a/failure", split="train", failure=True, episode="f1"),
        _episode("tasks/a/success-second", split="train", failure=False, episode="s2"),
        _episode("tasks/a/validation", split="val", failure=False, episode="v1"),
    ]

    selected = phantom.select_episodes(rows)

    assert [row["episode"] for row in selected] == ["s1", "f1", "v1"]


def test_expanded_selection_is_seeded_task_balanced_and_order_independent() -> None:
    rows = [
        {
            **_episode(
                f"tasks/{task}/{split}-{failure}-{i}",
                split=split,
                failure=failure,
                episode=f"{task}-{split}-{failure}-{i}",
            ),
            "task": task,
        }
        for split, failure in (("train", False), ("train", True), ("val", False))
        for task in ("carton", "egg")
        for i in range(5)
    ]
    selected = phantom.select_stratified_episodes(rows, 4, 0)
    assert selected == phantom.select_stratified_episodes(list(reversed(rows)), 4, 0)
    assert selected != phantom.select_stratified_episodes(rows, 4, 1)
    assert len({r["episode"] for r in selected}) == 12
    for offset in (0, 4, 8):
        assert (
            sorted(r["task"] for r in selected[offset : offset + 4]) == ["carton"] * 2 + ["egg"] * 2
        )


def test_expanded_selection_refuses_an_undersized_stratum() -> None:
    with pytest.raises(RuntimeError, match="need 2"):
        phantom.select_stratified_episodes([], 2, 0)


@pytest.mark.parametrize(
    "path", ["/tmp/out", "../escape", "tasks/../escape", "tasks\\escape", "tasks/./episode", ""]
)
def test_remote_path_rejects_noncanonical_or_escaping_paths(path: str) -> None:
    with pytest.raises(ValueError, match="unsafe remote path"):
        phantom.safe_remote_path(path)


def test_selection_rejects_unsafe_episode_path_before_tree_request() -> None:
    rows = [
        _episode("tasks/../escape", split="train", failure=False, episode="s1"),
        _episode("tasks/failure", split="train", failure=True, episode="f1"),
        _episode("tasks/validation", split="val", failure=False, episode="v1"),
    ]

    with pytest.raises(ValueError, match="unsafe remote path"):
        phantom.select_episodes(rows)


def test_selected_file_filter_keeps_only_episode_metadata_and_requested_streams() -> None:
    episode = "tasks/carton/ep1"

    assert phantom._is_selected_file(f"{episode}/meta.json", phantom.safe_remote_path(episode))
    assert phantom._is_selected_file(
        f"{episode}/camera_scene_color.zarr/data/0", phantom.safe_remote_path(episode)
    )
    assert not phantom._is_selected_file(
        f"{episode}/robot_state.zarr/data/0", phantom.safe_remote_path(episode)
    )
    assert not phantom._is_selected_file(
        "tasks/carton/ep10/meta.json", phantom.safe_remote_path(episode)
    )


def test_list_files_uses_pinned_revision_and_rejects_unsafe_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episode("tasks/carton/ep1", split="train", failure=False, episode="ep1")
    requested: list[str] = []

    class Response(io.BytesIO):
        headers = {"Link": '<https://attacker.invalid/steal>; rel="next"'}

    def fake_urlopen(url: str, *, timeout: int) -> Response:
        requested.append(url)
        return Response(
            json.dumps(
                [
                    {"type": "file", "path": "tasks/carton/ep1/meta.json", "size": 3},
                    {"type": "file", "path": "tasks/carton/ep1/robot_state/data", "size": 100},
                ]
            ).encode()
        )

    monkeypatch.setattr(phantom, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="leaves the expected"):
        phantom.list_selected_files([episode], "a" * 40)
    assert len(requested) == 1
    assert "/tree/" + "a" * 40 + "/" in requested[0]


def test_download_rejects_oversized_response_without_replacing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data" / "phantom"
    target = root / "tasks" / "ep1" / "meta.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(phantom, "urlopen", lambda url, timeout: Response(b"oversized"))

    with pytest.raises(RuntimeError, match="oversized download"):
        phantom.download_file(root, "a" * 40, {"path": "tasks/ep1/meta.json", "size": 2})

    assert target.read_bytes() == b"old"
    assert list(target.parent.glob("*.partial")) == []


def test_download_reuses_existing_correct_size_file_and_hashes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data" / "phantom"
    target = root / "tasks" / "ep1" / "meta.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")

    def unexpected_network(url: str, *, timeout: int) -> None:
        pytest.fail("same-size repeat run should not fetch the file again")

    monkeypatch.setattr(phantom, "urlopen", unexpected_network)

    result = phantom.download_file(
        root,
        "a" * 40,
        {
            "path": "tasks/ep1/meta.json",
            "size": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        },
    )

    assert result == {
        "path": "tasks/ep1/meta.json",
        "size": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }


def test_download_replaces_same_size_file_that_fails_recorded_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data" / "phantom"
    target = root / "tasks" / "ep1" / "meta.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(phantom, "urlopen", lambda url, timeout: Response(b"abc"))
    result = phantom.download_file(
        root,
        "a" * 40,
        {
            "path": "tasks/ep1/meta.json",
            "size": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        },
    )

    assert target.read_bytes() == b"abc"
    assert result["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_checksum_mismatch_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data" / "phantom"
    target = root / "tasks" / "ep1" / "meta.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(phantom, "urlopen", lambda url, timeout: Response(b"xyz"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        phantom.download_file(
            root,
            "a" * 40,
            {
                "path": "tasks/ep1/meta.json",
                "size": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            },
        )

    assert target.read_bytes() == b"old"
    assert list(target.parent.glob("*.partial")) == []


@pytest.mark.parametrize(
    "files",
    [
        [],
        [
            {"path": f"tasks/{name}/camera_scene_color.zarr/data/0", "size": 1}
            for name in ("success", "failure", "validation")
        ],
    ],
    ids=["no-files", "missing-meta-json"],
)
def test_download_subset_rejects_incomplete_inventory_before_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: list[dict[str, Any]]
) -> None:
    root = tmp_path / "data" / "phantom"
    selected = [
        _episode(f"tasks/{name}", split="train", failure=False, episode=name)
        for name in ("success", "failure", "validation")
    ]

    def fake_read_url(url: str) -> bytes:
        if "/revision/" in url:
            return json.dumps({"sha": "a" * 40}).encode()
        return b"{}\n"

    monkeypatch.setattr(phantom, "read_url", fake_read_url)
    monkeypatch.setattr(phantom, "select_episodes", lambda rows: selected)
    monkeypatch.setattr(phantom, "list_selected_files", lambda selected, revision: files)

    expected_error = "empty file inventory" if not files else "missing meta.json"
    with pytest.raises(RuntimeError, match=expected_error):
        phantom.download_subset(root, "main", 100)

    assert not (root / "manifests" / "download.json").exists()


def test_download_rejects_symlink_escape_before_network_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data" / "phantom"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "tasks").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        phantom,
        "urlopen",
        lambda url, timeout: pytest.fail("unsafe path must be rejected before network access"),
    )

    with pytest.raises(ValueError, match="escapes download root"):
        phantom.download_file(root, "a" * 40, {"path": "tasks/ep1/meta.json", "size": 1})

    assert list(outside.iterdir()) == []
