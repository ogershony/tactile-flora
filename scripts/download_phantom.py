"""Download a bounded, revision-pinned PHANTOM acceptance subset; never fetch all streams.

Owns subset selection, size limits and download provenance under data/.
Must never change source splits or claim that a subset is the complete dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import urlopen

REPO = "armteam/hapticwam-teleop-raw"
BASE = f"https://huggingface.co/datasets/{REPO}"
API = f"https://huggingface.co/api/datasets/{REPO}"
STREAMS = ("camera_scene_color", "tactile_left_wrench", "tactile_right_wrench", "gripper")
CHUNK_SIZE = 1024 * 1024
TIMEOUT = 120


def read_url(url: str) -> bytes:
    """Read a small catalog or metadata response with a finite timeout."""
    with urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def safe_remote_path(value: object) -> PurePosixPath:
    """Validate a repository path before using it in a request or filesystem path."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe remote path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe remote path {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"unsafe remote path {value!r}")
    return path


def contained_target(root: Path, remote_path: object) -> tuple[PurePosixPath, Path]:
    """Return a validated relative path and a target that stays below root."""
    path = safe_remote_path(remote_path)
    resolved_root = root.resolve()
    target = resolved_root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(resolved_root):
        raise ValueError(f"remote path escapes download root: {path}")
    return path, target


def select_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose the first contact success, failure and validation episode in catalog order."""
    selected = []
    for split, failure in (("train", False), ("train", True), ("val", False)):
        candidate = next(
            (
                row
                for row in rows
                if row.get("tactile_contact")
                and row.get("failure_demo") == failure
                and row.get("split") == split
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(f"no contact episodes for split={split}, failure={failure}")
        safe_remote_path(candidate.get("path"))
        selected.append(candidate)
    return selected


def select_stratified_episodes(
    rows: list[dict[str, Any]], per_stratum: int, seed: int
) -> list[dict[str, Any]]:
    """Sample each split/failure stratum across tasks, independently of acceptance results."""
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for split, failure in (("train", False), ("train", True), ("val", False)):
        buckets: dict[str, list[dict[str, Any]]] = {}
        seen: set[str] = set()
        for row in sorted(rows, key=lambda r: str(r.get("episode", ""))):
            if not (
                row.get("tactile_contact")
                and row.get("failure_demo") == failure
                and row.get("split") == split
            ):
                continue
            path = safe_remote_path(row.get("path"))
            eid = str(row["episode"])
            if eid in seen:
                raise ValueError(f"duplicate catalog episode: {eid}")
            seen.add(eid)
            buckets.setdefault(str(row.get("task", path.parent.name)), []).append(row)
        if len(seen) < per_stratum:
            raise RuntimeError(f"need {per_stratum} contact episodes for {split=}, {failure=}")
        tasks = sorted(buckets)
        rng.shuffle(tasks)
        for task in tasks:
            rng.shuffle(buckets[task])
        chosen: list[dict[str, Any]] = []
        while len(chosen) < per_stratum:
            for task in tasks:
                if buckets[task]:
                    chosen.append(buckets[task].pop())
                    if len(chosen) == per_stratum:
                        break
        selected.extend(chosen)
    return selected


def _is_selected_file(remote_file: object, episode_path: PurePosixPath) -> bool:
    """Check whether a validated file belongs to meta or a requested stream."""
    path = safe_remote_path(remote_file)
    prefix = episode_path.as_posix() + "/"
    if not path.as_posix().startswith(prefix):
        return False
    relative = path.as_posix()[len(prefix) :]
    return relative == "meta.json" or any(
        relative.startswith(f"{stream}.zarr/") or relative.startswith(f"{stream}/")
        for stream in STREAMS
    )


def _next_page(link_header: str, current_url: str, route_prefix: str) -> str | None:
    """Accept pagination only from the same Hugging Face API route."""
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        start, end = part.find("<"), part.find(">")
        if start < 0 or end <= start:
            raise ValueError("malformed pagination link")
        next_url = urljoin(current_url, part[start + 1 : end])
        parsed = urlsplit(next_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "huggingface.co"
            or not parsed.path.startswith(route_prefix)
        ):
            raise ValueError("pagination link leaves the expected Hugging Face API route")
        return next_url
    return None


def list_selected_files(selected: list[dict[str, Any]], revision: str) -> list[dict[str, Any]]:
    """Walk only the selected episode trees and return their required files."""
    files: list[dict[str, Any]] = []
    for episode in selected:
        episode_path = safe_remote_path(episode.get("path"))
        route = f"/api/datasets/{REPO}/tree/{revision}/"
        url: str | None = (
            f"https://huggingface.co{route}{quote(episode_path.as_posix())}?recursive=true&limit=1000"
        )
        while url:
            with urlopen(url, timeout=TIMEOUT) as response:
                entries = json.load(response)
                next_url = _next_page(response.headers.get("Link", ""), url, route)
            if not isinstance(entries, list):
                raise ValueError("invalid file tree response")
            for entry in entries:
                if entry.get("type") != "file":
                    continue
                remote_path = safe_remote_path(entry.get("path"))
                if not _is_selected_file(remote_path.as_posix(), episode_path):
                    continue
                size = entry.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise ValueError(f"invalid remote file size for {remote_path}")
                files.append({"path": remote_path.as_posix(), "size": size})
            url = next_url
    # API pagination or overlapping records must not cause duplicate downloads/size accounting.
    by_path = {row["path"]: row for row in files}
    if len(by_path) != len(files):
        sizes_by_path: dict[str, set[int]] = {}
        for row in files:
            sizes_by_path.setdefault(row["path"], set()).add(row["size"])
        if any(len(sizes) != 1 for sizes in sizes_by_path.values()):
            raise ValueError("file tree returned conflicting sizes for a remote path")
    return list(by_path.values())


def validate_inventory(selected: list[dict[str, Any]], files: list[dict[str, Any]]) -> None:
    """Require metadata and at least one requested stream for each selected episode."""
    for episode in selected:
        episode_path = safe_remote_path(episode.get("path")).as_posix()
        prefix = episode_path + "/"
        episode_files = [row["path"] for row in files if row["path"].startswith(prefix)]
        if not episode_files:
            raise RuntimeError(f"empty file inventory for selected episode {episode_path}")
        if f"{episode_path}/meta.json" not in episode_files:
            raise RuntimeError(f"missing meta.json for selected episode {episode_path}")
        has_stream = any(
            any(
                path.startswith(f"{episode_path}/{stream}.zarr/")
                or path.startswith(f"{episode_path}/{stream}/")
                for stream in STREAMS
            )
            for path in episode_files
        )
        if not has_stream:
            raise RuntimeError(f"no requested stream files for selected episode {episode_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(root: Path, revision: str, row: dict[str, Any]) -> dict[str, Any]:
    """Download one file with a strict byte ceiling and atomic replacement."""
    path, target = contained_target(root, row.get("path"))
    size = row.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"invalid expected size for {path}")
    expected_hash = row.get("sha256")
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise ValueError(f"invalid expected SHA-256 for {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after creating directories, which may have followed an existing symlink.
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"remote path escapes download root: {path}")
    reusable = (
        target.is_file()
        and target.stat().st_size == size
        and expected_hash is not None
        and _sha256(target) == expected_hash
    )
    if not reusable:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".partial", dir=target.parent
        )
        temporary = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as destination:
                url = f"{BASE}/resolve/{quote(revision, safe='')}/{quote(path.as_posix())}"
                with urlopen(url, timeout=TIMEOUT) as source:
                    written = 0
                    while chunk := source.read(min(CHUNK_SIZE, size - written + 1)):
                        written += len(chunk)
                        if written > size:
                            raise RuntimeError(
                                f"oversized download for {path}; expected {size} bytes"
                            )
                        destination.write(chunk)
                if written != size:
                    raise RuntimeError(
                        f"incomplete download {path}: received {written} of {size} bytes"
                    )
                destination.flush()
                os.fsync(destination.fileno())
            digest = _sha256(temporary)
            if expected_hash is not None and digest != expected_hash:
                raise RuntimeError(f"checksum mismatch for {path}")
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    else:
        digest = _sha256(target)
    return {"path": path.as_posix(), "size": size, "sha256": digest}


def atomic_write(path: Path, contents: str) -> None:
    """Write text to a temporary sibling and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def download_subset(
    root: Path,
    revision_name: str,
    max_bytes: int,
    *,
    per_stratum: int = 1,
    seed: int = 0,
    plan_only: bool = False,
) -> None:
    """Resolve a revision, select the bounded subset, and record its provenance."""
    revision_data = json.loads(read_url(f"{API}/revision/{quote(revision_name, safe='')}"))
    revision = revision_data["sha"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdefABCDEF" for char in revision)
    ):
        raise ValueError("revision API returned an invalid commit SHA")
    raw = read_url(f"{BASE}/resolve/{quote(revision, safe='')}/manifests/all.jsonl")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected = (
        select_episodes(rows)
        if per_stratum == 1
        else select_stratified_episodes(rows, per_stratum, seed)
    )
    files = list_selected_files(selected, revision)
    validate_inventory(selected, files)
    total = sum(row["size"] for row in files)
    if total > max_bytes:
        raise RuntimeError(f"subset needs {total} bytes, exceeds limit {max_bytes}")
    print(
        f"revision={revision} catalog={len(rows)} subset={len(selected)} "
        f"files={len(files)} bytes={total}",
        flush=True,
    )
    selection = {
        "method": "first-in-catalog" if per_stratum == 1 else "seeded-task-balanced",
        "per_stratum": per_stratum,
        "seed": seed,
        "catalog_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if plan_only:
        atomic_write(
            root / "manifests" / "plan.json",
            json.dumps(
                {
                    "revision": revision,
                    "selection": selection,
                    "episodes": selected,
                    "bytes": total,
                    "files": files,
                },
                indent=2,
            )
            + "\n",
        )
        return

    previous_hashes: dict[str, str] = {}
    previous_manifest = root / "manifests" / "download.json"
    if previous_manifest.is_file():
        try:
            previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
            if previous.get("revision") == revision:
                previous_hashes = {
                    row["path"]: row["sha256"]
                    for row in previous.get("files", [])
                    if isinstance(row, dict)
                    and isinstance(row.get("path"), str)
                    and isinstance(row.get("sha256"), str)
                    and len(row["sha256"]) == 64
                    and all(char in "0123456789abcdef" for char in row["sha256"])
                }
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            previous_hashes = {}
    download_rows = [{**row, "sha256": previous_hashes.get(row["path"])} for row in files]
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(lambda row: download_file(root, revision, row), download_rows))
    metadata = root / "manifests"
    metadata.mkdir(parents=True, exist_ok=True)
    if not metadata.resolve().is_relative_to(root.resolve()):
        raise ValueError("manifest directory escapes download root")
    all_manifest = "".join(json.dumps(row) + "\n" for row in selected)
    download_manifest = {
        "repo": REPO,
        "revision": revision,
        "catalog_episodes": len(rows),
        "subset_episodes": [row["episode"] for row in selected],
        "bytes": total,
        "selection": selection,
        "files": downloaded,
    }
    atomic_write(metadata / "all.jsonl", all_manifest)
    atomic_write(metadata / "download.json", json.dumps(download_manifest, indent=2) + "\n")
    print("subset complete", flush=True)


def main() -> None:
    """Select a contact success, failure and validation episode, then fetch needed streams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/phantom"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-bytes", type=int, default=250_000_000)
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=1,
        help="Episodes per train-success/train-failure/validation stratum",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    resolved_root = args.root.resolve()
    if not resolved_root.is_relative_to(Path("data").resolve()):
        parser.error("--root must be under data/")
    if args.max_bytes < 0:
        parser.error("--max-bytes must be nonnegative")
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    download_subset(
        resolved_root,
        args.revision,
        args.max_bytes,
        per_stratum=args.per_stratum,
        seed=args.seed,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
