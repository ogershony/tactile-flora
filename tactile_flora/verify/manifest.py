"""Writing and reading the accepted manifest and the frozen holdout id file.

Owns: build_manifest, read_manifest, manifest_hash, freeze_holdout, load_holdout, window_id and
the errors SplitFrozenError, HoldoutExistsError, SourceFailedError.
Must never: change the split of a frozen row, read raw datasets or run checks.

Split assignment: ``splits`` maps a split name to a list of session names and/or episode ids;
an episode not listed is train. Sampling caps (max_episodes, max_windows; 0 means no cap) use
``random.Random(cfg.seed)`` over id-sorted candidates, and rows are sorted, so the same inputs
give a byte-identical parquet. Clock shifts from the sync check go to a ``<name>.shifts.json``
sidecar because ManifestRow has no field for them.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast, get_args

import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    CheckResult,
    Episode,
    HoldoutFile,
    ManifestRow,
    Split,
    Window,
    episode_key,
)
from tactile_flora.verify.report import SOURCE_LEVEL

DEFAULT_HOLDOUT = Path("data/manifest/holdout_ids.json")
_ORIGIN_RANK = {"human": 0, "source": 1, "programmatic": 2, "pseudo": 3}
_SPLITS = get_args(Split)


class SplitFrozenError(RuntimeError):
    """Raised when a build would change the split of a frozen row."""


class HoldoutExistsError(FileExistsError):
    """Raised when freeze_holdout would overwrite an existing holdout file."""


class SourceFailedError(RuntimeError):
    """Raised when a source-level check failed, so no manifest may be built."""


def window_id(source: str, episode_id: str, prompt_kind: str, anchor_s: float) -> str:
    """Return sha1 of (source, episode_id, prompt_kind, anchor_s) as hex."""
    return hashlib.sha1(f"{source}|{episode_id}|{prompt_kind}|{anchor_s:.6f}".encode()).hexdigest()


def assign_split(episode: Episode, splits: dict[str, list[str]]) -> Split:
    """Return the split whose list names the episode's session or id; train when unlisted."""
    hits = [
        name
        for name, items in splits.items()
        if episode.session in items or episode.episode_id in items
    ]
    if len(hits) > 1:
        raise ValueError(f"{episode.episode_id} is assigned to several splits: {sorted(hits)}")
    split = hits[0] if hits else "train"
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return cast(Split, split)


def _label_origin(episode: Episode, window: Window) -> str:
    kinds = ("outcome",) if window.prompt_kind == "outcome" else ("dense", "event")
    origins = [lab.origin for lab in episode.labels if lab.kind in kinds]
    return min(origins, key=lambda o: _ORIGIN_RANK[o]) if origins else "none"


def _accepted(results: dict[str, CheckResult]) -> bool:
    return not any(not r.passed and r.action == "drop_episode" for r in results.values())


_SCHEMA = pa.schema(
    [
        ("window_id", pa.string()),
        ("source", pa.string()),
        ("episode_id", pa.string()),
        ("session", pa.string()),
        ("split", pa.string()),
        ("prompt_kind", pa.string()),
        ("anchor_s", pa.float64()),
        ("frame_hashes", pa.list_(pa.string())),
        ("norm_version", pa.string()),
        ("label_origin", pa.string()),
        ("license", pa.string()),
        ("report_version", pa.string()),
    ]
)


def build_manifest(
    source_cfg: SourceConfig,
    episodes_and_windows: Iterable[tuple[Episode, list[Window]]],
    check_results: dict[str, dict[str, CheckResult]],
    norm_version: str,
    report_version: str,
    holdout: HoldoutFile | None,
    splits: dict[str, list[str]],
    out_path: Path | str,
    *,
    hashes: dict[str, list[str]] | None = None,
) -> list[ManifestRow]:
    """Write the manifest parquet for one source and return its rows.

    Raises SourceFailedError if a source-level check failed, and SplitFrozenError (before
    writing anything) if an episode's assigned split differs from its frozen split.
    """
    failed = sorted(
        {
            n
            for per in check_results.values()
            for n, r in per.items()
            if n in SOURCE_LEVEL and not r.passed
        }
    )
    if failed:
        raise SourceFailedError(f"source {source_cfg.name} failed: {', '.join(failed)}")
    cache = hashes or {}
    pairs = sorted(
        (
            (ep, wins)
            for ep, wins in episodes_and_windows
            if ep.episode_id in check_results and _accepted(check_results[ep.episode_id])
        ),
        key=lambda p: p[0].episode_id,
    )
    if source_cfg.max_episodes and len(pairs) > source_cfg.max_episodes:
        keep = set(
            random.Random(source_cfg.seed).sample(range(len(pairs)), source_cfg.max_episodes)
        )
        pairs = [p for i, p in enumerate(pairs) if i in keep]
    candidates: list[tuple[ManifestRow, Episode]] = []
    for ep, wins in pairs:
        split = assign_split(ep, splits)
        if holdout is not None:
            frozen = holdout.splits.get(episode_key(ep.source, ep.episode_id))
            if frozen is not None and frozen != split:
                raise SplitFrozenError(
                    f"{ep.source}/{ep.episode_id} is frozen as {frozen!r}, build assigned {split!r}"
                )
        for w in wins:
            candidates.append(
                (
                    ManifestRow(
                        window_id=window_id(ep.source, ep.episode_id, w.prompt_kind, w.anchor_s),
                        source=ep.source,
                        episode_id=ep.episode_id,
                        session=ep.session,
                        split=split,
                        prompt_kind=w.prompt_kind,
                        anchor_s=w.anchor_s,
                        frame_hashes=list(cache.get(ep.episode_id, [])),
                        norm_version=norm_version,
                        label_origin=_label_origin(ep, w),
                        license=ep.license,
                        report_version=report_version,
                    ),
                    ep,
                )
            )
    unique = {row.window_id: row for row, _ in candidates}
    ordered = sorted(unique.values(), key=lambda r: r.window_id)
    if source_cfg.max_windows and len(ordered) > source_cfg.max_windows:
        keep_ids = set(
            random.Random(source_cfg.seed).sample(range(len(ordered)), source_cfg.max_windows)
        )
        ordered = [r for i, r in enumerate(ordered) if i in keep_ids]
    rows = sorted(ordered, key=lambda r: (r.source, r.episode_id, r.prompt_kind, r.anchor_s))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, Any] = {f: [getattr(r, f) for r in rows] for f in ManifestRow.model_fields}
    pq.write_table(pa.table(cols, schema=_SCHEMA), out)
    shifts = {
        ep.episode_id: check_results[ep.episode_id]["sync"].value
        for ep, _ in pairs
        if "sync" in check_results[ep.episode_id]
        and check_results[ep.episode_id]["sync"].action == "shift"
    }
    out.with_suffix(".shifts.json").write_text(
        json.dumps(shifts, indent=2, sort_keys=True), encoding="utf-8"
    )
    return rows


def read_manifest(path: Path | str) -> list[ManifestRow]:
    """Read a manifest parquet into ManifestRow objects."""
    return [ManifestRow.model_validate(r) for r in pq.read_table(path).to_pylist()]


def manifest_hash(path: Path | str) -> str:
    """Return the sha256 hex digest of the manifest file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_holdout(
    manifest_paths: Iterable[Path | str],
    out: Path | str = DEFAULT_HOLDOUT,
    git_hash: str = "unknown",
) -> HoldoutFile:
    """Write the holdout file once from the given manifests; refuses to overwrite.

    Every episode's split at this moment is frozen; the held-out ids are those not in train.
    """
    target = Path(out)
    if target.exists():
        raise HoldoutExistsError(f"{target} already exists; the holdout is frozen once")
    splits: dict[str, Split] = {}
    hashes: dict[str, str] = {}
    for mp in manifest_paths:
        hashes[Path(mp).stem] = manifest_hash(mp)
        for row in read_manifest(mp):
            key = episode_key(row.source, row.episode_id)
            if splits.setdefault(key, row.split) != row.split:
                raise SplitFrozenError(f"{key} has rows in several splits")
    doc = HoldoutFile(git_hash=git_hash, manifests=hashes, splits=dict(sorted(splits.items())))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc


def load_holdout(path: Path | str = DEFAULT_HOLDOUT) -> HoldoutFile:
    """Read the frozen holdout file."""
    return HoldoutFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
