"""Writing and reading the accepted manifest and the frozen holdout id file.

Owns: build_manifest, read_manifest, manifest_hash, freeze_holdout, load_holdout, window_id and
the errors SplitFrozenError, HoldoutExistsError, SourceFailedError.
Must never: change the split of a frozen row, read raw datasets or run checks.

Split assignment: ``splits`` maps a split name to a list of session names and/or episode ids;
an episode not listed is train. Sampling caps (max_episodes, max_windows; 0 means no cap) use
``random.Random(cfg.seed)`` over id-sorted candidates, and rows are sorted, so the same inputs
give a byte-identical parquet.

Caps apply to the TRAIN split only (val/test/benchmark are never thinned): ``max_episodes`` thins
train episodes, and ``max_windows`` (or the per-kind ``max_dense_windows`` /
``max_outcome_windows``, None = use ``max_windows``) thins train windows. Rows of episodes
frozen as train in the holdout file are never dropped by a cap. Sync shifts:
``ManifestRow.shift_s`` is the clock correction (``-lag``, see verify/windowing.py);
the ``<name>.shifts.json`` sidecar keeps the raw lags.

drop_channel: ``dropped_arms`` reads the failed arm(s) from the check detail (``arm=L,arm=R`` or the
older ``L: f_a, ...`` format). A drop_channel whose detail names no arm is treated
as drop_episode (always for single-arm episodes, conservatively for two-arm ones), and so is one
that removes every arm. verify/windowing.prepare_episode removes the named arms from the stream.
An eval_only source may not put rows in the train split (EvalOnlyTrainError).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast, get_args

import pyarrow as pa
import pyarrow.parquet as pq

from tactile_flora.config import SourceConfig
from tactile_flora.schema import (
    Arm,
    CheckResult,
    Episode,
    HoldoutFile,
    ManifestRow,
    Split,
    Window,
    episode_key,
)
from tactile_flora.verify.report import source_verdicts

DEFAULT_HOLDOUT = Path("data/manifest/holdout_ids.json")
_ORIGIN_RANK = {"human": 0, "source": 1, "programmatic": 2, "pseudo": 3}
_SPLITS = get_args(Split)


class SplitFrozenError(RuntimeError):
    """Raised when a build would change the split of a frozen row."""


class HoldoutExistsError(FileExistsError):
    """Raised when freeze_holdout would overwrite an existing holdout file."""


class EvalOnlyTrainError(ValueError):
    """Raised when an eval_only source would contribute rows to the train split."""


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
    """Fallback origin for windows that do not carry one: the top origin of the episode's labels."""
    if window.label_origin != "none":
        return window.label_origin
    kinds = ("outcome",) if window.prompt_kind == "outcome" else ("dense", "event")
    origins = [lab.origin for lab in episode.labels if lab.kind in kinds]
    return min(origins, key=lambda o: _ORIGIN_RANK[o]) if origins else "none"


_ARM_EQ = re.compile(r"\barm=([LR])\b")
_ARM_COLON = re.compile(r"(?:^|;\s*)([LR]):")


def dropped_arms(results: dict[str, CheckResult], arms: list[Arm]) -> set[Arm]:
    """Arms removed by failed drop_channel checks; every arm if a detail does not name one."""
    out: set[Arm] = set()
    for r in results.values():
        if r.passed or r.action != "drop_channel":
            continue
        named = set(_ARM_EQ.findall(r.detail)) or set(_ARM_COLON.findall(r.detail))
        if not named:
            return set(arms)
        out |= {a for a in arms if a in named}
    return out


def shift_of(results: dict[str, CheckResult]) -> float:
    """Clock correction shift_s from a "shift" sync result: -lag (0.0 when there is none)."""
    r = results.get("sync")
    if r is None or r.action != "shift" or r.value is None:
        return 0.0
    return -float(r.value)


def _accepted(results: dict[str, CheckResult], arms: list[Arm] | None = None) -> bool:
    if any(not r.passed and r.action == "drop_episode" for r in results.values()):
        return False
    return not (arms is not None and arms and set(arms) <= dropped_arms(results, arms))


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
        ("shift_s", pa.float64()),
    ]
)


def _cap_windows(
    ordered: list[ManifestRow], cfg: SourceConfig, holdout: HoldoutFile | None
) -> list[ManifestRow]:
    """Thin TRAIN rows to the caps; non-train rows and frozen train episodes are never dropped."""
    per_kind = cfg.max_dense_windows is not None or cfg.max_outcome_windows is not None
    caps: dict[str, int] = {
        "dense": cfg.max_windows if cfg.max_dense_windows is None else cfg.max_dense_windows,
        "outcome": cfg.max_windows if cfg.max_outcome_windows is None else cfg.max_outcome_windows,
    }

    def _frozen(r: ManifestRow) -> bool:
        return (
            holdout is not None
            and holdout.splits.get(episode_key(r.source, r.episode_id)) == "train"
        )

    def _thin(rows: list[ManifestRow], cap: int) -> set[str]:
        if not cap or len(rows) <= cap:
            return {r.window_id for r in rows}
        frozen = [r for r in rows if _frozen(r)]
        free = [r for r in rows if not _frozen(r)]
        room = max(0, cap - len(frozen))
        keep = random.Random(cfg.seed).sample(range(len(free)), min(room, len(free)))
        return {r.window_id for r in frozen} | {free[i].window_id for i in keep}

    train = [r for r in ordered if r.split == "train"]
    keep_ids: set[str] = set()
    if per_kind:
        for kind, cap in caps.items():
            keep_ids |= _thin([r for r in train if r.prompt_kind == kind], cap)
    else:
        keep_ids = _thin(train, cfg.max_windows)
    return [r for r in ordered if r.split != "train" or r.window_id in keep_ids]


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
    failed = sorted(n for n, ok in source_verdicts(check_results).items() if not ok)
    if failed:
        raise SourceFailedError(f"source {source_cfg.name} failed: {', '.join(failed)}")
    cache = hashes or {}
    pairs = sorted(
        (
            (ep, wins)
            for ep, wins in episodes_and_windows
            if ep.episode_id in check_results
            and _accepted(check_results[ep.episode_id], list(ep.force.arms))
        ),
        key=lambda p: p[0].episode_id,
    )

    def _frozen_train(ep: Episode) -> bool:
        return (
            holdout is not None
            and holdout.splits.get(episode_key(ep.source, ep.episode_id)) == "train"
        )

    if source_cfg.max_episodes:
        # only train episodes are thinned; frozen train episodes always stay
        train = [p for p in pairs if assign_split(p[0], splits) == "train"]
        if len(train) > source_cfg.max_episodes:
            frozen = [p for p in train if _frozen_train(p[0])]
            free = [p for p in train if not _frozen_train(p[0])]
            room = max(0, source_cfg.max_episodes - len(frozen))
            picked = set(
                random.Random(source_cfg.seed).sample(range(len(free)), min(room, len(free)))
            )
            dropped = {id(p[0]) for i, p in enumerate(free) if i not in picked}
            pairs = [p for p in pairs if id(p[0]) not in dropped]
    candidates: list[tuple[ManifestRow, Episode]] = []
    for ep, wins in pairs:
        split = assign_split(ep, splits)
        if source_cfg.eval_only and split == "train":
            raise EvalOnlyTrainError(
                f"{ep.source}/{ep.episode_id}: eval_only source cannot be in the train split"
            )
        if holdout is not None:
            frozen = holdout.splits.get(episode_key(ep.source, ep.episode_id))
            if frozen is not None and frozen != split:
                raise SplitFrozenError(
                    f"{ep.source}/{ep.episode_id} is frozen as {frozen!r}, build assigned {split!r}"
                )
        shift = shift_of(check_results[ep.episode_id])
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
                        shift_s=shift,
                    ),
                    ep,
                )
            )
    unique = {row.window_id: row for row, _ in candidates}
    ordered = _cap_windows(sorted(unique.values(), key=lambda r: r.window_id), source_cfg, holdout)
    rows = sorted(ordered, key=lambda r: (r.source, r.episode_id, r.prompt_kind, r.anchor_s))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, Any] = {f: [getattr(r, f) for r in rows] for f in ManifestRow.model_fields}
    pq.write_table(pa.table(cols, schema=_SCHEMA), out)
    shifts = {
        ep.episode_id: check_results[ep.episode_id]["sync"].value
        for ep, _ in pairs
        if check_results[ep.episode_id].get("sync") is not None
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
