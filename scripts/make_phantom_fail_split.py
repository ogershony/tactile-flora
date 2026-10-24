"""Hold out PHANTOM failure demos from the judge training set and make them a test benchmark.

Why: HapticWAM's split puts every PHANTOM failure demo (episode id containing ``_fail_``) in
train and ``phantom_val`` holds successes only, so no held-out PHANTOM failure exists to
evaluate a judge on. This script picks ``n_fail`` failure and ``n_succ`` success episodes of
the train split that have an outcome row in ``train_judge`` and writes:

* ``data/manifest/phantom_fail_test.parquet``: their OUTCOME rows with ``split`` set to
  ``test`` (window ids and every other column unchanged, same parquet schema);
* ``data/manifest/train_judge_pf.parquet``: ``train_judge`` minus every row (dense or outcome)
  of those episodes;
* the matching ``.shifts.json`` sidecars, copied from the inputs' sidecars and keyed
  ``source/episode_id`` like ``tf manifest merge`` writes them.

Selection follows the ``tf manifest`` reproducibility conventions: ``random.Random(seed)`` over
id-sorted candidates, stratified by the object named in the episode id
(``ep_<object>[_fail]_<timestamp>_<index>``) with the quota spread over the objects as evenly
as the candidates allow, rows sorted like ``build_manifest``'s, sidecars written with sorted
keys. The same inputs and arguments give byte-identical outputs.

Must never: modify the input manifests or their sidecars, change a window_id, or write outside
the manifest directory.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from tactile_flora.schema import ManifestRow, episode_key
from tactile_flora.verify.manifest import read_manifest, write_rows

SOURCE = "phantom"
FAIL_MARK = "_fail_"
TEST_NAME = "phantom_fail_test"
TRAIN_NAME = "train_judge_pf"


class Selection(NamedTuple):
    """The chosen episode ids (sorted) of each class."""

    failures: list[str]
    successes: list[str]

    @property
    def episode_ids(self) -> list[str]:
        """Every chosen id, failures first."""
        return [*self.failures, *self.successes]


def is_failure(episode_id: str) -> bool:
    """Return True for a HapticWAM failure demo (``ep_<object>_fail_...``)."""
    return FAIL_MARK in episode_id


def object_of(episode_id: str) -> str:
    """Return the object (stratum) named by the id ``ep_<object>[_fail]_<ts>_<idx>``."""
    parts = episode_id.split("_")
    if len(parts) < 4 or parts[0] != "ep" or not parts[1]:
        raise ValueError(f"not a PHANTOM episode id: {episode_id!r}")
    return parts[1]


def allocate(available: Mapping[str, int], n: int, rng: random.Random) -> dict[str, int]:
    """Spread a quota of ``n`` over the strata as evenly as ``available`` allows.

    One per stratum per round, skipping exhausted strata; the order within a round is a
    seeded shuffle of the sorted stratum names so a remainder goes to a seeded-random
    stratum rather than always the alphabetically first. Raises ValueError if the strata
    cannot supply ``n``.
    """
    total = sum(available.values())
    if n > total:
        raise ValueError(f"need {n} episodes but only {total} are available: {dict(available)}")
    order = sorted(available)
    rng.shuffle(order)
    out = dict.fromkeys(order, 0)
    left = n
    while left > 0:
        for name in order:
            if left == 0:
                break
            if out[name] < available[name]:
                out[name] += 1
                left -= 1
    return out


def candidates(phantom: Iterable[ManifestRow], judge: Iterable[ManifestRow]) -> list[str]:
    """Train-split PHANTOM episodes whose outcome row is in both manifests (sorted ids)."""
    judge_outcome = {
        r.window_id for r in judge if r.source == SOURCE and r.prompt_kind == "outcome"
    }
    ids = {
        r.episode_id
        for r in phantom
        if r.source == SOURCE
        and r.split == "train"
        and r.prompt_kind == "outcome"
        and r.window_id in judge_outcome
    }
    return sorted(ids)


def _pick(ids: Sequence[str], n: int, rng: random.Random) -> list[str]:
    """``n`` ids stratified by object: a seeded sample within each stratum's quota."""
    by_obj: dict[str, list[str]] = {}
    for eid in sorted(ids):
        by_obj.setdefault(object_of(eid), []).append(eid)
    quota = allocate({o: len(v) for o, v in by_obj.items()}, n, rng)
    chosen: list[str] = []
    for obj in sorted(by_obj):
        chosen.extend(rng.sample(by_obj[obj], quota[obj]))
    return sorted(chosen)


def select_episodes(
    phantom: Sequence[ManifestRow],
    judge: Sequence[ManifestRow],
    *,
    seed: int,
    n_fail: int,
    n_succ: int,
) -> Selection:
    """Choose ``n_fail`` failure and ``n_succ`` success episodes with ``random.Random(seed)``."""
    pool = candidates(phantom, judge)
    rng = random.Random(seed)
    fails = _pick([e for e in pool if is_failure(e)], n_fail, rng)
    succs = _pick([e for e in pool if not is_failure(e)], n_succ, rng)
    return Selection(fails, succs)


def _sorted(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    return sorted(rows, key=lambda r: (r.source, r.episode_id, r.prompt_kind, r.anchor_s))


def split_rows(
    phantom: Sequence[ManifestRow], judge: Sequence[ManifestRow], chosen: Iterable[str]
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Return (test rows, train rows).

    Test rows are the chosen episodes' PHANTOM outcome rows with ``split`` ``test``; train
    rows are ``judge`` without any row of a chosen episode, whatever its prompt kind.
    """
    keys = {episode_key(SOURCE, e) for e in chosen}
    test = [
        r.model_copy(update={"split": "test"})
        for r in phantom
        if r.prompt_kind == "outcome" and episode_key(r.source, r.episode_id) in keys
    ]
    train = [r for r in judge if episode_key(r.source, r.episode_id) not in keys]
    return _sorted(test), _sorted(train)


def sidecar_shifts(manifest_path: Path, rows: Iterable[ManifestRow]) -> dict[str, float]:
    """Raw sync lags of ``rows`` from ``<manifest>.shifts.json``, keyed ``source/episode_id``.

    Accepts a per-source sidecar (keys are bare episode ids, as ``build_manifest`` writes)
    and a merged one (keys already ``source/episode_id``). Episodes without an entry (no sync
    shift) are left out, as in ``merge_manifests``.
    """
    side = manifest_path.with_suffix(".shifts.json")
    if not side.exists():
        return {}
    raw: dict[str, float] = json.loads(side.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for src, eid in sorted({(r.source, r.episode_id) for r in rows}):
        key = episode_key(src, eid)
        value = raw.get(key, raw.get(eid))
        if value is not None:
            out[key] = float(value)
    return out


def write_manifest(rows: Sequence[ManifestRow], shifts: Mapping[str, float], out: Path) -> Path:
    """Write ``out`` (parquet, fixed manifest schema) and its sorted ``.shifts.json``."""
    path = write_rows(rows, out)
    path.with_suffix(".shifts.json").write_text(
        json.dumps(dict(shifts), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def run(manifest_dir: Path, *, seed: int, n_fail: int, n_succ: int) -> Selection:
    """Build both manifests under ``manifest_dir`` and print the choice and the counts."""
    phantom_path = manifest_dir / "phantom.parquet"
    judge_path = manifest_dir / "train_judge.parquet"
    phantom = read_manifest(phantom_path)
    judge = read_manifest(judge_path)
    sel = select_episodes(phantom, judge, seed=seed, n_fail=n_fail, n_succ=n_succ)
    test, train = split_rows(phantom, judge, sel.episode_ids)
    test_path = write_manifest(
        test, sidecar_shifts(phantom_path, test), manifest_dir / f"{TEST_NAME}.parquet"
    )
    train_path = write_manifest(
        train, sidecar_shifts(judge_path, train), manifest_dir / f"{TRAIN_NAME}.parquet"
    )

    print(f"seed {seed}: {len(sel.failures)} failure + {len(sel.successes)} success episodes")
    for label, ids in (("failure", sel.failures), ("success", sel.successes)):
        per_obj = Counter(object_of(e) for e in ids)
        print(f"{label} by object: " + ", ".join(f"{o} {n}" for o, n in sorted(per_obj.items())))
        for e in ids:
            print(f"  {label}: {e}")
    n_test_fail = sum(is_failure(r.episode_id) for r in test)
    print(
        f"{test_path}: {len(test)} outcome rows ({n_test_fail} failures, "
        f"{len(test) - n_test_fail} successes), split {sorted({r.split for r in test})}"
    )
    by_source = Counter(r.source for r in train)
    phantom_train = [r for r in train if r.source == SOURCE]
    n_train_fail = sum(is_failure(r.episode_id) for r in phantom_train)
    print(
        f"{train_path}: {len(train)} rows ("
        + ", ".join(f"{s} {n}" for s, n in sorted(by_source.items()))
        + f"); phantom failures {n_train_fail}, phantom successes "
        f"{len(phantom_train) - n_train_fail} (was {len(judge)} rows)"
    )
    return sel


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-fail", type=int, default=20)
    parser.add_argument("--n-succ", type=int, default=20)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifest"))
    args = parser.parse_args(argv)
    run(args.manifest_dir, seed=args.seed, n_fail=args.n_fail, n_succ=args.n_succ)


if __name__ == "__main__":
    main()
