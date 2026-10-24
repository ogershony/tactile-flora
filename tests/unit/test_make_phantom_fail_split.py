"""Selection logic of scripts/make_phantom_fail_split.py on a small synthetic manifest."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts import make_phantom_fail_split as pf
from tactile_flora.schema import ManifestRow
from tactile_flora.verify.manifest import read_manifest, window_id, write_rows

OBJECTS = ("Carton", "egg", "waffles", "whiteboard")


def _row(
    episode_id: str,
    kind: str,
    *,
    source: str = "phantom",
    split: str = "train",
    anchor_s: float = 0.0,
    shift_s: float = 0.0,
) -> ManifestRow:
    return ManifestRow(
        window_id=window_id(source, episode_id, kind, anchor_s),
        source=source,
        episode_id=episode_id,
        session=f"session_{episode_id}",
        split=split,  # type: ignore[arg-type]
        prompt_kind=kind,  # type: ignore[arg-type]
        anchor_s=anchor_s,
        frame_hashes=["abcd"] if source == "phantom" else [],
        norm_version="norm",
        label_origin="source" if kind == "outcome" else "programmatic",
        license="CC-BY-4.0",
        report_version="deadbeef",
        shift_s=shift_s,
        norm_group=episode_id.split("_")[1] if source == "phantom" else "all",
    )


def _synthetic() -> tuple[list[ManifestRow], list[ManifestRow]]:
    """A phantom manifest (dense + outcome rows) and the judge manifest built from it.

    Per object: 3 train failures and 3 train successes with an outcome row, one train failure
    without an outcome row, one val success; whiteboard has a single failure. The judge manifest
    holds the phantom train outcome rows plus reassemble_seg train outcome rows, and one stray
    phantom dense row to check that every row of a chosen episode is removed.
    """
    phantom: list[ManifestRow] = []
    for i, obj in enumerate(OBJECTS):
        n_fail = 1 if obj == "whiteboard" else 3
        for k in range(n_fail):
            eid = f"ep_{obj}_fail_17856{i}{k:04d}_00{k}"
            phantom += [_row(eid, "dense"), _row(eid, "outcome", shift_s=0.1 * (k + 1))]
        for k in range(3):
            eid = f"ep_{obj}_17857{i}{k:04d}_00{k}"
            phantom += [_row(eid, "dense"), _row(eid, "dense", anchor_s=10.0), _row(eid, "outcome")]
        phantom.append(_row(f"ep_{obj}_fail_17858{i}0000_009", "dense"))  # no outcome row
        phantom.append(_row(f"ep_{obj}_17859{i}0000_009", "outcome", split="val"))
    judge = [r for r in phantom if r.prompt_kind == "outcome" and r.split == "train"]
    judge += [_row(f"{n}_seg001", "outcome", source="reassemble_seg") for n in range(10)]
    judge.append(_row("ep_Carton_fail_1785600000_000", "dense"))
    return phantom, judge


def test_candidates_are_train_outcome_episodes_present_in_both_manifests() -> None:
    phantom, judge = _synthetic()
    ids = pf.candidates(phantom, judge)
    assert ids == sorted(ids) and len(ids) == 10 + 12
    assert not any("_009" in e for e in ids)  # val rows and outcome-less episodes excluded
    assert all(e.startswith("ep_") for e in ids)


def test_object_of_and_is_failure() -> None:
    assert pf.object_of("ep_Carton_fail_1785610763_001") == "Carton"
    assert pf.object_of("ep_whiteboard_1785783771_000") == "whiteboard"
    assert pf.is_failure("ep_egg_fail_1785683692_000") and not pf.is_failure("ep_egg_1_2")
    with pytest.raises(ValueError):
        pf.object_of("0_seg005")


def test_allocate_spreads_the_quota_evenly_and_refuses_too_many() -> None:
    avail = {"Carton": 12, "egg": 14, "waffles": 15, "whiteboard": 2}
    assert pf.allocate(avail, 20, random.Random(0)) == {
        "Carton": 6,
        "egg": 6,
        "waffles": 6,
        "whiteboard": 2,
    }
    assert pf.allocate({"a": 5, "b": 5}, 5, random.Random(3)) in (
        {"a": 3, "b": 2},
        {"a": 2, "b": 3},
    )
    with pytest.raises(ValueError):
        pf.allocate({"a": 1}, 2, random.Random(0))


def test_selection_is_seeded_stratified_and_order_independent() -> None:
    phantom, judge = _synthetic()
    sel = pf.select_episodes(phantom, judge, seed=0, n_fail=6, n_succ=5)
    assert len(sel.failures) == 6 and len(sel.successes) == 5
    assert all(pf.is_failure(e) for e in sel.failures)
    assert not any(pf.is_failure(e) for e in sel.successes)
    fails = Counter(pf.object_of(e) for e in sel.failures)
    assert fails["whiteboard"] == 1 and set(fails) == set(OBJECTS)
    assert all(1 <= fails[o] <= 2 for o in OBJECTS if o != "whiteboard")
    succ = Counter(pf.object_of(e) for e in sel.successes)
    assert set(succ) == set(OBJECTS) and sorted(succ.values()) == [1, 1, 1, 2]
    # seeded: the same seed gives the same choice whatever the row order; ids come out sorted
    again = pf.select_episodes(phantom[::-1], judge[::-1], seed=0, n_fail=6, n_succ=5)
    assert again == sel
    assert sel.failures == sorted(sel.failures) and sel.successes == sorted(sel.successes)
    with pytest.raises(ValueError):
        pf.select_episodes(phantom, judge, seed=0, n_fail=11, n_succ=0)


def test_split_rows_moves_outcome_rows_to_test_and_drops_every_chosen_row() -> None:
    phantom, judge = _synthetic()
    sel = pf.select_episodes(phantom, judge, seed=1, n_fail=4, n_succ=4)
    test, train = pf.split_rows(phantom, judge, sel.episode_ids)
    assert len(test) == 8 and {r.prompt_kind for r in test} == {"outcome"}
    assert {r.split for r in test} == {"test"} and {r.source for r in test} == {"phantom"}
    assert sorted(r.episode_id for r in test) == sorted(sel.episode_ids)
    chosen = set(sel.episode_ids)
    # the test rows keep their window ids and every other column of the source row
    by_id = {r.window_id: r for r in phantom}
    for r in test:
        assert by_id[r.window_id].model_copy(update={"split": "test"}) == r
    # disjointness: no chosen episode (dense or outcome) stays in train
    assert not {r.episode_id for r in train if r.source == "phantom"} & chosen
    assert {r.window_id for r in train} == {
        r.window_id for r in judge if r.episode_id not in chosen
    }
    assert len(train) == len(judge) - 8 - ("ep_Carton_fail_1785600000_000" in chosen)
    assert test == sorted(test, key=lambda r: (r.source, r.episode_id, r.prompt_kind, r.anchor_s))


def test_sidecar_shifts_accepts_bare_and_prefixed_keys(tmp_path: Path) -> None:
    rows = [_row("ep_egg_1_000", "outcome"), _row("0_seg001", "outcome", source="reassemble_seg")]
    bare = tmp_path / "phantom.parquet"
    bare.with_suffix(".shifts.json").write_text(json.dumps({"ep_egg_1_000": 0.05, "other": 1.0}))
    assert pf.sidecar_shifts(bare, rows) == {"phantom/ep_egg_1_000": 0.05}
    merged = tmp_path / "train_judge.parquet"
    merged.with_suffix(".shifts.json").write_text(
        json.dumps({"phantom/ep_egg_1_000": 0.05, "reassemble_seg/0_seg001": -0.2})
    )
    assert pf.sidecar_shifts(merged, rows) == {
        "phantom/ep_egg_1_000": 0.05,
        "reassemble_seg/0_seg001": -0.2,
    }
    assert pf.sidecar_shifts(tmp_path / "none.parquet", rows) == {}


def test_run_writes_both_manifests_and_sidecars_reproducibly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phantom, judge = _synthetic()
    write_rows(phantom, tmp_path / "phantom.parquet")
    (tmp_path / "phantom.shifts.json").write_text(
        json.dumps({r.episode_id: r.shift_s for r in phantom if r.shift_s})
    )
    write_rows(judge, tmp_path / "train_judge.parquet")
    (tmp_path / "train_judge.shifts.json").write_text(
        json.dumps({f"{r.source}/{r.episode_id}": r.shift_s for r in judge if r.shift_s})
    )
    sel = pf.run(tmp_path, seed=0, n_fail=4, n_succ=4)
    out = capsys.readouterr().out
    assert all(e in out for e in sel.episode_ids) and "8 outcome rows (4 failures" in out
    test = read_manifest(tmp_path / "phantom_fail_test.parquet")
    train = read_manifest(tmp_path / "train_judge_pf.parquet")
    assert [r.episode_id for r in test] == sorted(sel.episode_ids)
    assert {r.split for r in test} == {"test"} and len(train) == len(judge) - 8 - (
        "ep_Carton_fail_1785600000_000" in sel.episode_ids
    )
    assert (
        pq.read_table(tmp_path / "phantom_fail_test.parquet").schema
        == pq.read_table(tmp_path / "phantom.parquet").schema
    )
    shifts = json.loads((tmp_path / "phantom_fail_test.shifts.json").read_text())
    assert shifts == {f"phantom/{e}": r.shift_s for e in sel.failures for r in phantom
                      if r.episode_id == e and r.prompt_kind == "outcome"}  # fmt: skip
    pf_shifts = json.loads((tmp_path / "train_judge_pf.shifts.json").read_text())
    assert not any(k.split("/", 1)[1] in sel.episode_ids for k in pf_shifts)
    first = {p.name: p.read_bytes() for p in tmp_path.glob("*_pf.*")}
    first |= {p.name: p.read_bytes() for p in tmp_path.glob("phantom_fail_test.*")}
    pf.main(["--manifest-dir", str(tmp_path), "--n-fail", "4", "--n-succ", "4"])
    for name, data in first.items():
        assert (tmp_path / name).read_bytes() == data, name
