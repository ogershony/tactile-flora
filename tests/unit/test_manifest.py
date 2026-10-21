from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tactile_flora.config import SourceConfig
from tactile_flora.schema import CheckResult, Episode, HoldoutFile
from tactile_flora.verify import manifest as mf
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.verify_ctx import fit_consts, outcome_window

CFG = SourceConfig(
    name="synthetic",
    root="x",
    license_allowed=["CC0"],
    max_windows=0,
    max_episodes=0,
    eval_only=False,
    expected_label_rate=0.3,
    seed=7,
)
EPS = [make_pad_episode(s) for s in range(6)]
CONSTS = fit_consts(EPS)
OK = {"integrity": CheckResult(passed=True), "spot_check": CheckResult(passed=True)}
RESULTS = {e.episode_id: dict(OK) for e in EPS}
SPLITS = {"val": ["s4"], "test": ["s5"]}


def _pairs(eps: list[Episode] = EPS):
    return [(e, [outcome_window(e, CONSTS)]) for e in eps]


def _build(path: Path, cfg: SourceConfig = CFG, splits=SPLITS, holdout=None, results=RESULTS):
    return mf.build_manifest(cfg, _pairs(), results, "nv", "rv", holdout, splits, path)


def test_rows_fields_and_ids(tmp_path: Path) -> None:
    rows = _build(
        tmp_path / "m.parquet",
        results={**RESULTS, "pad-0": {"license": CheckResult(passed=False, action="drop_episode")}},
    )
    assert len(rows) == 5 and "pad-0" not in {r.episode_id for r in rows}
    r = rows[-1]
    assert r.split == "test" and r.session == "s5" and r.norm_version == "nv"
    assert r.report_version == "rv" and r.license == "CC0" and r.label_origin == "none"
    assert r.window_id == mf.window_id("synthetic", "pad-5", "outcome", 0.0)
    assert len(r.window_id) == 40
    assert {x.split for x in rows} == {"train", "val", "test"}
    assert mf.read_manifest(tmp_path / "m.parquet") == rows
    assert len(mf.manifest_hash(tmp_path / "m.parquet")) == 64


def test_window_id_depends_on_all_parts() -> None:
    base = mf.window_id("a", "e", "dense", 1.0)
    assert (
        len(
            {
                base,
                mf.window_id("b", "e", "dense", 1.0),
                mf.window_id("a", "f", "dense", 1.0),
                mf.window_id("a", "e", "outcome", 1.0),
                mf.window_id("a", "e", "dense", 2.0),
            }
        )
        == 5
    )


def test_source_failure_blocks_manifest(tmp_path: Path) -> None:
    bad = {e: {**r, "spot_check": CheckResult(passed=False)} for e, r in RESULTS.items()}
    with pytest.raises(mf.SourceFailedError):
        _build(tmp_path / "m.parquet", results=bad)
    assert not (tmp_path / "m.parquet").exists()


def test_caps_and_seed_are_byte_identical(tmp_path: Path) -> None:
    capped = CFG.model_copy(update={"max_episodes": 4, "max_windows": 3})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    rows = _build(a, capped)
    _build(b, capped)
    # caps thin train only: 3 train rows + the val and the test row
    assert sorted(r.split for r in rows) == ["test", "train", "train", "train", "val"]
    assert a.read_bytes() == b.read_bytes()
    assert mf.manifest_hash(a) == mf.manifest_hash(b)
    other = _build(tmp_path / "c.parquet", capped.model_copy(update={"seed": 8}))
    assert {r.window_id for r in other} != {r.window_id for r in rows}


def test_caps_never_thin_eval_and_frozen_train_stays(tmp_path: Path) -> None:
    capped = CFG.model_copy(update={"max_episodes": 1, "max_windows": 1})
    rows = _build(tmp_path / "a.parquet", capped)
    assert [r.split for r in rows].count("val") == 1 and [r.split for r in rows].count("test") == 1
    assert [r.split for r in rows].count("train") == 1
    doc = HoldoutFile(
        git_hash="g",
        manifests={},
        splits={"synthetic/pad-0": "train", "synthetic/pad-1": "train", "synthetic/pad-4": "val"},
    )
    rows = _build(tmp_path / "b.parquet", capped, holdout=doc)
    train = {r.episode_id for r in rows if r.split == "train"}
    assert {"pad-0", "pad-1"} <= train  # frozen train episodes survive both caps


def test_per_kind_caps_are_train_only(tmp_path: Path) -> None:
    cfg = CFG.model_copy(update={"max_dense_windows": 1, "max_outcome_windows": 2})
    rows = _build(tmp_path / "a.parquet", cfg)
    train = [r for r in rows if r.split == "train"]
    assert len(train) == 2 and {r.prompt_kind for r in train} == {"outcome"}
    assert len(rows) == 2 + 2  # val and test untouched


def test_eval_only_source_refuses_train_rows(tmp_path: Path) -> None:
    with pytest.raises(mf.EvalOnlyTrainError):
        _build(tmp_path / "a.parquet", CFG.model_copy(update={"eval_only": True}))
    ok = {"val": [f"s{i}" for i in range(4)] + ["s4"], "test": ["s5"]}
    assert _build(tmp_path / "b.parquet", CFG.model_copy(update={"eval_only": True}), splits=ok)


def test_shift_s_and_drop_channel(tmp_path: Path) -> None:
    res = {e: dict(r) for e, r in RESULTS.items()}
    res["pad-0"]["sync"] = CheckResult(passed=True, value=0.25, action="shift")
    res["pad-1"]["sensor_health"] = CheckResult(passed=False, action="drop_channel", detail="arm=L")
    rows = _build(tmp_path / "a.parquet", results=res)
    by = {r.episode_id: r for r in rows}
    assert by["pad-0"].shift_s == -0.25 and by["pad-2"].shift_s == 0.0
    assert "pad-1" in by  # only arm L is dropped; windowing removes it, the episode stays
    res["pad-2"]["sensor_health"] = CheckResult(passed=False, action="drop_channel", detail="")
    assert "pad-2" not in {r.episode_id for r in _build(tmp_path / "b.parquet", results=res)}
    assert mf.dropped_arms(res["pad-1"], ["L", "R"]) == {"L"}
    r = CheckResult(passed=False, action="drop_channel", detail="R: f_a, f_b; L: f_a")
    assert mf.dropped_arms({"x": r}, ["L", "R"]) == {"L", "R"}
    r2 = CheckResult(passed=False, action="drop_channel", detail="")
    assert mf.dropped_arms({"x": r2}, ["L", "R"]) == {"L", "R"}


def test_freeze_then_rebuild_refuses_split_change(tmp_path: Path) -> None:
    m = tmp_path / "manifest" / "train.parquet"
    _build(m)
    frozen = tmp_path / "manifest" / "holdout_ids.json"
    doc = mf.freeze_holdout([m], frozen, git_hash="abc")
    assert doc.held_out_ids == {"synthetic/pad-4", "synthetic/pad-5"}
    assert mf.load_holdout(frozen) == doc
    with pytest.raises(mf.HoldoutExistsError):
        mf.freeze_holdout([m], frozen)

    before = m.read_bytes()
    moved = {"val": ["s5"], "test": ["s4"]}  # swap two frozen episodes
    with pytest.raises(mf.SplitFrozenError):
        _build(m, splits=moved, holdout=doc)
    leak = {"val": [], "test": []}  # a frozen val episode would become train
    with pytest.raises(mf.SplitFrozenError):
        _build(m, splits=leak, holdout=doc)
    assert m.read_bytes() == before
    # unchanged assignment still rebuilds identically
    _build(m, holdout=doc)
    assert m.read_bytes() == before


def test_new_episode_after_freeze_is_allowed(tmp_path: Path) -> None:
    m = tmp_path / "m.parquet"
    doc = HoldoutFile(splits={"synthetic/pad-4": "val"})
    rows = _build(m, holdout=doc)
    assert {r.episode_id: r.split for r in rows}["pad-4"] == "val"


def test_freeze_reads_pq_and_sorted(tmp_path: Path) -> None:
    m = tmp_path / "m.parquet"
    _build(m)
    assert pq.read_table(m).num_rows == 6
    doc = mf.freeze_holdout([m], tmp_path / "h.json")
    assert list(doc.splits) == sorted(doc.splits) and doc.manifests == {"m": mf.manifest_hash(m)}


def test_shifts_sidecar(tmp_path: Path) -> None:
    res = {
        e: {**r, "sync": CheckResult(passed=True, value=0.05, action="shift")}
        for e, r in RESULTS.items()
    }
    m = tmp_path / "m.parquet"
    _build(m, results=res)
    assert '"pad-0": 0.05' in m.with_suffix(".shifts.json").read_text()


def test_unknown_split_rejected() -> None:
    with pytest.raises(ValueError):
        mf.assign_split(EPS[0], {"bogus": ["s0"]})
    with pytest.raises(ValueError):
        mf.assign_split(EPS[0], {"val": ["s0"], "test": ["pad-0"]})
