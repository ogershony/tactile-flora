import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from tactile_flora import cli
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


def test_source_sync_median_failure_blocks_manifest(tmp_path: Path) -> None:
    # Each episode passes the 250 ms gate, but the source median must be under 100 ms.
    bad = {
        e: {**r, "sync": CheckResult(passed=True, value=-0.15, action="shift")}
        for e, r in RESULTS.items()
    }
    with pytest.raises(mf.SourceFailedError, match="sync_median"):
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
    for per in res.values():
        per["sync"] = CheckResult(passed=True, value=0.0)
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


def test_rows_record_the_norm_group(tmp_path: Path) -> None:
    carton = make_pad_episode(7).model_copy(
        update={"source": "phantom", "meta": {"task_folder": "Carton_fail", "object": "Carton"}}
    )
    results = {**RESULTS, carton.episode_id: dict(OK)}
    rows = mf.build_manifest(
        CFG, _pairs([*EPS, carton]), results, "nv", "rv", None, SPLITS, tmp_path / "m.parquet"
    )
    groups = {r.episode_id: r.norm_group for r in rows}
    assert groups[carton.episode_id] == "Carton"
    assert {g for e, g in groups.items() if e != carton.episode_id} == {"all"}
    assert "norm_group" in pq.read_table(tmp_path / "m.parquet").schema.names
    assert mf.read_manifest(tmp_path / "m.parquet") == rows


def test_build_manifest_uses_the_source_sync_gate(tmp_path: Path) -> None:
    lagged = {
        e.episode_id: {**OK, "sync": CheckResult(passed=True, value=0.15, action="shift")}
        for e in EPS
    }
    with pytest.raises(mf.SourceFailedError, match="sync_median"):
        _build(tmp_path / "m.parquet", results=lagged)
    relaxed = CFG.model_copy(update={"sync_median_s": 0.2})
    rows = _build(tmp_path / "m.parquet", cfg=relaxed, results=lagged)
    assert len(rows) == 6 and all(r.shift_s == -0.15 for r in rows)


def test_build_manifest_uses_the_source_p99_gate(tmp_path: Path) -> None:
    import json

    doc = {"grasp": [1.0], "free": [0.0], "p99": [44.0], "problems": []}
    tailed = {
        e.episode_id: {**OK, "normalization": CheckResult(passed=True, detail=json.dumps(doc))}
        for e in EPS
    }
    with pytest.raises(mf.SourceFailedError, match="normalization"):
        _build(tmp_path / "m.parquet", results=tailed)
    relaxed = CFG.model_copy(update={"norm_p99_max": 60.0})
    assert len(_build(tmp_path / "m.parquet", cfg=relaxed, results=tailed)) == 6


def test_build_manifest_uses_the_source_grasp_band(tmp_path: Path) -> None:
    doc = {"grasp": [1.25], "free": [0.0], "p99": [2.0], "problems": []}
    heavy = {
        e.episode_id: {**OK, "normalization": CheckResult(passed=True, detail=json.dumps(doc))}
        for e in EPS
    }
    with pytest.raises(mf.SourceFailedError, match="normalization"):
        _build(tmp_path / "m.parquet", results=heavy)
    relaxed = CFG.model_copy(update={"norm_grasp_range": (0.8, 1.3)})
    assert len(_build(tmp_path / "m.parquet", cfg=relaxed, results=heavy)) == 6


def _other_source(path: Path, results=None) -> list:
    """A second source ("other") with the same six sessions, val = s4, built into ``path``."""
    eps = [e.model_copy(update={"source": "other", "episode_id": f"o-{e.episode_id}"}) for e in EPS]
    res = results or {e.episode_id: dict(OK) for e in eps}
    pairs = [(e, [outcome_window(e, CONSTS)]) for e in eps]
    cfg = CFG.model_copy(update={"name": "other"})
    return mf.build_manifest(cfg, pairs, res, "nv2", "rv2", None, {"val": ["s4"]}, path)


def test_merge_manifests_concatenates_filters_and_keeps_fields(tmp_path: Path) -> None:
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _build(a)
    _other_source(b)
    rows = mf.merge_manifests([a, b], tmp_path / "train.parquet", splits=("train",))
    assert {r.split for r in rows} == {"train"}
    assert len(rows) == 4 + 5  # a: 6 - val - test; b: 6 - val
    by_src = {r.source: r for r in rows}
    assert by_src["synthetic"].report_version == "rv" and by_src["synthetic"].norm_version == "nv"
    assert by_src["other"].report_version == "rv2" and by_src["other"].norm_version == "nv2"
    assert {r.norm_group for r in rows} == {"all"} and {r.license for r in rows} == {"CC0"}
    assert mf.read_manifest(tmp_path / "train.parquet") == rows
    assert rows == sorted(rows, key=lambda r: (r.source, r.episode_id, r.prompt_kind, r.anchor_s))
    assert len(mf.merge_manifests([a, b], tmp_path / "all.parquet")) == 12
    val = mf.merge_manifests([a, b], tmp_path / "val.parquet", splits=["val"])
    assert [(r.source, r.episode_id) for r in val] == [("other", "o-pad-4"), ("synthetic", "pad-4")]
    # identical duplicates collapse; a differing duplicate, an empty or unknown selection refuse
    assert mf.merge_manifests([a, a], tmp_path / "dup.parquet") == mf.read_manifest(a)
    with pytest.raises(ValueError, match="no rows"):
        mf.merge_manifests([a, b], tmp_path / "x.parquet", splits=["benchmark"])
    with pytest.raises(ValueError, match="unknown splits"):
        mf.merge_manifests([a], tmp_path / "x.parquet", splits=["nope"])
    with pytest.raises(ValueError, match="at least one"):
        mf.merge_manifests([a], tmp_path / "x.parquet", splits=[])
    changed = [
        r.model_copy(update={"norm_group": "egg"}) if i == 0 else r
        for i, r in enumerate(mf.read_manifest(a))
    ]
    mf.write_rows(changed, tmp_path / "a2.parquet")
    with pytest.raises(mf.MergeConflictError):
        mf.merge_manifests([a, tmp_path / "a2.parquet"], tmp_path / "x.parquet")
    assert not (tmp_path / "x.parquet").exists()


def test_merge_manifests_filters_prompt_kinds(tmp_path: Path) -> None:
    a = tmp_path / "a.parquet"
    _build(a)
    every = mf.read_manifest(a)
    kinds = {r.prompt_kind for r in every}
    assert kinds == {"outcome"}  # the synthetic source carries outcome windows only
    rows = mf.merge_manifests([a], tmp_path / "o.parquet", prompt_kinds=["outcome"])
    assert rows == every
    with pytest.raises(ValueError, match="no rows"):
        mf.merge_manifests([a], tmp_path / "d.parquet", prompt_kinds=["dense"])
    for bad in ([], ["nope"]):
        with pytest.raises(ValueError, match="prompt kinds"):
            mf.merge_manifests([a], tmp_path / "x.parquet", prompt_kinds=bad)
    (tmp_path / "manifest").mkdir()
    _build(tmp_path / "manifest" / "a.parquet")
    args = ["manifest", "merge", "--out", "o", "--split", "train", "--data-root", str(tmp_path)]
    res = CliRunner().invoke(cli.app, [*args, "--prompt-kind", "outcome", "a"])
    assert res.exit_code == 0 and "rows: 4" in res.output, res.output
    res = CliRunner().invoke(cli.app, [*args, "--prompt-kind", "dense", "a"])
    assert res.exit_code == 1 and "ValueError" in (res.output + str(res.stderr or ""))


def test_merge_shifts_sidecar_keys_by_source(tmp_path: Path) -> None:
    shift = CheckResult(passed=True, value=0.05, action="shift")
    res = {e: {**r, "sync": shift} for e, r in RESULTS.items()}
    a = tmp_path / "a.parquet"
    _build(a, results=res)
    rows = mf.merge_manifests([a], tmp_path / "m.parquet", splits=["train"])
    side = json.loads((tmp_path / "m.shifts.json").read_text())
    assert side == {f"synthetic/{r.episode_id}": 0.05 for r in rows} and len(side) == 4
    # a merged file is itself a valid input (its sidecar is already keyed by source)
    again = mf.merge_manifests([tmp_path / "m.parquet"], tmp_path / "m2.parquet")
    assert again == rows
    assert json.loads((tmp_path / "m2.shifts.json").read_text()) == side


def test_cli_manifest_merge(tmp_path: Path) -> None:
    (tmp_path / "manifest").mkdir()
    _build(tmp_path / "manifest" / "a.parquet")
    _other_source(tmp_path / "manifest" / "b.parquet")
    args = ["manifest", "merge", "--out", "train", "--split", "train", "--data-root", str(tmp_path)]
    res = CliRunner().invoke(cli.app, [*args, "a", "b"])
    assert res.exit_code == 0, res.output
    assert "rows: 9" in res.output and "other train outcome: 5" in res.output
    assert len(mf.read_manifest(tmp_path / "manifest" / "train.parquet")) == 9
    args = ["manifest", "merge", "--out", "x", "--split", "benchmark", "--data-root", str(tmp_path)]
    bad = CliRunner().invoke(cli.app, [*args, "a"])
    assert bad.exit_code == 1 and "ValueError" in (bad.output + str(bad.stderr or ""))
