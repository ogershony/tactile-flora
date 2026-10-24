"""Tests for the segment mode of the reassemble_lerobot adapter (the reassemble_seg source)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from tactile_flora.config import ForceConfig, SegmentSelect, SourceConfig, load_config
from tactile_flora.schema import CheckResult, WristArmForce
from tactile_flora.sources.base import get_source, make_adapter
from tactile_flora.sources.reassemble_lerobot import (
    ReassembleLerobotAdapter,
    ReassembleSegAdapter,
    parse_segment_id,
    segment_episode_id,
    segment_qualifies,
    select_segments,
)
from tactile_flora.verify import (
    INHERITED_CHECKS,
    _Parent,
    inherit_results,
    manifest,
    run_manifest,
    run_verify,
)
from tactile_flora.verify.windowing import episode_windows, prompt_kinds
from tactile_flora.video.readers import AutoReader, PyAVReader
from tests.fixtures.reassemble_lerobot import make_fixture as fx

ROOT = Path(__file__).parents[1] / "fixtures" / "reassemble_lerobot"
FORCE_CFG = load_config("configs/force.yaml", ForceConfig)


def _cfg(
    root: Path = ROOT,
    *,
    name: str = "reassemble_seg",
    unit: str = "segment",
    min_duration_s: float = 0.2,
    max_total: dict[str, int] | None = None,
    verify_from: str | None = None,
    inherited_drops: str = "apply",
) -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "name": name,
            "root": str(root),
            "license_allowed": ["CC-BY-4.0"],
            "max_windows": 0,
            "max_episodes": 0,
            "eval_only": False,
            "expected_label_rate": 0.33,
            "seed": 0,
            "episode_unit": unit,
            "segment_select": {
                "min_duration_s": min_duration_s,
                "success_multiple": 2,
                "max_total": max_total or {},
                "seed": 0,
            },
            "verify_from": verify_from,
            "inherited_drops": inherited_drops,
        }
    )


def _seg(root: Path = ROOT, **kw: object) -> ReassembleSegAdapter:
    a = ReassembleSegAdapter(_cfg(root, **kw))  # type: ignore[arg-type]
    a.discover(root)
    return a


def test_registered_and_config_loads() -> None:
    assert get_source("reassemble_seg") is ReassembleSegAdapter
    assert issubclass(ReassembleSegAdapter, ReassembleLerobotAdapter)
    cfg = load_config("configs/sources/reassemble_seg.yaml", SourceConfig)
    assert cfg.episode_unit == "segment" and cfg.verify_from == "reassemble_lerobot"
    assert cfg.inherited_drops == "record" and cfg.root == "data/raw/reassemble_lerobot"
    assert cfg.segment_select == SegmentSelect(
        min_duration_s=1.0, success_multiple=2, max_total={"test": 450}, seed=0
    )
    base = load_config("configs/sources/reassemble_lerobot.yaml", SourceConfig)
    assert (cfg.sync_median_s, cfg.norm_p99_max, cfg.norm_grasp_range) == (
        base.sync_median_s,
        base.norm_p99_max,
        base.norm_grasp_range,
    )
    assert base.episode_unit == "recording" and base.verify_from is None
    a = make_adapter("reassemble_seg")
    assert a.name == "reassemble_seg"
    with pytest.raises(ValueError, match="episode_unit"):
        ReassembleSegAdapter(_cfg(unit="recording"))


def test_segment_ids_round_trip() -> None:
    assert segment_episode_id(56, 3) == "56_seg003"
    assert parse_segment_id("56_seg003") == (56, 3)
    assert parse_segment_id("1_seg1234") == (1, 1234)
    for bad in ("56", "a_seg003", "56_seg", "56_segx"):
        with pytest.raises(ValueError):
            parse_segment_id(bad)


def test_segment_qualifies_rule() -> None:
    t = np.arange(40) / 10.0
    seg = {"frame0": 5, "frame1": 16, "idle": False}  # last frame 1.5, first 0.5: 1.0 s
    assert segment_qualifies(seg, t, 1.0)
    assert not segment_qualifies({**seg, "frame1": 15}, t, 1.0)  # 0.9 s
    assert not segment_qualifies({**seg, "idle": True}, t, 1.0)
    assert not segment_qualifies({"frame0": 5, "frame1": 5, "idle": False}, t, 0.0)


def test_select_segments_matches_the_e0b_idiom() -> None:
    items = [(f"s{i}", True) for i in range(10)] + [(f"f{i}", False) for i in range(3)]
    rule = SegmentSelect(success_multiple=2, seed=0)
    picked = select_segments(items, rule, "train")
    # E0b: random.seed(0); random.shuffle(succ); succ[:min(len, max(0, cap - nf), 2 nf)]
    succ = [eid for eid, ok in items if ok]
    random.seed(0)
    random.shuffle(succ)
    assert picked == ["f0", "f1", "f2"] + succ[:6]
    assert select_segments(items, rule, "train") == picked  # seeded: reproducible
    capped = select_segments(items, SegmentSelect(max_total={"test": 5}), "test")
    assert capped == ["f0", "f1", "f2"] + succ[:2]
    assert select_segments(items, SegmentSelect(max_total={"test": 2}), "test") == [
        "f0",
        "f1",
        "f2",
    ]
    assert select_segments(items, SegmentSelect(max_total={"train": 5}), "test") == picked
    assert select_segments([(f"s{i}", True) for i in range(4)], rule, "train") == []
    assert select_segments(items, SegmentSelect(success_multiple=0.5), "train") == [
        "f0",
        "f1",
        "f2",
        succ[0],
    ]


def test_discover_enumerates_qualifying_segments_and_balances_per_split() -> None:
    a = _seg()
    # episode 0 (train): idle(6) PICK ok(9) INSERT fail(9) idle(6); episode 1 (test): idle
    # fail(3) PICK ok(9). At 30 fps a 9-frame segment spans 0.267 s >= 0.2.
    assert a.discover(ROOT) == ["0_seg002", "0_seg001"]  # failures first, then picked successes
    assert a.splits() == {"train": ["0_seg002", "0_seg001"], "val": [], "test": []}
    # the test recording has no failure, so its successes are not drawn (2 x 0)
    assert _seg(min_duration_s=1.0).discover(ROOT) == []
    assert ReassembleLerobotAdapter(_cfg(name="reassemble_lerobot", unit="recording")).discover(
        ROOT
    ) == ["0", "1"]


def test_segment_episode_is_sliced_from_its_recording() -> None:
    a = _seg()
    rec = ReassembleLerobotAdapter(_cfg(name="reassemble_lerobot", unit="recording"))
    rec.discover(ROOT)
    whole = rec.load("0")
    seg = a.load("0_seg002")
    assert seg.source == "reassemble_seg" and seg.episode_id == "0_seg002"
    assert seg.task == "Insert square peg 3." and seg.session == "2025-01-11"
    assert seg.duration_s == pytest.approx(8 / 30) and seg.license == "CC-BY-4.0"
    assert seg.force.t == pytest.approx([k / 30 for k in range(9)])
    seg_arm, whole_arm = seg.force.arms["R"], whole.force.arms["R"]
    assert isinstance(seg_arm, WristArmForce) and isinstance(whole_arm, WristArmForce)
    assert seg_arm.fz == pytest.approx(whole_arm.fz[15:24])
    assert seg.gripper is not None and whole.gripper is not None
    assert seg.gripper.arms["R"] == pytest.approx(whole.gripper.arms["R"][15:24])
    assert set(seg.cameras) == {"scene", "scene_2", "wrist_right"}
    for role, ref in seg.cameras.items():
        assert ref.path == whole.cameras[role].path and ref.n_frames == 9 and ref.fps == 30
    assert seg.meta["video_from_s"] == pytest.approx({r: 0.5 for r in seg.cameras})
    assert [(lab.kind, lab.origin, lab.outcome) for lab in seg.labels] == [("outcome", "source", 1)]
    assert a.native_labels("0_seg002") == seg.labels
    assert a.load("0_seg001").labels[0].outcome == 5
    m = seg.meta
    assert m["recording"] == "2025-01-11-14-45-48" and m["port_split"] == "train"
    assert m["parent_episode_id"] == "0" and m["episode_index"] == 0 and m["segment_index"] == 2
    assert m["segment"]["frame0"] == 15 and m["segment"]["frame1"] == 24
    assert m["segment_t0_s"] == pytest.approx(0.5) and m["clock_offset_s"] == pytest.approx(0.5)
    assert m["prompt_kinds"] == ["outcome"] and prompt_kinds(seg) == frozenset({"outcome"})
    assert prompt_kinds(whole) == frozenset({"dense", "outcome"})
    assert m["known_issues"] == ["ft_invalid_last_action"]
    assert "exclude_from_norm_fit" not in m
    with pytest.raises(KeyError):
        a.load("0_seg000")  # idle: not a qualifying segment


def test_segment_frames_come_from_the_segment_span_of_a_concatenated_mp4() -> None:
    # recording 1 starts 1.0 s into the shared mp4; its PICK segment starts at frame 3
    a = _seg()
    a._seg_split["1_seg001"] = "test"  # pyright: ignore[reportPrivateUsage]
    a._ids.append("1_seg001")  # pyright: ignore[reportPrivateUsage]
    seg = a.load("1_seg001")
    rec = ReassembleLerobotAdapter(_cfg(name="reassemble_lerobot", unit="recording"))
    rec.discover(ROOT)
    whole = rec.load("1")
    assert seg.meta["video_from_s"]["scene"] == pytest.approx(1.0 + 3 / 30)
    assert seg.cameras["scene"].n_frames == 9 and seg.duration_s == pytest.approx(8 / 30)
    reader = PyAVReader()
    assert reader.frame_count(seg, "scene") == 9
    for i in (0, 4, 8):
        np.testing.assert_array_equal(
            reader.read(seg, "scene", i), reader.read(whole, "scene", 3 + i)
        )
    with pytest.raises(IndexError):
        reader.read(seg, "scene", 9)


def test_segment_windows_are_outcome_only(tmp_path: Path) -> None:
    fx.write_long(tmp_path)
    a = _seg(tmp_path, min_duration_s=1.0)
    assert a.discover(tmp_path) == ["0_seg002", "0_seg001"]
    seg = a.load("0_seg002")
    assert seg.duration_s == pytest.approx(179 / 30)
    (win,) = episode_windows(seg, FORCE_CFG, None, "train")
    assert win.prompt_kind == "outcome" and win.anchor_s == 0.0
    assert win.target == "Rationale: no rationale\n\nScore: 1" and win.label_origin == "source"
    assert len(win.frame_times) == 60  # min(64, floor(10 * 180 / 30)) = 60, already even
    assert all(0 <= t <= seg.duration_s for t in win.frame_times)


def _parent(results: dict[str, dict[str, CheckResult]]) -> _Parent:
    from tactile_flora.force.normalize import ArmNorm, NormConstants

    arm = ArmNorm(baseline=0, grasp=1, sigma=0.1, tau=0.3, scale=1)
    consts = NormConstants(source="p", channel="wrist", groups={"all": {"R": arm}})
    pooled = CheckResult(
        passed=True, value=1.0, detail='{"grasp":[1.0],"free":[0.0],"p99":[2.0],"problems":[]}'
    )
    return _Parent(source="p", results=results, consts=consts, normalization=pooled)


def test_inherit_results_copies_parent_checks_and_honours_inherited_drops() -> None:
    seg = _seg().load("0_seg002")
    own = {
        n: CheckResult(passed=True, detail=f"own {n}")
        for n in ("label_sanity", "spot_check", "prompt_diff", "holdout", "license")
    }
    parent = _parent(
        {
            "0": {
                "integrity": CheckResult(passed=True),
                "sensor_health": CheckResult(
                    passed=False, action="drop_episode", detail="arm=R; R: fz: flat"
                ),
                "sync": CheckResult(
                    passed=False, value=0.5, action="drop_episode", detail="lag R=+500ms"
                ),
                "normalization": CheckResult(
                    passed=False, detail='{"grasp":[3.0],"free":[0.9],"p99":[9.0],"problems":["x"]}'
                ),
                "overlap": CheckResult(passed=True, detail="empty benchmark index"),
            }
        }
    )
    applied = inherit_results(_cfg(verify_from="p"), parent, seg, own)
    assert list(applied) == [
        "integrity", "sensor_health", "sync", "normalization", "label_sanity", "spot_check",
        "prompt_diff", "holdout", "overlap", "license",
    ]  # fmt: skip
    assert set(applied) - set(own) == set(INHERITED_CHECKS)
    assert applied["sensor_health"].action == "drop_episode" and not applied["sensor_health"].passed
    assert applied["sensor_health"].detail == "inherited from p/0: arm=R; R: fz: flat"
    assert applied["sync"].value == 0.5 and applied["sync"].action == "drop_episode"
    assert applied["normalization"] == parent.normalization  # pooled, not the recording's own
    assert applied["integrity"].passed and applied["integrity"].detail == "inherited from p/0"
    assert applied["label_sanity"].detail == "own label_sanity"
    recorded = inherit_results(_cfg(verify_from="p", inherited_drops="record"), parent, seg, own)
    for name in ("sensor_health", "sync"):
        assert not recorded[name].passed and recorded[name].action == "keep"
        assert recorded[name].detail.endswith("(recorded, not applied: inherited_drops=record)")
    assert manifest.shift_of(recorded) == 0.0 and manifest.shift_of(applied) == 0.0
    orphan = seg.model_copy(update={"meta": {**seg.meta, "parent_episode_id": "9"}})
    missing = inherit_results(_cfg(verify_from="p"), parent, orphan, own)
    assert missing["integrity"].action == "drop_episode" and "p/9" in missing["integrity"].detail


def test_verify_from_reuses_the_recording_level_checks(tmp_path: Path) -> None:
    root = tmp_path / "reassemble_lerobot"
    fx.write_long(root)
    data = tmp_path / "data"
    reader = AutoReader(".")
    rec_cfg = _cfg(root, name="reassemble_lerobot", unit="recording").model_copy(
        update={"expected_label_rate": 0.88}
    )
    run_verify(rec_cfg, FORCE_CFG, data, reader)
    n = len(list((data / "verified" / "reassemble_lerobot" / "spot").glob("*.png")))
    parent = run_verify(rec_cfg, FORCE_CFG, data, reader, spot_pass=n)
    assert parent.source_ok, parent.source_verdicts
    with pytest.raises(RuntimeError, match="verify_from"):
        run_verify(_cfg(root, min_duration_s=1.0, verify_from="nope"), FORCE_CFG, data, reader)
    cfg = _cfg(root, min_duration_s=1.0, verify_from="reassemble_lerobot")
    first = run_verify(cfg, FORCE_CFG, data, reader)
    assert first.n_episodes == 2 and first.norm_fit_episodes == 0
    assert first.source_verdicts["spot_check"] is False  # awaiting review
    vdir = data / "verified" / "reassemble_seg"
    assert (
        json.loads((vdir / "norm.json").read_text())["norm_version"]
        == json.loads((data / "verified" / "reassemble_lerobot" / "norm.json").read_text())[
            "norm_version"
        ]
    )
    assert not (vdir / "hashes.parquet").exists()  # no frames hashed (no decode)
    m = len(list((vdir / "spot").glob("*.png")))
    assert m == 2
    summary = run_verify(cfg, FORCE_CFG, data, reader, spot_pass=m)
    assert summary.source_ok, (summary.episode_failures, summary.source_verdicts)
    from tactile_flora.verify import report

    results = report.read_checks(vdir)
    for eid in ("0_seg001", "0_seg002"):
        for name in INHERITED_CHECKS - {"normalization"}:
            assert results[eid][name].detail.startswith("inherited from reassemble_lerobot/0")
        pooled = json.loads(results[eid]["normalization"].detail)
        assert len(pooled["grasp"]) == 1
        assert pooled["inherited_from"] == "reassemble_lerobot"
    rows = manifest.read_manifest(
        run_manifest(cfg, FORCE_CFG, data, name="seg", report_version="rv")
    )
    assert [(r.episode_id, r.prompt_kind, r.split, r.norm_group) for r in rows] == [
        ("0_seg001", "outcome", "train", "all"),
        ("0_seg002", "outcome", "train", "all"),
    ]
    assert {r.frame_hashes == [] for r in rows} == {True}
    assert {r.label_origin for r in rows} == {"source"}
