import json
from pathlib import Path

import numpy as np

from tactile_flora.config import RenderConfig
from tactile_flora.force.normalize import ArmNorm, NormConstants
from tactile_flora.schema import (
    Arm,
    BenchmarkHashRow,
    Episode,
    Label,
    PadArmForce,
    Span,
    WristArmForce,
)
from tactile_flora.verify import checks, hashing
from tests.fixtures.synthetic import make_pad_episode, make_wrist_episode
from tests.fixtures.synthetic_video import SyntheticFrameReader
from tests.fixtures.verify_ctx import FORCE_CFG, clean_episodes, make_ctx, outcome_window

EPS = clean_episodes()
EP = EPS[0]
CTX = make_ctx(EPS)


def test_all_checks_listed_in_spec_order() -> None:
    assert checks.CHECK_NAMES == [
        "integrity",
        "sensor_health",
        "sync",
        "normalization",
        "label_sanity",
        "spot_check",
        "prompt_diff",
        "holdout",
        "overlap",
        "license",
    ]


def test_clean_episode_passes_every_check() -> None:
    results = checks.run_checks(EP, CTX)
    assert list(results) == checks.CHECK_NAMES
    assert {n: r.detail for n, r in results.items() if not r.passed} == {}


def test_checks_do_not_mutate_the_episode() -> None:
    before = EP.model_dump_json()
    checks.run_checks(EP, CTX)
    assert EP.model_dump_json() == before


def test_integrity_nonmonotonic_timestamps() -> None:
    t = list(EP.force.t)
    t[50], t[51] = t[51], t[50]
    bad = EP.model_copy(update={"force": EP.force.model_copy(update={"t": t})})
    r = checks.check_integrity(bad, CTX)
    assert not r.passed and r.action == "drop_episode" and "increasing" in r.detail


def test_integrity_video_problems() -> None:
    r = checks.check_integrity(EP, make_ctx(EPS, reader=SyntheticFrameReader(fail={EP.episode_id})))
    assert not r.passed and "does not decode" in r.detail
    r = checks.check_integrity(
        EP, make_ctx(EPS, reader=SyntheticFrameReader(counts={EP.episode_id: 7}))
    )
    assert not r.passed and "decoded 7" in r.detail


def test_integrity_wrong_force_rate() -> None:
    bad = EP.model_copy(update={"force": EP.force.model_copy(update={"rate_hz": 500.0})})
    assert not checks.check_integrity(bad, CTX).passed


def _flat_pad(ep: Episode, arm: str, name: str) -> Episode:
    data = ep.force.arms[arm]  # type: ignore[index]
    assert isinstance(data, PadArmForce)
    flat = data.model_copy(update={name: [1.0] * len(getattr(data, name))})
    arms = {**ep.force.arms, arm: flat}
    return ep.model_copy(update={"force": ep.force.model_copy(update={"arms": arms})})


def test_sensor_health_flat_channel() -> None:
    r = checks.check_sensor_health(_flat_pad(EP, "L", "f_a"), CTX)
    assert not r.passed and r.action == "drop_channel" and "L: f_a: flat" in r.detail


def test_sensor_health_all_channels_dead_drops_episode() -> None:
    bad = _flat_pad(_flat_pad(EP, "L", "f_a"), "L", "f_b")
    bad = _flat_pad(_flat_pad(bad, "R", "f_a"), "R", "f_b")
    assert checks.check_sensor_health(bad, CTX).action == "drop_episode"


def test_sensor_health_saturation_and_baseline() -> None:
    data = EP.force.arms["L"]
    assert isinstance(data, PadArmForce)
    clipped = [min(v, 1.5) for v in data.f_a]
    sat = EP.model_copy(
        update={
            "force": EP.force.model_copy(
                update={"arms": {**EP.force.arms, "L": data.model_copy(update={"f_a": clipped})}}
            )
        }
    )
    r = checks.check_sensor_health(sat, CTX)
    assert not r.passed and "saturated" in r.detail
    stats = {f"{EP.session}/{a}/{c}": (9.0, 0.05) for a in "LR" for c in ("f_a", "f_b")}
    assert not checks.check_sensor_health(EP, make_ctx(EPS, baseline_stats=stats)).passed


def test_sync_clean_records_small_lag() -> None:
    r = checks.check_sync(EP, CTX)
    assert r.passed and r.value is not None and abs(r.value) < 0.1
    assert r.action in ("keep", "shift")


def test_sync_400ms_lag_drops_episode() -> None:
    t = [x + 0.4 for x in EP.force.t]
    bad = EP.model_copy(update={"force": EP.force.model_copy(update={"t": t})})
    r = checks.check_sync(bad, CTX)
    assert not r.passed and r.action == "drop_episode"
    assert r.value is not None and 0.25 < r.value < 0.6


def test_sync_without_gripper_passes() -> None:
    assert checks.check_sync(make_pad_episode(1, with_gripper=False), CTX).passed


def test_normalization_fails_on_wrong_constants() -> None:
    c = CTX.norm
    arms = {a: ArmNorm(**{**n.model_dump(), "scale": 3 * n.scale}) for a, n in c.arms.items()}
    wrong = NormConstants(source=c.source, channel=c.channel, arms=arms)
    r = checks.check_normalization(EP, make_ctx(EPS, norm=wrong))
    assert not r.passed and "grasp median" in r.detail
    assert not checks.check_normalization(EP, make_ctx(EPS, norm=None)).passed


def test_normalization_value_near_one() -> None:
    r = checks.check_normalization(EP, CTX)
    assert r.value is not None and abs(r.value - 1.0) < 0.35


def _labeled(bad: bool) -> Episode:
    q = 1 if bad else 3
    lab = Label(kind="dense", origin="source", spans=[Span(t0=0, t1=1, quality=q)])  # type: ignore[arg-type]
    return EP.model_copy(update={"labels": [lab]})


def test_label_sanity() -> None:
    assert checks.has_bad_label(_labeled(True)) and not checks.has_bad_label(_labeled(False))
    assert checks.observed_label_rate([_labeled(True), _labeled(False)]) == 0.5
    assert checks.observed_label_rate([EP]) is None
    n = {"labeled_episode_count": 20}
    assert checks.check_label_sanity(EP, make_ctx(EPS, observed_label_rate=0.5, **n)).passed
    r = checks.check_label_sanity(EP, make_ctx(EPS, observed_label_rate=0.9, **n))
    assert not r.passed
    assert not checks.check_label_sanity(EP, make_ctx(EPS, observed_label_rate=0.1, **n)).passed
    # below the minimum labeled-episode count the rate is not assessed
    few = make_ctx(EPS, observed_label_rate=0.9, labeled_episode_count=3)
    assert checks.check_label_sanity(EP, few).passed


def test_spot_check_threshold() -> None:
    assert checks.check_spot_check(EP, make_ctx(EPS, spot_reviewer_pass=17)).passed
    assert not checks.check_spot_check(EP, make_ctx(EPS, spot_reviewer_pass=16)).passed
    r = checks.check_spot_check(EP, make_ctx(EPS, spot_reviewer_pass=None))
    assert not r.passed and "awaiting" in r.detail
    assert not checks.check_spot_check(EP, make_ctx(EPS, spot_n=0)).passed


def test_spot_strips_and_review(tmp_path: Path) -> None:
    reader = SyntheticFrameReader()
    items = [(e, outcome_window(e, CTX.norm)) for e in EPS]
    d = tmp_path / "spot"
    idx = checks.write_spot_strips("synthetic", items, reader, d, n=3, seed=1)
    assert idx == d / "index.json" and len(list(d.glob("*.png"))) == 3
    assert checks.read_spot_index(d) == (3, None)
    checks.record_spot_review(d, 3)
    assert checks.read_spot_index(d) == (3, 3)
    again = checks.write_spot_strips("synthetic", items, reader, tmp_path / "b", n=3, seed=1)
    assert [p.name for p in sorted(d.glob("*.png"))] == [
        p.name for p in sorted(again.parent.glob("*.png"))
    ]
    r = checks.check_spot_check(EP, make_ctx(EPS, spot_n=3, spot_reviewer_pass=3))
    assert r.passed


def test_prompt_diff_passes_and_budget_fails() -> None:
    r = checks.check_prompt_diff(EP, CTX)
    assert r.passed and r.value is not None and r.value > 0
    tiny = RenderConfig(**{**FORCE_CFG.render.model_dump(), "max_tokens": 10})
    r = checks.check_prompt_diff(EP, make_ctx(EPS, render_cfg=tiny))
    assert not r.passed and "budget" in r.detail


def test_holdout() -> None:
    assert checks.check_holdout(EP, make_ctx(EPS, holdout_ids=frozenset({"other"}))).passed
    bad = make_ctx(EPS, holdout_ids=frozenset({EP.episode_id}))
    r = checks.check_holdout(EP, bad)
    assert not r.passed and r.action == "drop_episode"
    keyed = make_ctx(EPS, holdout_ids=frozenset({f"synthetic/{EP.episode_id}"}))
    assert not checks.check_holdout(EP, keyed).passed
    # frozen as held out and assigned to val is fine
    ok = make_ctx(EPS, holdout_ids=frozenset({EP.episode_id}), split_of={EP.episode_id: "val"})
    assert checks.check_holdout(EP, ok).passed


def test_holdout_pseudo_in_eval_split() -> None:
    pseudo = EP.model_copy(update={"labels": [Label(kind="dense", origin="pseudo")]})
    w = outcome_window(pseudo, CTX.norm, split="test")
    ctx = make_ctx(EPS, windows={pseudo.episode_id: [w]})
    assert not checks.check_holdout(pseudo, ctx).passed
    w_train = outcome_window(pseudo, CTX.norm, split="train")
    assert checks.check_holdout(
        pseudo, make_ctx(EPS, windows={pseudo.episode_id: [w_train]})
    ).passed


def test_overlap_planted_duplicate() -> None:
    reader = SyntheticFrameReader()
    bench_ep = make_pad_episode(50).model_copy(update={"source": "bench"})
    row = hashing.episode_hashes(bench_ep, reader)
    index = [
        BenchmarkHashRow(
            source="bench", episode_id="b", h_first=row[0], h_mid=row[1], h_last=row[2]
        )
    ]
    assert checks.check_overlap(EP, make_ctx(EPS, benchmark_index=index)).passed
    dup = SyntheticFrameReader(alias={EP.episode_id: "pad-50"})
    r = checks.check_overlap(EP, make_ctx(EPS, benchmark_index=index, reader=dup))
    assert not r.passed and r.action == "drop_episode" and r.value == 0.0
    # a row that only matches two of three hashes is not an overlap
    partial = [
        index[0].model_copy(
            update={"h_last": hashing.phash64(np.random.default_rng(3).uniform(0, 255, (32, 32)))}
        )
    ]
    assert checks.check_overlap(EP, make_ctx(EPS, benchmark_index=partial, reader=dup)).passed


def test_license() -> None:
    assert checks.check_license(EP, CTX).passed
    r = checks.check_license(EP.model_copy(update={"license": "CC-BY-NC-4.0"}), CTX)
    assert not r.passed and r.action == "drop_episode"


def _with_arm(ep: Episode, arm: Arm, data: object) -> Episode:
    arms = {**ep.force.arms, arm: data}
    return ep.model_copy(update={"force": ep.force.model_copy(update={"arms": arms})})


def test_saturation_counts_top_rail_only() -> None:
    data = EP.force.arms["L"]
    assert isinstance(data, PadArmForce)
    floored = [max(v, 0.9) for v in data.f_a]  # a long run at the minimum is healthy
    ok = _with_arm(EP, "L", data.model_copy(update={"f_a": floored}))
    assert checks.check_sensor_health(ok, CTX).passed
    top = [min(v, 0.5) for v in data.f_a]
    bad = _with_arm(EP, "L", data.model_copy(update={"f_a": top}))
    r = checks.check_sensor_health(bad, CTX)
    assert not r.passed and "saturated" in r.detail


def test_saturation_value_rail() -> None:
    data = EP.force.arms["L"]
    assert isinstance(data, PadArmForce)
    th = CTX.thresholds.model_copy()
    object.__setattr__(th, "saturation_value", 0.3)
    ctx = CTX.model_copy(update={"thresholds": th})
    r = checks.check_sensor_health(EP, ctx)
    assert not r.passed and "saturated" in r.detail


def test_drop_channel_names_arms() -> None:
    r = checks.check_sensor_health(_flat_pad(EP, "L", "f_a"), CTX)
    assert r.action == "drop_channel" and r.detail.startswith("arm=L;")
    both = _flat_pad(_flat_pad(EP, "L", "f_a"), "R", "f_b")
    r = checks.check_sensor_health(both, CTX)
    assert r.action == "drop_episode" and r.detail.startswith("arm=L,arm=R;")


def _flat_wrist(ep: Episode, arm: Arm, names: tuple[str, ...]) -> Episode:
    data = ep.force.arms[arm]
    assert isinstance(data, WristArmForce)
    return _with_arm(
        ep, arm, data.model_copy(update={n: [0.0] * len(getattr(data, n)) for n in names})
    )


def test_wrist_health() -> None:
    wr = make_wrist_episode(0, arms=("L", "R"))
    ctx = make_ctx(EPS)
    assert checks.check_sensor_health(wr, ctx).passed  # fy and torques are flat in the fixture
    one = _flat_wrist(wr, "L", ("fx", "fy", "fz"))
    r = checks.check_sensor_health(one, ctx)
    assert r.action == "drop_channel" and r.detail.startswith("arm=L;")
    both = _flat_wrist(one, "R", ("fx", "fy", "fz"))
    assert checks.check_sensor_health(both, ctx).action == "drop_episode"


def test_normalization_records_json_stats_unclipped() -> None:
    r = checks.check_normalization(EP, CTX)
    doc = json.loads(r.detail)
    assert set(doc) == {"grasp", "free", "p99", "problems"} and len(doc["p99"]) == 2
    assert abs(doc["free"][0]) < 0.1


def test_baxter_onset_lag_rule() -> None:
    # the fixture's force onset trails gripper closure (g >= 0.5) by 175 ms
    ctx = make_ctx(EPS, source="baxter")
    r = checks.check_sync(EP, ctx)
    assert not r.passed and r.action == "drop_episode" and "onset" in r.detail
    assert checks.check_sync(EP, CTX).passed  # other sources: xcorr rule only
    early = EP.model_copy(
        update={"force": EP.force.model_copy(update={"t": [x - 0.1 for x in EP.force.t]})}
    )
    r = checks.check_sync(early, ctx)
    assert r.passed and "onset L=+75ms" in r.detail


def test_spot_mp4_written(tmp_path: Path) -> None:
    items = [(e, outcome_window(e, CTX.norm)) for e in EPS]
    idx = checks.write_spot_strips("synthetic", items, SyntheticFrameReader(), tmp_path, n=2)
    doc = json.loads(idx.read_text())
    assert len(doc["videos"]) == 2 and all((tmp_path / v).stat().st_size > 0 for v in doc["videos"])


def test_prompt_diff_uses_tokenizer_json(tmp_path: Path) -> None:
    from tokenizers import Tokenizer, models

    path = tmp_path / "tok.json"
    Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(path))
    cfg = RenderConfig(
        **{**FORCE_CFG.render.model_dump(), "max_tokens": 10, "tokenizer_json": str(path)}
    )
    r = checks.check_prompt_diff(EP, make_ctx(EPS, render_cfg=cfg))
    assert r.passed and r.value == 1.0
