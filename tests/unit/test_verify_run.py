import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tactile_flora import cli
from tactile_flora.config import SourceConfig
from tactile_flora.schema import Episode, Label, Window
from tactile_flora.sources import base
from tactile_flora.verify import (
    checks,
    manifest,
    record_spot_pass,
    report,
    run_manifest,
    run_verify,
)
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.synthetic_video import SyntheticFrameReader
from tests.fixtures.verify_ctx import FORCE_CFG, outcome_window

EPS = {f"pad-{s}": make_pad_episode(s) for s in range(6)}
EPS["pad-5"] = EPS["pad-5"].model_copy(update={"license": "GPL"})  # dropped by the license check
for k, e in EPS.items():
    lab = Label(kind="outcome", origin="source", outcome=1 if k in ("pad-0", "pad-1") else 5)
    EPS[k] = e.model_copy(update={"labels": [lab]})


class _Adapter:
    name = "synthetic_run"

    def discover(self, root: Path) -> list[str]:
        return list(EPS)

    def load(self, episode_id: str) -> Episode:
        return EPS[episode_id]

    def splits(self) -> dict[str, list[str]]:
        return {"val": ["s4"]}

    def native_labels(self, episode_id: str) -> list[Label]:
        return []


CFG = SourceConfig(
    name="synthetic_run", root="x", license_allowed=["CC0"], max_windows=0, max_episodes=0,
    eval_only=False, expected_label_rate=0.33, seed=1,
)  # fmt: skip


def _windows(ep: Episode, split: str) -> list[Window]:
    from tests.fixtures.verify_ctx import fit_consts

    return [outcome_window(ep, fit_consts(list(EPS.values())), split)]


def test_verify_spot_review_then_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(base._REGISTRY, "synthetic_run", _Adapter)  # pyright: ignore[reportPrivateUsage]
    reader = SyntheticFrameReader()
    s1 = run_verify(CFG, FORCE_CFG, tmp_path, reader, windows_fn=_windows)
    assert not s1.source_ok and s1.source_verdicts["spot_check"] is False
    assert s1.episode_ok["pad-5"] is False and s1.episode_ok["pad-0"] is True
    assert (tmp_path / "verified/synthetic_run/report.md").exists()
    assert (tmp_path / "verified/synthetic_run/norm.json").exists()
    assert len(list((tmp_path / "verified/synthetic_run/spot").glob("*.png"))) == 6
    with pytest.raises(manifest.SourceFailedError):
        run_manifest(CFG, FORCE_CFG, tmp_path, windows_fn=_windows)

    s2 = run_verify(CFG, FORCE_CFG, tmp_path, reader, windows_fn=_windows, spot_pass=6)
    assert s2.source_ok, s2.source_verdicts
    assert s2.sync_median_s is not None and s2.sync_median_s < 0.1
    path = run_manifest(CFG, FORCE_CFG, tmp_path, report_version="rv", windows_fn=_windows)
    rows = manifest.read_manifest(path)
    assert {r.episode_id for r in rows} == {"pad-0", "pad-1", "pad-2", "pad-3", "pad-4"}
    assert all(len(r.frame_hashes) == 3 and r.label_origin == "source" for r in rows)
    assert {r.episode_id: r.split for r in rows}["pad-4"] == "val"

    doc = manifest.freeze_holdout([path], tmp_path / "manifest/holdout_ids.json")
    assert doc.held_out_ids == {"synthetic/pad-4"}
    # a frozen val episode is not leakage when assigned val, so verify still passes it
    s3 = run_verify(CFG, FORCE_CFG, tmp_path, reader, windows_fn=_windows)
    assert s3.episode_ok["pad-4"]


class _BrokenReader(SyntheticFrameReader):
    """A reader whose video for pad-2 is missing (as five RH20T cfg5 scene files are)."""

    def _check(self, episode: Episode) -> None:
        if episode.episode_id == "pad-2":
            raise FileNotFoundError("pad-2 scene mp4 was not downloaded")

    def frame_count(self, episode: Episode, role: str) -> int:
        self._check(episode)
        return super().frame_count(episode, role)

    def read(self, episode: Episode, role: str, index: int):  # type: ignore[override]
        self._check(episode)
        return super().read(episode, role, index)

    def read_many(self, episode: Episode, role: str, indices):  # type: ignore[override]
        self._check(episode)
        return super().read_many(episode, role, indices)


def test_spot_strips_skip_episodes_that_failed_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(base._REGISTRY, "synthetic_run", _Adapter)  # pyright: ignore[reportPrivateUsage]
    s = run_verify(CFG, FORCE_CFG, tmp_path, _BrokenReader(), windows_fn=_windows)
    assert s.episode_ok["pad-2"] is False and "integrity" in s.episode_failures["pad-2"]
    assert s.episode_ok["pad-0"] is True
    index = json.loads((tmp_path / "verified/synthetic_run/spot/index.json").read_text())
    assert index["n_windows"] == len(EPS) - 1
    assert not any("pad-2" in f for f in index["files"])
    assert (tmp_path / "verified/synthetic_run/checks.parquet").exists()


def test_record_spot_pass_rescores_only_the_spot_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(base._REGISTRY, "synthetic_run", _Adapter)  # pyright: ignore[reportPrivateUsage]
    vdir = tmp_path / "verified/synthetic_run"
    with pytest.raises(RuntimeError, match="run tf verify first"):
        record_spot_pass(CFG, tmp_path, 6)
    s1 = run_verify(CFG, FORCE_CFG, tmp_path, SyntheticFrameReader(), windows_fn=_windows)
    assert not s1.source_ok
    before = report.read_checks(vdir)
    s2 = record_spot_pass(CFG, tmp_path, 6)  # 6 strips: need ceil(17/20 * 6) = 6
    assert s2.source_ok and s2.source_verdicts["spot_check"] and s2.n_episodes == len(EPS)
    assert s2.episode_ok["pad-5"] is False and s2.sync_median_s == s1.sync_median_s
    after = report.read_checks(vdir)
    for eid, per in after.items():
        assert per["spot_check"].passed and per["spot_check"].value == 6
        rest = {k: v for k, v in per.items() if k != "spot_check"}
        assert rest == {k: v for k, v in before[eid].items() if k != "spot_check"}
    assert "- spot_check: pass" in (vdir / "report.md").read_text()
    assert json.loads((vdir / "spot/index.json").read_text())["reviewer_pass"] == 6
    path = run_manifest(CFG, FORCE_CFG, tmp_path, report_version="rv", windows_fn=_windows)
    assert len(manifest.read_manifest(path)) == 5
    s3 = record_spot_pass(CFG, tmp_path, 2)
    assert not s3.source_ok and s3.source_verdicts["spot_check"] is False
    with pytest.raises(manifest.SourceFailedError):
        run_manifest(CFG, FORCE_CFG, tmp_path, windows_fn=_windows)


def test_cli_banner_and_missing_config(tmp_path: Path) -> None:
    res = CliRunner().invoke(
        cli.app,
        [
            "manifest",
            "freeze-holdout",
            "--config",
            "configs/sources/phantom.yaml",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0 and "config:" in res.output and "holdout_ids.json" in res.output
    again = CliRunner().invoke(
        cli.app,
        [
            "manifest",
            "freeze-holdout",
            "--config",
            "configs/sources/phantom.yaml",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert again.exit_code == 1 and "HoldoutExistsError" in again.output
    assert checks.CHECK_NAMES  # keep import used


def test_norm_fit_items_skips_excluded_and_rejects_orphan_groups() -> None:
    from tactile_flora.force import normalize
    from tactile_flora.force.resample import resample
    from tactile_flora.verify import norm_fit_items
    from tests.fixtures.verify_ctx import TIMES

    f = [resample(e.force, e.gripper, TIMES, FORCE_CFG.resample) for e in EPS.values()]
    kept = norm_fit_items([("a", f[0], False), ("a", f[1], True), ("b", f[2], False)])
    assert [(g, x is y) for (g, x), y in zip(kept, (f[0], f[2]), strict=True)] == [
        ("a", True),
        ("b", True),
    ]
    with pytest.raises(normalize.NormError, match="orphan|only episodes excluded"):
        norm_fit_items([("a", f[0], False), ("c", f[1], True)])


def test_excluded_episodes_do_not_feed_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tactile_flora.force import normalize
    from tactile_flora.schema import PadArmForce

    def _squeezed(ep: Episode, gain: float) -> Episode:
        arms = {}
        for a, d in ep.force.arms.items():
            assert isinstance(d, PadArmForce)
            arms[a] = PadArmForce(
                f_a=[0.5 + (v - 0.5) * gain for v in d.f_a],
                f_b=[0.5 + (v - 0.5) * gain for v in d.f_b],
            )
        return ep.model_copy(update={"force": ep.force.model_copy(update={"arms": arms})})

    # two "over-squeeze" episodes at 3x the hold force, flagged by the adapter
    eps = dict(EPS)
    for k in ("pad-0", "pad-1"):
        eps[k] = _squeezed(eps[k], 3.0).model_copy(update={"meta": {"exclude_from_norm_fit": True}})

    class _Flagged(_Adapter):
        name = "synthetic_flagged"

        def discover(self, root: Path) -> list[str]:
            return list(eps)

        def load(self, episode_id: str) -> Episode:
            return eps[episode_id]

    monkeypatch.setitem(base._REGISTRY, "synthetic_flagged", _Flagged)  # pyright: ignore[reportPrivateUsage]
    cfg = CFG.model_copy(update={"name": "synthetic_flagged"})
    s = run_verify(cfg, FORCE_CFG, tmp_path, SyntheticFrameReader(), windows_fn=_windows)
    assert s.norm_error is None and s.norm_fit_episodes == 4
    consts = normalize.load(tmp_path / "verified/synthetic_flagged/norm.json")
    assert abs(consts.arms["L"].grasp - 5.0) < 0.3  # the 13 N squeezes did not move the median
    # the flagged episodes are still normalized (against the typical grasp) and checked
    assert s.episode_ok["pad-0"] and "pad-0" in s.episode_failures or s.episode_ok["pad-0"]
    # a group made only of excluded episodes cannot be fitted: the source fails normalization
    eps["pad-0"] = eps["pad-0"].model_copy(
        update={
            "source": "phantom",
            "meta": {"task_folder": "x_fail", "exclude_from_norm_fit": True},
        }
    )
    s2 = run_verify(cfg, FORCE_CFG, tmp_path, SyntheticFrameReader(), windows_fn=_windows)
    assert s2.norm_error is not None and "x" in s2.norm_error
    assert s2.source_verdicts["normalization"] is False


def test_source_config_overrides_only_its_two_gates() -> None:
    from tactile_flora.schema import VerifyThresholds
    from tactile_flora.verify import _thresholds

    assert _thresholds(CFG) == VerifyThresholds()
    th = _thresholds(CFG.model_copy(update={"sync_median_s": 0.2, "norm_p99_max": 60.0}))
    assert (th.sync_median_s, th.norm_p99_max) == (0.2, 60.0)
    assert th.model_copy(update={"sync_median_s": 0.1, "norm_p99_max": 3.0}) == VerifyThresholds()
