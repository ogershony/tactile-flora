from pathlib import Path

import pytest
from typer.testing import CliRunner

from tactile_flora import cli
from tactile_flora.config import SourceConfig
from tactile_flora.schema import Episode, Label, Window
from tactile_flora.sources import base
from tactile_flora.verify import checks, manifest, run_manifest, run_verify
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
