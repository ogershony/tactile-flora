"""End-to-end: real adapters + fixtures through run_verify and run_manifest give real windows."""

from __future__ import annotations

from pathlib import Path

from tactile_flora.config import ForceConfig, SourceConfig, load_config
from tactile_flora.verify import manifest, run_manifest, run_verify
from tactile_flora.video.readers import AutoReader
from tests.fixtures.phantom import make_fixture as phantom_fx
from tests.fixtures.reassemble import make_fixture as reassemble_fx
from tests.fixtures.rh20t import make_fixture as rh20t_fx

FORCE_CFG = load_config("configs/force.yaml", ForceConfig)


def _cfg(name: str, root: Path, lic: str) -> SourceConfig:
    return SourceConfig(
        name=name, root=str(root), license_allowed=[lic], max_windows=0, max_episodes=0,
        eval_only=False, expected_label_rate=0.3, seed=0,
    )  # fmt: skip


def _run(cfg: SourceConfig, data: Path) -> list[manifest.ManifestRow]:
    reader = AutoReader(".")
    first = run_verify(cfg, FORCE_CFG, data, reader)
    # spot strips are sampled by the first run; record a passing human review of all of them
    n = len(list((data / "verified" / cfg.name / "spot").glob("*.png")))
    summary = run_verify(cfg, FORCE_CFG, data, reader, spot_pass=n)
    assert summary.source_ok, (first.episode_failures, summary.source_verdicts)
    path = run_manifest(cfg, FORCE_CFG, data, name="t", report_version="rv")
    return manifest.read_manifest(path)


def _kinds(rows: list[manifest.ManifestRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.prompt_kind] = out.get(r.prompt_kind, 0) + 1
    return out


def test_phantom_manifest_has_outcome_and_dense_windows(tmp_path: Path) -> None:
    root = tmp_path / "phantom"
    phantom_fx.write_long(root)
    rows = _run(_cfg("phantom", root, "CC-BY-4.0"), tmp_path / "data")
    kinds = _kinds(rows)
    assert kinds.get("outcome") == 2 and kinds.get("dense", 0) >= 2
    assert {r.label_origin for r in rows if r.prompt_kind == "outcome"} == {"source"}
    assert {r.label_origin for r in rows if r.prompt_kind == "dense"} <= {"source", "programmatic"}


def test_rh20t_manifest_has_outcome_and_dense_windows(tmp_path: Path) -> None:
    root = tmp_path / "rh20t"
    rh20t_fx.write_long(root)
    rows = _run(_cfg("rh20t", root, "CC-BY-NC-4.0"), tmp_path / "data")
    kinds = _kinds(rows)
    assert kinds.get("outcome") == 1 and kinds.get("dense", 0) >= 1


def test_rh20t_unrated_episode_has_no_outcome_window(tmp_path: Path) -> None:
    root = tmp_path / "rh20t"
    rh20t_fx.write_long(root, rating=-1)
    rows = _run(_cfg("rh20t", root, "CC-BY-NC-4.0"), tmp_path / "data")
    assert _kinds(rows).get("outcome", 0) == 0 and _kinds(rows).get("dense", 0) >= 1


def test_reassemble_manifest_has_outcome_and_dense_windows(tmp_path: Path) -> None:
    root = tmp_path / "reassemble"
    reassemble_fx.write_long(root)
    rows = _run(_cfg("reassemble", root, "CC-BY-4.0"), tmp_path / "data")
    kinds = _kinds(rows)
    assert kinds.get("outcome") == 1 and kinds.get("dense", 0) >= 1


def test_airoa_and_sharpadex_fixtures_assemble_labels() -> None:
    from tactile_flora.sources.airoa import AiroaAdapter
    from tactile_flora.sources.sharpadex import SharpadexAdapter
    from tactile_flora.verify.windowing import SourceInputs, assemble_labels

    fx = Path(__file__).parents[1] / "fixtures"
    for name, cls, eid in (
        ("airoa", AiroaAdapter, "0"),
        ("sharpadex", SharpadexAdapter, "pick_cup/season_7_train/0"),
    ):
        cfg = _cfg(name, fx / name, "CC-BY-4.0")
        ad = cls(cfg)
        ad.discover(fx / name)
        labels = assemble_labels(ad.load(eid), FORCE_CFG, None, SourceInputs(adapter=ad))
        assert all(lab.origin == "source" for lab in labels)
