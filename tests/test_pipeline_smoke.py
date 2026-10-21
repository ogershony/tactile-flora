"""verify -> manifest -> windows -> one eval cell -> pseudo labels, all on synthetic fixtures."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tactile_flora import cli
from tactile_flora.eval import harness
from tactile_flora.eval.inference import StubBackend
from tactile_flora.labels.pseudo import PseudoRefused, run_pseudo
from tactile_flora.schema import Label
from tactile_flora.verify import manifest, run_manifest, run_verify
from tests.fixtures import eval_env as E
from tests.fixtures.synthetic_video import SyntheticFrameReader
from tests.fixtures.verify_ctx import FORCE_CFG

# the 64-line dense force block of the fixtures exceeds the production token budget
CFG = FORCE_CFG.model_copy(
    update={"render": FORCE_CFG.render.model_copy(update={"max_tokens": 99999})}
)


def test_pipeline_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    E.install(monkeypatch)
    reader = SyntheticFrameReader()
    s1 = run_verify(E.SOURCE_CFG, CFG, tmp_path, reader)
    assert (tmp_path / "verified/synthetic/norm.json").exists() and s1.n_episodes == 6
    s2 = run_verify(E.SOURCE_CFG, CFG, tmp_path, reader, spot_pass=12)
    assert s2.source_ok, s2.source_verdicts
    mpath = run_manifest(E.SOURCE_CFG, CFG, tmp_path, name="synth", report_version="rv")
    rows = manifest.read_manifest(mpath)
    assert {r.prompt_kind for r in rows} == {"dense", "outcome"}

    cfg = E.eval_cfg(mpath, tmp_path)
    cell = harness.run_eval(
        "released", "synth", "real", cfg, FORCE_CFG, tmp_path, StubBackend(), seed=0,
        source_cfgs={"synthetic": E.SOURCE_CFG},
    )  # fmt: skip
    preds = pd.read_parquet(cell / "predictions.parquet")
    assert len(preds) == len(rows) and not preds["parse_failed"].any()
    doc = json.loads((cell / "metrics.json").read_text())
    assert doc["manifest_hash"] == manifest.manifest_hash(mpath)
    assert "## synth" in harness.write_report(cfg, tmp_path).read_text()

    out = run_pseudo(E.SOURCE_CFG, tmp_path, StubBackend(), limit=2)
    pseudo = pd.read_parquet(out)
    assert out == tmp_path / "verified/synthetic/pseudo.parquet" and len(pseudo) == 2
    assert Label.model_validate_json(pseudo["label_json"][0]).origin == "pseudo"
    with pytest.raises(PseudoRefused):
        run_pseudo(E.SOURCE_CFG.model_copy(update={"eval_only": True}), tmp_path, StubBackend())


def test_cli_eval_run_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    E.install(monkeypatch)
    mpath = E.write_manifest(tmp_path)
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        f"output_dir: {tmp_path}/eval\nbenchmarks:\n  synth:\n    manifest: {mpath}\n"
        "    prompts: [outcome]\n    metrics: [mae]\n    score_only: true\n"
    )
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    (src_dir / "synthetic.yaml").write_text(
        "name: synthetic\nroot: x\nlicense_allowed: [CC0]\nmax_windows: 0\nmax_episodes: 0\n"
        "eval_only: false\nexpected_label_rate: 0.3\nseed: 1\n"
    )
    res = CliRunner().invoke(
        cli.app,
        ["eval", "run", "--config", str(cfg_path), "--benchmark", "synth", "--force-mode", "real",
         "--backend", "stub", "--data-root", str(tmp_path), "--sources-dir", str(src_dir)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "config:" in res.output and "git hash:" in res.output and "output:" in res.output
    assert (tmp_path / "eval/released/synth/real/predictions.parquet").exists()
    bad = CliRunner().invoke(
        cli.app,
        ["eval", "run", "--config", str(cfg_path), "--benchmark", "nope", "--backend", "stub"],
    )
    assert bad.exit_code != 0
