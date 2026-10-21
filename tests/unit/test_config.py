from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from tactile_flora.cli import app
from tactile_flora.config import (
    BaxterConfig,
    EvalConfig,
    ForceConfig,
    GridConfig,
    SourceConfig,
    TrainConfig,
    load_config,
)
from tactile_flora.sources.base import get_source, registered_sources

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SOURCES = [
    "phantom",
    "sharpadex",
    "rh20t",
    "reassemble",
    "airoa",
    "hrdexdb",
    "softvtbench",
    "baxter",
]


def test_every_source_config_loads() -> None:
    files = sorted((CONFIGS / "sources").glob("*.yaml"))
    assert sorted(f.stem for f in files) == sorted(SOURCES)
    for f in files:
        cfg = load_config(f, SourceConfig)
        assert cfg.name == f.stem


def test_other_configs_load() -> None:
    load_config(CONFIGS / "force.yaml", ForceConfig)
    load_config(CONFIGS / "eval" / "benchmarks.yaml", EvalConfig)
    load_config(CONFIGS / "eval" / "grid.yaml", GridConfig)
    load_config(CONFIGS / "baxter.yaml", BaxterConfig)
    load_config(CONFIGS / "train" / "force.yaml", TrainConfig)
    load_config(CONFIGS / "train" / "control.yaml", TrainConfig)


def test_force_and_control_differ_in_force_mode_only() -> None:
    a = yaml.safe_load((CONFIGS / "train" / "force.yaml").read_text())
    b = yaml.safe_load((CONFIGS / "train" / "control.yaml").read_text())
    assert a.keys() == b.keys()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"force_mode"}


def test_extra_keys_forbidden(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIGS / "sources" / "phantom.yaml").read_text())
    raw["typo"] = 1
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(p, SourceConfig)


def test_every_adapter_registered() -> None:
    assert registered_sources() == sorted(SOURCES)
    for name in SOURCES:
        assert get_source(name).name == name


def test_cli_help_lists_every_subcommand() -> None:
    runner = CliRunner()
    top = runner.invoke(app, ["--help"])
    assert top.exit_code == 0
    for cmd in ["verify", "manifest", "pseudo", "train", "eval", "baxter"]:
        assert cmd in top.output
    nested = {
        "manifest": ["freeze-holdout"],
        "eval": ["run", "grid", "report"],
        "baxter": ["calibrate", "record", "label", "to-lerobot"],
    }
    for group, subs in nested.items():
        out = runner.invoke(app, [group, "--help"]).output
        for sub in subs:
            assert sub in out
