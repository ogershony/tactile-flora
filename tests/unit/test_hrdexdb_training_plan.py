"""An active experiment may survive HEAD changes, but not changed learning inputs."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.cluster.run_hrdexdb_sft import planned_report_version
from tactile_flora.config import LoraConfig, TrainConfig


def make_plan(tmp_path: Path):
    data = tmp_path / "manifest.parquet"
    data.write_bytes(b"frozen manifest")
    core = tmp_path / "train_core.py"
    core.write_text("learning_code = 1\n")
    cfg = TrainConfig(
        manifest=str(data),
        base_model="stub",
        force_mode="real",
        seed=17,
        epochs=4,
        learning_rate=2e-5,
        warmup_frac=0.1,
        grad_clip=1.0,
        effective_batch=16,
        lora=LoraConfig(rank=32, alpha=64, target_modules=["q_proj"]),
        output_dir=str(tmp_path / "runs"),
    )
    plan = {
        "slurm_training_job": 123,
        "git_hash": "pinned-generation-version",
        "configs": {"force": cfg.model_dump()},
        "manifest_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "core_code_sha256": {str(core): hashlib.sha256(core.read_bytes()).hexdigest()},
    }
    (tmp_path / "training-plan.json").write_text(json.dumps(plan))
    return cfg, data, core


def test_only_the_planned_job_uses_the_frozen_generation_version(tmp_path, monkeypatch):
    cfg, _, _ = make_plan(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert planned_report_version(tmp_path, cfg) is None
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    assert planned_report_version(tmp_path, cfg) is None
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert planned_report_version(tmp_path, cfg) == "pinned-generation-version"


@pytest.mark.parametrize("changed", ["config", "manifest", "code"])
def test_active_plan_refuses_changed_learning_inputs(tmp_path, monkeypatch, changed):
    cfg, data, core = make_plan(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    if changed == "config":
        cfg = cfg.model_copy(update={"epochs": 8})
    elif changed == "manifest":
        data.write_bytes(b"different examples")
    else:
        core.write_text("learning_code = 2\n")
    with pytest.raises(ValueError):
        planned_report_version(tmp_path, cfg)
