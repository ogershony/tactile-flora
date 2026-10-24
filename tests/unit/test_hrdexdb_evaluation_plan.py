"""Queued arm evaluations must resolve their own frozen training plan before GPU loading."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts.cluster import evaluate_hrdexdb_clean as evaluation
from tactile_flora.config import ForceConfig, load_config


def test_plan_resolution(tmp_path: Path) -> None:
    shared = tmp_path / "training-plan.json"
    arm = tmp_path / "training-plan-chart.json"
    assert evaluation.training_plan_path(tmp_path, "chart") == shared
    arm.write_text("{}")
    assert evaluation.training_plan_path(tmp_path, "chart") == arm
    explicit = tmp_path / "chosen.json"
    assert evaluation.training_plan_path(tmp_path, "chart", explicit) == explicit


def test_force_worker_uses_arm_plan_and_only_its_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config").mkdir()
    arm = "chart"
    adapter = tmp_path / "runs" / arm / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"mock adapter")
    plan = tmp_path / f"training-plan-{arm}.json"
    plan.write_text(json.dumps({"base_model": "pinned-checkpoint"}))
    benchmark = "hrdexdb_aligned_val"
    cfg = tmp_path / "config/eval.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "benchmarks": {
                    benchmark: {
                        "manifest": "val.parquet",
                        "metrics": ["accuracy"],
                        "prompts": ["outcome"],
                    }
                },
                "output_dir": str(tmp_path / "eval"),
            }
        )
    )
    force_cfg = load_config("configs/force.yaml", ForceConfig)
    force_path = tmp_path / "config/force.yaml"
    force_path.write_text(yaml.safe_dump(force_cfg.model_dump()))
    monkeypatch.setattr(evaluation.manifest, "manifest_hash", lambda _: "manifest-hash")
    calls = []

    def backend(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs["model_id"]))
        return object()

    output = tmp_path / "cell"
    output.mkdir()
    (output / "metrics.json").write_text("{}")
    monkeypatch.setattr(evaluation, "HFBackend", backend)
    monkeypatch.setattr(evaluation, "run_eval", lambda *a, **kw: output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_hrdexdb_clean.py",
            "--base",
            str(tmp_path),
            "--eval-config",
            str(cfg),
            "--benchmark",
            benchmark,
            "--force-config",
            str(force_path),
            "--force-run",
            arm,
            "--force-modes",
            "chart_text,chart",
            "--worker",
            "2",
        ],
    )
    evaluation.main()
    assert calls == [(str(adapter), "pinned-checkpoint")]
    receipt = json.loads((tmp_path / f"evaluation-{benchmark}-{arm}-plan.json").read_text())
    assert receipt["training_plan_path"] == str(plan)
    assert receipt["training_plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert not (tmp_path / f"evaluation-{benchmark}-plan.json").exists()
