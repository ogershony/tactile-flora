"""A portable model comparison must refuse mismatched truth across conditions."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from scripts.cluster import summarize_hrdexdb_expanded as summary


def completed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "BASE", tmp_path)
    plan = {
        "optimizer_steps_per_run": 2,
        "manifest_sha256": "trainhash",
        "validation_manifest_sha256": "val",
        "test_manifest_sha256": "test",
        "evaluation_conditions": {"released": ["omitted", "real"]},
    }
    (tmp_path / "training-plan.json").write_text(json.dumps(plan))
    for name in ["expanded-v2-force", "expanded-v2-control"]:
        directory = tmp_path / "runs" / name
        (directory / "adapter").mkdir(parents=True)
        (directory / "loss.csv").write_text("step,loss,lr\n1,2.0,0.01\n2,1.0,0.001\n")
        (directory / "manifest_hash.txt").write_text("trainhash\n")
        save_file(
            {"lora_B": np.ones((2, 2), dtype=np.float32)},
            directory / "adapter/adapter_model.safetensors",
        )
    monkeypatch.setattr(summary.manifest, "manifest_hash", lambda path: path.stem)
    monkeypatch.setattr(
        summary.manifest,
        "read_manifest",
        lambda path: [SimpleNamespace(episode_id="a"), SimpleNamespace(episode_id="b")],
    )
    for split in ["val", "test"]:
        results = []
        for mode in ["omitted", "real"]:
            path = tmp_path / "eval/released" / f"hrdexdb_aligned_{split}" / mode
            path.mkdir(parents=True)
            pred = pd.DataFrame(
                {
                    "episode_id": ["a", "b"],
                    "window_id": ["wa", "wb"],
                    "truth_target": ["Score: 5", "Score: 1"],
                    "expected_score": [5.0, 1.0],
                }
            )
            pred.to_parquet(path / "predictions.parquet")
            (path / "config.json").write_text("{}")
            results.append(
                {
                    "checkpoint": "released",
                    "force_mode": mode,
                    "manifest_hash": split,
                    "parse_failed": 0,
                    "metrics": {"accuracy": 1.0, "auroc": 1.0},
                }
            )
        (tmp_path / f"evaluation-hrdexdb_aligned_{split}-result.json").write_text(
            json.dumps({"plan": {}, "results": results})
        )


def test_summary_checks_and_exports_completed_comparison(tmp_path, monkeypatch):
    completed_run(tmp_path, monkeypatch)
    summary.main()
    result = json.loads((tmp_path / "final-result.json").read_text())
    assert len(result["training_artifacts"]) == 2
    assert result["evaluation"]["test"]["cells"][0]["failure_recall"] == 1.0
    assert "100.0%" in (tmp_path / "final-result.md").read_text()


def test_summary_refuses_different_truth_targets(tmp_path, monkeypatch):
    completed_run(tmp_path, monkeypatch)
    path = tmp_path / "eval/released/hrdexdb_aligned_val/real/predictions.parquet"
    pred = pd.read_parquet(path)
    pred.loc[0, "truth_target"] = "Score: 1"
    pred.to_parquet(path)
    with pytest.raises(ValueError, match="different truth targets"):
        summary.main()
    assert not (tmp_path / "final-result.json").exists()
