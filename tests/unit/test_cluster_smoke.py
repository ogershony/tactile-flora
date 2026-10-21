"""Tests for the lightweight CLI of the optional cluster smoke tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.cluster import smoke_train


def test_cli_options_do_not_require_training_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """The help and argument parser remain usable without optional ML packages."""
    original_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"torch", "transformers", "peft"}:
            raise AssertionError(f"unexpected optional import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    args = smoke_train.make_parser().parse_args(
        ["--steps", "12", "--device", "cuda:0", "--output", "/tmp/cluster-smoke"]
    )
    assert args.steps == 12
    assert args.device == "cuda:0"
    assert str(args.output) == "/tmp/cluster-smoke"


@pytest.mark.parametrize("value", ["0", "1", "-1", "not-a-number"])
def test_steps_must_allow_loss_comparison(value: str) -> None:
    """Reject step counts that cannot compare initial and final loss."""
    with pytest.raises(argparse.ArgumentTypeError):
        smoke_train.minimum_two_steps(value)


def test_run_rejects_single_step_before_optional_imports(tmp_path: Path) -> None:
    """The callable API enforces the CLI's two-step minimum."""
    with pytest.raises(ValueError, match="at least two steps"):
        smoke_train.run(steps=1, output=tmp_path, device_name="cpu")


def test_workspace_smoke_adapter_is_readable() -> None:
    """The existing adapter artifact passes the same structural checks as new runs."""
    pytest.importorskip("safetensors")
    adapter = Path(__file__).parents[2] / "data/cluster/smoke/adapter"
    record = smoke_train._artifact_record(adapter)
    assert record["valid"] is True
    assert record["files"]["adapter_model.safetensors"]["tensor_count"] > 0
