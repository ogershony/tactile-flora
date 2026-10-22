"""Acceptance input geometry and failure records for the released-model smoke."""

import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cluster import smoke_inference
from tests.fixtures.synthetic import make_pad_episode


def test_rocm_triton_distribution_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(name: str) -> str:
        if name == "triton-rocm":
            return "3.7.1"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(smoke_inference.importlib.metadata, "version", version)
    assert smoke_inference.package_version("triton") == "3.7.1"
    assert smoke_inference.package_version("absent") == "unavailable"


def test_acceptance_windows_need_no_training_labels() -> None:
    ep = make_pad_episode(0).model_copy(update={"labels": []})
    dense, outcome = smoke_inference.windows_for(ep)
    assert dense.prompt_kind == "dense" and len(dense.frame_times) == 64
    assert dense.frame_times[0] == -2 and dense.frame_times[-1] == 13.75
    assert outcome.prompt_kind == "outcome" and 2 <= len(outcome.frame_times) <= 64
    assert min(outcome.frame_times) >= 0
    assert max(outcome.frame_times) <= ep.duration_s
    assert dense.target == outcome.target == ""


def test_insufficient_memory_replaces_stale_success_without_weight_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        hub.HfApi,
        "model_info",
        lambda *a, **k: SimpleNamespace(
            sha="pinned",
            siblings=[SimpleNamespace(rfilename="model.safetensors", size=55_000_000_000)],
        ),
    )

    def no_download(*args: object, **kwargs: object) -> None:
        pytest.fail("insufficient memory must be detected before downloading weights")

    monkeypatch.setattr(hub, "snapshot_download", no_download)
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"status": "passed", "gate_passed": True}))
    args = smoke_inference.make_parser().parse_args(["--output", str(tmp_path)])
    assert smoke_inference.run(args) == 2
    recorded = json.loads(result.read_text())
    assert recorded["status"] == "blocked" and recorded["gate_passed"] is False
    assert recorded["revision"] == "pinned" and recorded["cases"] == []
