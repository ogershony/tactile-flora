import json
from pathlib import Path
from typing import Any

import pytest

from tactile_flora.config import LoraConfig, TrainConfig
from tactile_flora.train import sft
from tactile_flora.verify import manifest
from tests.fixtures.train_data import VOCAB, StubTokenizer, windows_source, write_manifest
from tests.fixtures.verify_ctx import FORCE_CFG


def _cfg(tmp_path: Path, manifest_path: Path, **over: Any) -> TrainConfig:
    base: dict[str, Any] = {
        "manifest": str(manifest_path),
        "base_model": "stub",
        "force_mode": "real",
        "seed": 0,
        "epochs": 1,
        "learning_rate": 1e-2,
        "warmup_frac": 0.03,
        "grad_clip": 1.0,
        "effective_batch": 2,
        "lora": LoraConfig(rank=2, alpha=4, target_modules=["q_proj"]),
        "output_dir": str(tmp_path / "runs"),
    }
    base.update(over)
    return TrainConfig(**base)


def _freeze(tmp_path: Path, mp: Path) -> None:
    manifest.freeze_holdout([mp], tmp_path / "manifest" / "holdout_ids.json")


def _run(tmp_path: Path, cfg: TrainConfig, **kw: Any) -> Path:
    return sft.train(
        cfg,
        FORCE_CFG,
        tmp_path,
        tokenizer=StubTokenizer(),
        windows_source=windows_source,
        report_version="rv",
        **kw,
    )


def test_refuses_without_holdout(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "manifest" / "train.parquet")
    with pytest.raises(sft.TrainRefused, match="holdout_ids.json"):
        _run(tmp_path, _cfg(tmp_path, mp), dry_run=True)


def test_refuses_stale_report_version(tmp_path: Path) -> None:
    mp = write_manifest(tmp_path / "manifest" / "train.parquet", report_version="old")
    _freeze(tmp_path, mp)
    with pytest.raises(sft.TrainRefused, match="report_version"):
        _run(tmp_path, _cfg(tmp_path, mp), dry_run=True)
    # the dev override lets it through
    run = _run(tmp_path, _cfg(tmp_path, mp), dry_run=True, allow_stale_manifest=True)
    assert (run / "dry_run.json").exists()


def test_dry_run_writes_histogram(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mp = write_manifest(tmp_path / "manifest" / "train.parquet")
    _freeze(tmp_path, mp)
    run = _run(tmp_path, _cfg(tmp_path, mp), dry_run=True, run_name="force")
    doc = json.loads((run / "dry_run.json").read_text())
    assert doc["n_samples"] == 3 and doc["histogram"]["max"] >= doc["histogram"]["min"] > 0
    assert doc["eta_s"] == 3 * sft.DEFAULT_S_PER_WINDOW
    assert "estimated wall-clock" in capsys.readouterr().out
    assert not (run / "adapter").exists()


def test_make_scheduler_and_groups() -> None:
    torch = pytest.importorskip("torch")
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sch = sft.make_scheduler(opt, 100, 0.03)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] < lrs[2] <= 1.0 and lrs[-1] < 0.01 and max(lrs) == pytest.approx(1.0, abs=0.01)


def test_smoke_twenty_steps(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    model = nn.Sequential(nn.Embedding(VOCAB, 8), nn.Linear(8, VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    sch = sft.make_scheduler(opt, 20, 0.1)
    g = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(20):
        ids = torch.randint(0, VOCAB, (1, 12), generator=g)
        labels = ids.clone()
        labels[:, :8] = -100
        batches.append({"input_ids": ids, "labels": labels})
    # one repeated batch so the loss must fall
    batches = [batches[0]] * 20
    losses = sft.train_steps(
        _Wrap(model), batches, opt, sch, accum=1, grad_clip=1.0, run_dir=tmp_path
    )
    assert len(losses) == 20 and losses[-1] < losses[0]
    assert (tmp_path / "loss.csv").read_text().count("\n") == 21
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


class _Wrap:
    """Callable wrapper exposing the train()/parameters() surface train_steps needs."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def train(self) -> None:
        self.model.train()

    def parameters(self) -> Any:
        return self.model.parameters()

    def __call__(self, input_ids: Any) -> Any:
        return self.model(input_ids)


def test_full_train_with_backend_factory(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    mp = write_manifest(tmp_path / "manifest" / "train.parquet")
    _freeze(tmp_path, mp)
    model = torch.nn.Sequential(torch.nn.Embedding(VOCAB, 8), torch.nn.Linear(8, VOCAB))

    class _Backend:
        def __init__(self, cfg: TrainConfig) -> None:
            self.model, self.device, self.world_size, self.rank = _Wrap(model), None, 1, 0

        def trainable_parameters(self) -> list[Any]:
            return list(model.parameters())

        def make_batch(self, sample: Any) -> dict[str, Any]:
            return {
                "input_ids": torch.tensor([sample.input_ids]),
                "labels": torch.tensor([sample.labels]),
            }

        def save_adapter(self, path: Path) -> None:
            path.mkdir(parents=True)
            torch.save(model.state_dict(), path / "adapter.pt")

    run = _run(tmp_path, _cfg(tmp_path, mp), dry_run=False, backend_factory=_Backend, run_name="r")
    for name in ("config.yaml", "manifest_hash.txt", "git_hash.txt", "loss.csv", "adapter"):
        assert (run / name).exists(), name
    assert (run / "git_hash.txt").read_text().strip() == "rv"
    assert (run / "loss.csv").read_text().count("\n") == 3  # header + 2 steps (3 samples, batch 2)
