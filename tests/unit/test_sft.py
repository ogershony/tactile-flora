import json
from pathlib import Path
from types import SimpleNamespace
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


def test_distributed_batches_pad_with_zero_weight_and_preserve_the_true_mean() -> None:
    samples: list[Any] = list(range(5))
    groups = sft.ordered_groups(samples, 3, seed=0)
    assert [len(g) for g in groups] == [3, 2]
    per_rank = []
    for rank in range(2):
        backend: Any = SimpleNamespace(
            rank=rank, world_size=2, make_batch=lambda sample: {"value": sample}
        )
        per_rank.append(list(sft._batches(groups, backend, 1)))
    assert len(per_rank[0]) == len(per_rank[1]) == 3
    for start, stop, expected in ((0, 2, 1.0), (2, 3, 3.5)):
        actual = sum(
            b["value"] * b["loss_weight"] for rank in per_rank for b in rank[start:stop]
        ) / (2 * (stop - start))
        assert actual == pytest.approx(expected)
    assert per_rank[1][1]["loss_weight"] == 0


def test_partial_global_batch_stays_last_after_shuffle() -> None:
    samples: list[Any] = list(range(17))
    groups = sft.ordered_groups(samples, 4, seed=7)
    assert [len(g) for g in groups] == [4, 4, 4, 4, 1]
    assert {id(x) for g in groups for x in g} == {id(x) for x in samples}


@pytest.mark.parametrize("world_size", [1, 2])
def test_short_epoch_batch_has_its_own_update(world_size) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Embedding(VOCAB, 8), torch.nn.Linear(8, VOCAB))
    ids = torch.ones((1, 8), dtype=torch.long)
    labels = ids.clone()
    labels[:, :4] = -100
    backend: Any = SimpleNamespace(
        rank=0,
        world_size=world_size,
        make_batch=lambda sample: {"input_ids": ids, "labels": labels},
    )
    samples: list[Any] = list(range(25))
    groups = sft.ordered_groups(samples, 16, seed=17)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = sft.make_scheduler(optimizer, 8, 0.1)
    losses = sft.train_steps(
        _Wrap(model),
        sft._batches(groups, backend, 4),
        optimizer,
        scheduler,
        accum=16 // world_size,
        grad_clip=1.0,
    )
    assert len(losses) == 8
    assert scheduler.last_epoch == 8


def test_training_autocast_handles_mixed_stored_dtypes() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Embedding(VOCAB, 8).to(dtype=torch.bfloat16), torch.nn.Linear(8, VOCAB)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = sft.make_scheduler(optimizer, 2, 0.03)
    ids = torch.ones((1, 8), dtype=torch.long)
    labels = ids.clone()
    labels[:, :4] = -100
    losses = sft.train_steps(
        _Wrap(model),
        [{"input_ids": ids, "labels": labels}] * 2,
        optimizer,
        scheduler,
        accum=1,
        grad_clip=1,
        device="cpu",
        autocast_bf16=True,
    )
    assert len(losses) == 2 and all(torch.isfinite(torch.tensor(losses)))
    assert model[0].weight.dtype == torch.bfloat16 and model[1].weight.dtype == torch.float32


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


def test_early_stop_on_score_loss(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    model = nn.Sequential(nn.Embedding(VOCAB, 8), nn.Linear(8, VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    sch = sft.make_scheduler(opt, 20, 0.1)
    ids = torch.randint(0, VOCAB, (1, 12), generator=torch.Generator().manual_seed(0))
    labels = ids.clone()
    labels[:, :8] = -100
    batches = [{"input_ids": ids, "labels": labels}] * 20
    # A floor no finite loss can miss: training stops as soon as min_steps and the window allow.
    losses = sft.train_steps(
        _Wrap(model),
        batches,
        opt,
        sch,
        accum=1,
        grad_clip=1.0,
        run_dir=tmp_path,
        early_stop=(1e9, 3, 5),
    )
    assert len(losses) == 5
    rows = (tmp_path / "loss.csv").read_text().splitlines()
    assert rows[0] == "step,loss,lr,score_loss" and len(rows) == 6
    assert float(rows[1].split(",")[3]) > 0
    assert '"early_stop": 5' in (tmp_path / "log.jsonl").read_text()
    # Without the rule nothing changes.
    losses = sft.train_steps(_Wrap(model), batches, opt, sch, accum=1, grad_clip=1.0)
    assert len(losses) == 20


def test_score_loss_picks_the_last_digit_token_before_the_template_tail() -> None:
    torch = pytest.importorskip("torch")
    # answer: [template..., digit=5, im_end=90, newline=91]; digits are ids 1..5
    labels = torch.tensor([[-100, -100, 30, 31, 5, 90, 91]])
    logits = torch.zeros(1, 7, VOCAB)
    logits[0, 3, 5] = 8.0  # position 3 predicts position 4 (the digit) confidently
    logits[0, 4, 90] = 8.0
    with_ids = sft._score_loss(logits, labels, None, score_ids=[1, 2, 3, 4, 5])
    without = sft._score_loss(logits, labels, None)
    assert with_ids is not None and without is not None
    assert float(with_ids) < 0.05  # the digit itself (logit 8 against 96 zeros)
    assert float(without) < 0.05  # second-to-last token: im_end here, also confident
    logits[0, 3, 5] = 0.0
    unsure = sft._score_loss(logits, labels, None, score_ids=[1, 2, 3, 4, 5])
    assert unsure is not None and float(unsure) > 1.0
    assert sft._score_loss(logits, labels, None, score_ids=[7]) is None
