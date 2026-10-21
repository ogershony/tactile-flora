"""Shared builder for the eval tests: fake adapter, verified norm.json and a manifest on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_flora.config import BenchmarkSpec, EvalConfig, SourceConfig
from tactile_flora.force import normalize
from tactile_flora.schema import Episode, Label, Span
from tactile_flora.sources import base
from tactile_flora.verify import manifest
from tactile_flora.verify.windowing import episode_windows
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.verify_ctx import FORCE_CFG, fit_consts

METRICS = ["mae", "macro_f1", "balanced_acc", "bad_action_f1", "failure_recall"]


def make_episodes() -> dict[str, Episode]:
    """Six pad episodes with an outcome and a dense label; the first two are failures."""
    eps: dict[str, Episode] = {}
    for s in range(6):
        ep = make_pad_episode(s, arms=("L",))
        bad = s < 2
        labels = [
            Label(kind="outcome", origin="source", outcome=1 if bad else 5),
            Label(
                kind="dense", origin="source",
                spans=[Span(t0=0.5, t1=1.5, quality=1, rationale="excessive force")] if bad else [],
            ),
        ]  # fmt: skip
        eps[ep.episode_id] = ep.model_copy(
            update={"labels": labels, "meta": {"object": f"o{s % 2}"}}
        )
    return eps


EPISODES = make_episodes()


class FakeAdapter:
    """Registered as source "synthetic" by ``install``."""

    name = "synthetic"

    def discover(self, root: Path) -> list[str]:
        return list(EPISODES)

    def load(self, episode_id: str) -> Episode:
        return EPISODES[episode_id]

    def splits(self) -> dict[str, list[str]]:
        return {"val": ["s4"]}

    def native_labels(self, episode_id: str) -> list[Label]:
        return []


SOURCE_CFG = SourceConfig(
    name="synthetic", root="x", license_allowed=["CC0"], max_windows=0, max_episodes=0,
    eval_only=False, expected_label_rate=0.33, seed=1,
)  # fmt: skip


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the fake adapter under "synthetic"."""
    monkeypatch.setitem(base._REGISTRY, "synthetic", FakeAdapter)  # pyright: ignore[reportPrivateUsage]


def write_manifest(data_root: Path, name: str = "synth") -> Path:
    """Write norm.json for the fake source and a manifest of every episode's windows."""
    consts = fit_consts(list(EPISODES.values()))
    normalize.save(consts, data_root / "verified" / "synthetic" / "norm.json")
    pairs = [(ep, episode_windows(ep, FORCE_CFG, consts, "val")) for ep in EPISODES.values()]
    path = data_root / "manifest" / f"{name}.parquet"
    manifest.build_manifest(
        SOURCE_CFG, pairs, {e: {} for e in EPISODES}, consts.version(), "rv", None, {}, path
    )
    return path


def eval_cfg(manifest_path: Path, data_root: Path, *, score_only: bool = False) -> EvalConfig:
    """EvalConfig with one benchmark "synth" over ``manifest_path``."""
    spec = BenchmarkSpec(
        manifest=str(manifest_path),
        prompts=["outcome"] if score_only else ["dense", "outcome"],
        metrics=METRICS,
        score_only=score_only,
    )
    return EvalConfig(benchmarks={"synth": spec}, output_dir=str(data_root / "eval"))
