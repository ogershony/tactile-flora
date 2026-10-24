import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tactile_flora.config import BenchmarkSpec, GridConfig
from tactile_flora.eval import harness, roboreward
from tactile_flora.eval.cache import cache_path
from tactile_flora.eval.inference import StubBackend
from tactile_flora.prompts.flora import VideoSpec
from tests.fixtures import eval_env as E
from tests.fixtures.verify_ctx import FORCE_CFG


class Recording(StubBackend):
    """Stub that remembers the prompts and video specs it was given."""

    def __init__(self) -> None:
        super().__init__(outcome_score=5)
        self.seen: list[tuple[str, str, VideoSpec]] = []

    def generate(self, messages: Any, video: VideoSpec, **kw: Any) -> Any:
        user = " ".join(c.get("text", "") for c in messages[1]["content"])
        self.seen.append((str(messages[0]["content"][0]["text"]), user, video))
        return super().generate(messages, video, **kw)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    E.install(monkeypatch)
    return tmp_path, E.write_manifest(tmp_path)


def run(env: tuple[Path, Path], mode: str, backend: Any = None, ckpt: str = "released") -> Path:
    root, mp = env
    return harness.run_eval(
        ckpt, "synth", mode, E.eval_cfg(mp, root), FORCE_CFG, root,  # type: ignore[arg-type]
        backend or StubBackend(), seed=3, source_cfgs={"synthetic": E.SOURCE_CFG},
    )  # fmt: skip


def test_run_eval_writes_predictions_and_metrics(env: tuple[Path, Path]) -> None:
    root, mp = env
    out = run(env, "omitted")
    assert out == root / "eval/released/synth/omitted"
    preds = pd.read_parquet(out / "predictions.parquet")
    assert len(preds) == 12 and set(preds["prompt_kind"]) == {"dense", "outcome"}
    assert not preds["parse_failed"].any()
    assert (
        preds.loc[preds["prompt_kind"] == "outcome", "expected_score"].eq(4.75).all()
    )  # 0.9 * 5 + 0.025 * 10
    doc = json.loads((out / "metrics.json").read_text())
    from tactile_flora.verify import manifest

    assert doc["manifest_hash"] == manifest.manifest_hash(mp) and doc["parse_failed"] == 0
    assert doc["git_hash"] and doc["config_hash"] and doc["n_episodes"] == 6
    m = doc["metrics"]
    # all-normal dense prediction misses every failure; outcome 5 vs truth 1 (2 eps) or 5 (4 eps)
    assert m["failure_recall"] == 0.0 and m["mae"] == pytest.approx(400 * 2 / 6)
    assert doc["per_object"]["o0"]["mae"] == pytest.approx(400 / 3)
    assert len((out / "log.jsonl").read_text().splitlines()) == 12


def test_force_modes_change_the_prompt(env: tuple[Path, Path]) -> None:
    seen: dict[str, list[tuple[str, str, VideoSpec]]] = {}
    for mode in ("omitted", "real", "zero", "shuffled", "video_blank"):
        be = Recording()
        run(env, mode, be)
        seen[mode] = be.seen
    assert all("Force (normalized" not in u for _, u, _ in seen["omitted"])
    for mode in ("real", "zero", "shuffled", "video_blank"):
        assert all("Force (normalized" in u and "fingertip" in s for s, u, _ in seen[mode])
    zero_lines = [ln for ln in seen["zero"][0][1].splitlines() if ln.startswith("t=")]
    assert zero_lines and all("F=0.00" in ln and "c=0" in ln for ln in zero_lines)
    assert [u for _, u, _ in seen["real"]] != [u for _, u, _ in seen["shuffled"]]
    assert [u for _, u, _ in seen["real"]] == [u for _, u, _ in seen["video_blank"]]
    assert all(v.blank for _, _, v in seen["video_blank"])
    assert not any(v.blank for _, _, v in seen["real"])


def test_score_only_mode_and_dense_budget(env: tuple[Path, Path]) -> None:
    root, mp = env
    be = Recording()
    harness.run_eval(
        "released", "synth", "omitted", E.eval_cfg(mp, root, score_only=True), FORCE_CFG, root,
        be, seed=0, source_cfgs={"synthetic": E.SOURCE_CFG},
    )  # fmt: skip
    assert len(be.seen) == 6 and all("Do not provide a rationale" in s for s, _, _ in be.seen)


def test_parse_failure_is_counted(env: tuple[Path, Path]) -> None:
    be = StubBackend(text_fn=lambda m, v: "garbage")
    out = run(env, "omitted", be)
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["parse_failed"] == 12
    # unparsable outcomes are replaced by score 3 and stay in MAE: |3 - 1| = |3 - 5| = 2
    assert doc["metrics"]["mae"] == pytest.approx(200.0)


def test_grid_skips_existing_and_report(env: tuple[Path, Path]) -> None:
    root, mp = env
    cfg = E.eval_cfg(mp, root)
    grid = GridConfig(
        checkpoints=["released", "data/runs/force"],
        modes={"synth": ["omitted", "real", "shuffled"]},
        seed=0,
        checkpoint_modes={"released": ["omitted"]},
    )
    calls: list[str] = []

    def factory(ck: str) -> StubBackend:
        calls.append(ck)
        return StubBackend()

    kw: dict[str, Any] = {"source_cfgs": {"synthetic": E.SOURCE_CFG}}
    cells = harness.run_grid(grid, cfg, FORCE_CFG, root, factory, **kw)
    assert sorted(c.relative_to(root / "eval").as_posix() for c in cells) == [
        "force/synth/omitted", "force/synth/real", "force/synth/shuffled", "released/synth/omitted",
    ]  # fmt: skip
    assert harness.run_grid(grid, cfg, FORCE_CFG, root, factory, **kw) == []
    assert calls == ["released", "data/runs/force"]
    shutil.rmtree(root / "eval/force/synth/real")
    assert len(harness.run_grid(grid, cfg, FORCE_CFG, root, factory, **kw)) == 1

    report = harness.write_report(cfg, root).read_text()
    assert "## synth" in report and "| force / real |" in report
    assert "shuffle delta (failure_recall)" in report
    real_row = next(ln for ln in report.splitlines() if ln.startswith("| force / real"))
    assert real_row.rstrip().endswith("+0.000 | - |")  # stub ignores force, so real == shuffled
    assert "omitted_minus_control" in report

    f = root / "eval/force/synth/shuffled/metrics.json"
    doc = json.loads(f.read_text())
    doc["manifest_hash"] = "0" * 64
    f.write_text(json.dumps(doc))
    with pytest.raises(harness.ManifestMismatchError):
        harness.write_report(cfg, root)


def test_roboreward_accepts_only_omitted(tmp_path: Path) -> None:
    spec = BenchmarkSpec(manifest=str(tmp_path / "rr.parquet"), prompts=["outcome"],
                         metrics=["mae"], score_only=True)  # fmt: skip
    cfg = E.eval_cfg(tmp_path / "x", tmp_path).model_copy(
        update={"benchmarks": {"roboreward_500": spec}}
    )
    with pytest.raises(ValueError, match="only force mode 'omitted'"):
        harness.run_eval("released", "roboreward_500", "real", cfg, FORCE_CFG, tmp_path,
                         StubBackend(), seed=0)  # fmt: skip
    grid = GridConfig(
        checkpoints=["released"], modes={"roboreward_500": ["omitted", "zero"]}, seed=0
    )
    with pytest.raises(ValueError, match="only force mode 'omitted'"):
        harness.expand_grid(grid, cfg)


def test_make_roboreward_and_eval(tmp_path: Path) -> None:
    meta = tmp_path / "test" / "metadata.jsonl"
    meta.parent.mkdir()
    rows = [
        {
            "file_name": f"v{i:03d}.mp4",
            "task": f"task {i} failure",
            "reward": 1 + (i % 5) if i % 3 else 5,
        }
        for i in range(60)
    ]
    meta.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "manifest" / "rr.parquet"
    kw: dict[str, Any] = {"n": 20, "seed": 4, "probe": lambda p: (10.0, 30)}
    roboreward.make_roboreward(meta, out, **kw)
    first = out.read_bytes()
    roboreward.make_roboreward(meta, out, **kw)
    assert out.read_bytes() == first and cache_path(out).exists()
    df = pd.read_parquet(cache_path(out))
    assert len(df) == 20 and set(df["target"]) == {f"Score: {k}" for k in range(1, 6)}
    assert df["task"].str.contains("failure").sum() == 0  # outcome words stripped
    alloc = roboreward.allocate({1: 10, 2: 10, 3: 70, 4: 5, 5: 5}, 20)
    assert sum(alloc.values()) == 20 and all(v >= 1 for v in alloc.values())

    spec = BenchmarkSpec(manifest=str(out), prompts=["outcome"], metrics=["mae"], score_only=True)
    cfg = E.eval_cfg(out, tmp_path).model_copy(update={"benchmarks": {"roboreward_500": spec}})
    res = harness.run_eval("released", "roboreward_500", "omitted", cfg, FORCE_CFG, tmp_path,
                           StubBackend(outcome_score=3), seed=0)  # fmt: skip
    doc = json.loads((res / "metrics.json").read_text())
    truth = df["target"].str.removeprefix("Score: ").astype(int)
    assert doc["metrics"]["mae"] == pytest.approx((truth - 3).abs().mean() * 100)
    assert doc["n_windows"] == 20


def test_checkpoint_tag_adapter_and_collision() -> None:
    assert harness.checkpoint_tag("data/runs/force/adapter") == "force"
    assert harness.checkpoint_tag("data/runs/control") == "control"
    assert harness.checkpoint_tag("released") == "released"
    cfg = E.eval_cfg(Path("x"), Path("y"))
    grid = GridConfig(
        checkpoints=["a/force/adapter", "b/force"], modes={"synth": ["omitted"]}, seed=0
    )
    with pytest.raises(harness.CheckpointTagCollisionError):
        harness.expand_grid(grid, cfg)


def test_grid_reruns_stale_cells(env: tuple[Path, Path]) -> None:
    root, mp = env
    cfg = E.eval_cfg(mp, root)
    grid = GridConfig(checkpoints=["released"], modes={"synth": ["omitted"]}, seed=0)
    kw: dict[str, Any] = {"source_cfgs": {"synthetic": E.SOURCE_CFG}}

    def run_g(c: Any, g: GridConfig = grid) -> list[Path]:
        return harness.run_grid(g, c, FORCE_CFG, root, lambda ck: StubBackend(), **kw)

    assert len(run_g(cfg)) == 1 and run_g(cfg) == []
    cell = root / "eval/released/synth/omitted"
    (cell / "metrics.json").unlink()  # predictions exist but the last file does not
    assert len(run_g(cfg)) == 1
    assert run_g(cfg) == []
    assert len(run_g(cfg, grid.model_copy(update={"seed": 9}))) == 1  # config hash changed
    doc = json.loads((cell / "metrics.json").read_text())
    doc["manifest_hash"] = "0" * 64
    (cell / "metrics.json").write_text(json.dumps(doc))
    assert len(run_g(cfg, grid.model_copy(update={"seed": 9}))) == 1  # manifest hash changed


def test_missing_logprobs_not_silently_integer(env: tuple[Path, Path]) -> None:
    class NoLp(StubBackend):
        def generate(self, messages: Any, video: VideoSpec, **kw: Any) -> Any:
            return (
                super()
                .generate(messages, video, **kw)
                .model_copy(update={"score_token_logprobs": None})
            )

    out = run(env, "omitted", NoLp(outcome_score=5))
    preds = pd.read_parquet(out / "predictions.parquet")
    assert preds.loc[preds["prompt_kind"] == "outcome", "expected_score"].isna().all()
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["no_logprobs"] == 6 and doc["warnings"]


def test_dense_frames_keyed_by_source_and_episode(env: tuple[Path, Path]) -> None:
    out = run(env, "omitted")
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["n_episodes"] == 6


def test_provider_called_once(env: tuple[Path, Path]) -> None:
    root, mp = env
    inner = harness.make_provider(mp, {"dense", "outcome"}, FORCE_CFG, Path("configs/sources"),
                                  {"synthetic": E.SOURCE_CFG})  # fmt: skip
    calls: list[int] = []

    def counting() -> Any:
        calls.append(1)
        return inner()

    harness.run_eval("released", "synth", "shuffled", E.eval_cfg(mp, root), FORCE_CFG, root,
                     StubBackend(), seed=1, provider=counting)  # fmt: skip
    assert len(calls) == 1


def test_grid_closes_previous_backend_when_checkpoint_changes(env: tuple[Path, Path]) -> None:
    root, mp = env
    cfg = E.eval_cfg(mp, root)
    grid = GridConfig(
        checkpoints=["released", "data/runs/force"], modes={"synth": ["omitted", "real"]}, seed=0
    )
    events: list[str] = []

    class Closing(StubBackend):
        def __init__(self, ck: str) -> None:
            super().__init__()
            self.ck = ck

        def close(self) -> None:
            events.append(f"close {self.ck}")

    def factory(ck: str) -> StubBackend:
        events.append(f"open {ck}")
        return Closing(ck)

    kw: dict[str, Any] = {"source_cfgs": {"synthetic": E.SOURCE_CFG}}
    cells = harness.run_grid(grid, cfg, FORCE_CFG, root, factory, **kw)
    assert len(cells) == 4  # two modes per checkpoint, no caps
    assert events == ["open released", "close released", "open data/runs/force"]
