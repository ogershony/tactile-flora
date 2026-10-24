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
    "reassemble_lerobot",
    "reassemble_seg",
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


def test_sync_gate_is_relaxed_only_where_measured() -> None:
    gates = {
        f.stem: load_config(f, SourceConfig).sync_median_s
        for f in (CONFIGS / "sources").glob("*.yaml")
    }
    # phantom: measured median 0.15 s (docs/deviations.md 3); reassemble_lerobot: 0.225 s;
    # reassemble_seg inherits reassemble_lerobot's checks and carries the same gates
    assert gates["phantom"] == 0.2 and gates["reassemble_lerobot"] == 0.25
    assert gates["reassemble_seg"] == 0.25
    relaxed = {"phantom", "reassemble_lerobot", "reassemble_seg"}
    assert {k: v for k, v in gates.items() if k not in relaxed} == {
        k: None for k in gates if k not in relaxed
    }
    code = load_config(CONFIGS / "force.code.yaml", ForceConfig)
    assert (code.normalize.grasp_g, code.normalize.min_group_frac) == (0.6, 0.25)
    p99 = {
        f.stem: load_config(f, SourceConfig).norm_p99_max
        for f in (CONFIGS / "sources").glob("*.yaml")
    }
    assert {k for k, v in p99.items() if v is not None} == {
        "reassemble",
        "reassemble_lerobot",
        "reassemble_seg",
    }
    assert p99["reassemble"] == 60 and p99["reassemble_lerobot"] == 60
    assert p99["reassemble_seg"] == 60


def test_other_configs_load() -> None:
    load_config(CONFIGS / "force.yaml", ForceConfig)
    load_config(CONFIGS / "force.code.yaml", ForceConfig)
    load_config(CONFIGS / "eval" / "benchmarks.yaml", EvalConfig)
    load_config(CONFIGS / "eval" / "grid.yaml", GridConfig)
    load_config(CONFIGS / "baxter.yaml", BaxterConfig)
    load_config(CONFIGS / "train" / "force.yaml", TrainConfig)
    load_config(CONFIGS / "train" / "control.yaml", TrainConfig)
    load_config(CONFIGS / "train" / "judge.yaml", TrainConfig)
    load_config(CONFIGS / "train" / "judge_control.yaml", TrainConfig)
    load_config(CONFIGS / "train" / "judge_pf.yaml", TrainConfig)
    load_config(CONFIGS / "train" / "judge_pf_control.yaml", TrainConfig)


def test_source_gate_overrides_are_validated() -> None:
    base = load_config(CONFIGS / "sources" / "reassemble_lerobot.yaml", SourceConfig)
    assert base.sync_median_s == 0.25 and base.norm_grasp_range == (0.8, 1.3)
    assert load_config(CONFIGS / "sources" / "phantom.yaml", SourceConfig).norm_grasp_range is None
    for bad in ((1.1, 1.3), (0.8, 0.9), (0.0, 1.2)):
        with pytest.raises(ValidationError):
            base.model_copy(update={"norm_grasp_range": bad}).model_validate(
                {**base.model_dump(), "norm_grasp_range": bad}
            )


def test_committed_grid_runs_released_with_real_and_shuffled() -> None:
    """The released checkpoint reads the force block (2026-10-22 probe), so its force rows run."""
    from tactile_flora.eval.harness import expand_grid

    grid = load_config(CONFIGS / "eval" / "grid.yaml", GridConfig)
    assert grid.checkpoint_modes["released"] == ["omitted", "real", "shuffled"]
    assert grid.checkpoint_modes["data/runs/control"] == ["omitted"]
    assert grid.checkpoint_modes["data/runs/judge_control"] == ["omitted"]
    assert "data/runs/judge" in grid.checkpoints and "data/runs/judge_control" in grid.checkpoints
    cells = expand_grid(grid, load_config(CONFIGS / "eval" / "benchmarks.yaml", EvalConfig))
    by = {}
    for ckpt, bench, mode in cells:
        by.setdefault((ckpt, bench), []).append(mode)
    for bench in ("phantom_val", "reassemble_test"):
        assert by[("released", bench)] == ["omitted", "real", "shuffled"]
        assert by[("data/runs/control", bench)] == ["omitted"]
        assert by[("data/runs/force", bench)] == [
            "omitted",
            "zero",
            "real",
            "shuffled",
            "video_blank",
        ]
    assert by[("released", "roboreward_500")] == ["omitted"]
    assert by[("data/runs/force", "roboreward_500")] == ["omitted"]
    # the judge benchmark: outcome prompts, omitted / real / shuffled; the control stays omitted
    seg = load_config(CONFIGS / "eval" / "benchmarks.yaml", EvalConfig).benchmarks[
        "reassemble_seg_test"
    ]
    assert seg.prompts == ["outcome"] and seg.manifest.endswith("reassemble_seg_test.parquet")
    assert seg.metrics == ["balanced_acc", "failure_recall", "macro_f1", "bad_action_f1", "mae"]
    assert by[("released", "reassemble_seg_test")] == ["omitted", "real", "shuffled"]
    assert by[("data/runs/judge", "reassemble_seg_test")] == ["omitted", "real", "shuffled"]
    assert by[("data/runs/judge_control", "reassemble_seg_test")] == ["omitted"]
    assert by[("data/runs/control", "reassemble_seg_test")] == ["omitted"]
    # held-out PHANTOM failures (docs/deviations.md 8): same shape as the segment benchmark;
    # the judge_pf pair is in the grid and its control is force-free
    pft = load_config(CONFIGS / "eval" / "benchmarks.yaml", EvalConfig).benchmarks[
        "phantom_fail_test"
    ]
    assert pft.prompts == ["outcome"] and pft.manifest == "data/manifest/phantom_fail_test.parquet"
    assert pft.metrics == seg.metrics and not pft.score_only
    assert grid.modes["phantom_fail_test"] == ["omitted", "real", "shuffled"]
    assert grid.checkpoint_modes["data/runs/judge_pf_control"] == ["omitted"]
    assert {"data/runs/judge_pf", "data/runs/judge_pf_control"} <= set(grid.checkpoints)
    for ckpt in ("released", "data/runs/judge", "data/runs/judge_pf"):
        assert by[(ckpt, "phantom_fail_test")] == ["omitted", "real", "shuffled"]
    for ckpt in ("data/runs/control", "data/runs/judge_control", "data/runs/judge_pf_control"):
        assert by[(ckpt, "phantom_fail_test")] == ["omitted"]


def test_force_code_config_differs_in_format_and_budget_only() -> None:
    dec = load_config(CONFIGS / "force.yaml", ForceConfig)
    code = load_config(CONFIGS / "force.code.yaml", ForceConfig)
    assert (dec.render.format, code.render.format) == ("decimal", "code")
    assert dec.resample == code.resample and dec.events == code.events
    assert dec.normalize == code.normalize and dec.normalize.grasp_g == 0.6
    diff = {k for k in dec.render.model_dump() if getattr(dec.render, k) != getattr(code.render, k)}
    # summary: the decimal summary lines are opt-in (HRDexDB v3, origin/main), the code summary is
    # on in force.code.yaml
    assert diff == {"format", "max_tokens", "summary"}
    assert (dec.render.summary, code.render.summary) == (False, True)
    assert code.render.max_tokens < dec.render.max_tokens
    assert code.render.sharp_drop == code.events.slip_df


def test_force_and_control_differ_in_force_mode_only() -> None:
    a = yaml.safe_load((CONFIGS / "train" / "force.yaml").read_text())
    b = yaml.safe_load((CONFIGS / "train" / "control.yaml").read_text())
    assert a.keys() == b.keys()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"force_mode"}
    assert (a["force_mode"], b["force_mode"]) == ("real", "omitted")


def test_judge_configs_differ_from_force_in_manifest_only() -> None:
    force = load_config(CONFIGS / "train" / "force.yaml", TrainConfig).model_dump()
    judge = load_config(CONFIGS / "train" / "judge.yaml", TrainConfig).model_dump()
    control = load_config(CONFIGS / "train" / "judge_control.yaml", TrainConfig).model_dump()
    assert force.keys() == judge.keys() == control.keys()
    # epochs was 3 at first; set to 1 on 2026-10-22 so one judge epoch (3.1 h) fits the Colab
    # night; failure_weight added 2026-10-22 after the weight-1.0 judge collapsed to "success"
    assert {k for k in force if force[k] != judge[k]} <= {"manifest", "epochs", "failure_weight"}
    assert judge["manifest"] == "data/manifest/train_judge.parquet" and judge["epochs"] >= 1
    assert {k for k in judge if judge[k] != control[k]} == {"force_mode"}
    assert (judge["force_mode"], control["force_mode"]) == ("real", "omitted")
    # 2.2 was set from 851/382 on 2026-10-22; the dry run counts 808 successes / 425 failures
    # (ratio 1.90) in train_judge.parquet. The force/control runs stay flat.
    assert judge["failure_weight"] == control["failure_weight"] == 2.2
    assert force["failure_weight"] == 1.0
    for name in ("force.yaml", "control.yaml"):
        assert "failure_weight" not in yaml.safe_load((CONFIGS / "train" / name).read_text())


def test_judge_pf_configs_differ_from_judge_in_manifest_and_weight_only() -> None:
    """The held-out-failure judge pair (docs/deviations.md 8) mirrors judge / judge_control."""
    judge = load_config(CONFIGS / "train" / "judge.yaml", TrainConfig).model_dump()
    pf = load_config(CONFIGS / "train" / "judge_pf.yaml", TrainConfig).model_dump()
    control = load_config(CONFIGS / "train" / "judge_pf_control.yaml", TrainConfig).model_dump()
    assert {k for k in judge if judge[k] != pf[k]} == {"manifest", "failure_weight"}
    assert pf["manifest"] == "data/manifest/train_judge_pf.parquet"
    # successes / failures of train_judge_pf.parquet: 788 / 405 (the 2026-10-22 dry run:
    # 1,233 - 40 rows; 425 - 20 failures), rounded to two decimals
    assert pf["failure_weight"] == round(788 / 405, 2) == 1.95
    assert {k for k in pf if pf[k] != control[k]} == {"force_mode"}
    assert (pf["force_mode"], control["force_mode"]) == ("real", "omitted")


def test_train_failure_weight_defaults_to_one_and_must_be_positive() -> None:
    raw = yaml.safe_load((CONFIGS / "train" / "force.yaml").read_text())
    assert TrainConfig.model_validate(raw).failure_weight == 1.0
    assert TrainConfig.model_validate({**raw, "failure_weight": 2.2}).failure_weight == 2.2
    for bad in (0, -1.0):
        with pytest.raises(ValidationError):
            TrainConfig.model_validate({**raw, "failure_weight": bad})


def test_segment_source_config_fields() -> None:
    cfg = load_config(CONFIGS / "sources" / "reassemble_seg.yaml", SourceConfig)
    base = load_config(CONFIGS / "sources" / "reassemble_lerobot.yaml", SourceConfig)
    assert cfg.root == base.root and cfg.license_allowed == base.license_allowed
    assert cfg.episode_unit == "segment" and base.episode_unit == "recording"
    assert cfg.verify_from == "reassemble_lerobot" and cfg.inherited_drops == "record"
    assert base.inherited_drops == "apply" and base.verify_from is None
    assert cfg.segment_select.max_total == {"test": 450}
    assert cfg.segment_select.success_multiple == 2 and cfg.segment_select.seed == 0
    assert cfg.max_windows == 0 and cfg.max_episodes == 0 and cfg.cameras is None
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({**cfg.model_dump(), "episode_unit": "clip"})
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({**cfg.model_dump(), "inherited_drops": "ignore"})
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {**cfg.model_dump(), "segment_select": {"min_duration_s": -1.0}}
        )


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


def test_reward_v2_configs_differ_from_v1_in_epochs_and_dense_weight_only() -> None:
    for name in ("force", "control"):
        v1 = load_config(CONFIGS / "train" / f"{name}.yaml", TrainConfig).model_dump()
        v2 = load_config(CONFIGS / "train" / f"{name}_v2.yaml", TrainConfig).model_dump()
        assert {k for k in v1 if v1[k] != v2[k]} == {"epochs", "dense_failure_weight"}
        # 279 all-normal / 78 non-normal dense windows in train.parquet (2026-10-22)
        assert v2["epochs"] == 3 and v2["dense_failure_weight"] == round(279 / 78, 2)
