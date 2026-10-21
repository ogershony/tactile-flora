"""The ``tf`` command line: one subcommand per stage.

Owns: argument parsing, the per-command banner (config path, git hash, output path) and exit codes.
Must never: contain stage logic, read YAML outside load_config, or write outside data/.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from tactile_flora.config import (
    BaxterConfig,
    EvalConfig,
    ForceConfig,
    GridConfig,
    SourceConfig,
    TrainConfig,
    load_config,
)

app = typer.Typer(help="tactile-flora: force-aware FLORA-Rewarder pipeline.", no_args_is_help=True)
manifest_app = typer.Typer(help="Build the accepted manifest.", invoke_without_command=True)
eval_app = typer.Typer(help="Evaluate checkpoints.", no_args_is_help=True)
baxter_app = typer.Typer(help="Baxter capture stack.", no_args_is_help=True)
app.add_typer(manifest_app, name="manifest")
app.add_typer(eval_app, name="eval")
app.add_typer(baxter_app, name="baxter")

ConfigOpt = Annotated[Path, typer.Option("--config", help="Path to the YAML config.")]
ForceConfigOpt = Annotated[Path, typer.Option("--force-config", help="Path to configs/force.yaml.")]
DataRootOpt = Annotated[Path, typer.Option("--data-root", help="Root of the data/ tree.")]


def git_hash() -> str:
    """Short git hash of the repo, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run(config: Path, output: Path, entry: Callable[[], object]) -> None:
    """Print the banner, then call the stage entry point; unimplemented stages exit non-zero."""
    typer.echo(f"config:   {config}")
    typer.echo(f"git hash: {git_hash()}")
    typer.echo(f"output:   {output}")
    try:
        entry()
    except NotImplementedError as exc:
        typer.echo(f"not implemented: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except typer.Exit:
        raise
    except (RuntimeError, FileExistsError) as exc:  # failed invariants: frozen splits, bad source
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def verify(
    config: ConfigOpt,
    source: Annotated[str, typer.Option(help="Source name.")],
    force_config: ForceConfigOpt = Path("configs/force.yaml"),
    data_root: DataRootOpt = Path("data"),
    spot_pass: Annotated[
        int | None, typer.Option("--spot-pass", help="Reviewer's pass count for the spot check.")
    ] = None,
) -> None:
    """Run every verification check over a source and write its report."""
    from tactile_flora.verify import run_verify
    from tactile_flora.verify.video import PyAVFrameReader

    cfg = load_config(config, SourceConfig)
    if cfg.name != source:
        raise typer.BadParameter(
            f"config is for {cfg.name!r}, not {source!r}", param_hint="--source"
        )
    fcfg = load_config(force_config, ForceConfig)
    out = data_root / "verified" / source

    def _entry() -> None:
        summary = run_verify(
            cfg, fcfg, data_root, PyAVFrameReader(Path(cfg.root)), spot_pass=spot_pass
        )
        for name, ok in summary.source_verdicts.items():
            typer.echo(f"{name}: {'pass' if ok else 'FAIL'}")
        n_ok = sum(summary.episode_ok.values())
        typer.echo(f"episodes: {n_ok}/{summary.n_episodes} pass; report {summary.report_path}")
        if not summary.source_ok:
            raise typer.Exit(code=1)

    _run(config, out, _entry)


@manifest_app.callback()
def manifest(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to the YAML config.")
    ] = None,
    name: Annotated[str, typer.Option(help="Manifest name.")] = "train",
    force_config: ForceConfigOpt = Path("configs/force.yaml"),
    data_root: DataRootOpt = Path("data"),
) -> None:
    """Turn passing episodes into manifest rows."""
    if ctx.invoked_subcommand is not None:
        return
    if config is None:
        raise typer.BadParameter("--config is required", param_hint="--config")
    from tactile_flora.verify import full_git_hash, run_manifest

    cfg = load_config(config, SourceConfig)
    fcfg = load_config(force_config, ForceConfig)
    out = data_root / "manifest" / f"{name}.parquet"
    _run(
        config,
        out,
        lambda: run_manifest(cfg, fcfg, data_root, name=name, report_version=full_git_hash()),
    )


@manifest_app.command("freeze-holdout")
def freeze_holdout(
    config: ConfigOpt,
    manifests: Annotated[
        list[Path] | None, typer.Option("--manifest", help="Manifest parquet(s) to freeze.")
    ] = None,
    data_root: DataRootOpt = Path("data"),
) -> None:
    """Write data/manifest/holdout_ids.json once; later runs never change frozen splits."""
    from tactile_flora.verify import manifest as mf

    load_config(config, SourceConfig)
    out = data_root / "manifest" / "holdout_ids.json"
    paths = manifests or sorted((data_root / "manifest").glob("*.parquet"))

    def _entry() -> None:
        doc = mf.freeze_holdout(paths, out, git_hash())
        typer.echo(f"froze {len(doc.splits)} episodes, {len(doc.held_out_ids)} held out")

    _run(config, out, _entry)


@app.command()
def pseudo(config: ConfigOpt, source: Annotated[str, typer.Option(help="Source name.")]) -> None:
    """Vision-only FLORA pseudo-labeling (refuses eval-only sources)."""
    from tactile_flora.labels import pseudo as ps

    cfg = load_config(config, SourceConfig)
    if cfg.eval_only:
        typer.echo(f"refusing: source {source} is eval_only", err=True)
        raise typer.Exit(code=1)
    _run(config, Path("data/verified") / source / "pseudo", lambda: ps.run_pseudo(source))


@app.command()
def train(
    config: ConfigOpt,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Tokenize, print histogram, exit.")
    ] = False,
) -> None:
    """LoRA SFT on the accepted manifest."""
    from tactile_flora.train import sft

    cfg = load_config(config, TrainConfig)
    _run(config, Path(cfg.output_dir) / config.stem, lambda: sft.run_sft(cfg, dry_run=dry_run))


@eval_app.command("run")
def eval_run(
    config: ConfigOpt,
    checkpoint: Annotated[str, typer.Option(help="Checkpoint path or 'released'.")] = "released",
    benchmark: Annotated[str, typer.Option(help="Benchmark name.")] = "phantom_val",
    force_mode: Annotated[str, typer.Option(help="Force mode.")] = "omitted",
) -> None:
    """Evaluate one checkpoint x benchmark x force-mode cell."""
    from tactile_flora.eval import harness

    cfg = load_config(config, EvalConfig)
    out = Path(cfg.output_dir) / checkpoint / benchmark / force_mode
    _run(config, out, lambda: harness.run_eval(checkpoint, benchmark, force_mode, out))


@eval_app.command("grid")
def eval_grid(config: ConfigOpt) -> None:
    """Expand the full condition matrix, skipping cells whose predictions exist."""
    from tactile_flora.eval import harness

    cfg = load_config(config, GridConfig)

    def _entry() -> None:
        _ = harness
        raise NotImplementedError("milestone 10")

    _run(config, Path("data/eval") / f"grid_{len(cfg.checkpoints)}_checkpoints", _entry)


@eval_app.command("report")
def eval_report(config: ConfigOpt) -> None:
    """Render markdown tables per benchmark."""
    cfg = load_config(config, EvalConfig)

    def _entry() -> None:
        raise NotImplementedError("milestone 10")

    _run(config, Path(cfg.output_dir) / "report.md", _entry)


@baxter_app.command("calibrate")
def baxter_calibrate(config: ConfigOpt) -> None:
    """Fit per-pad weight-to-newton curves."""
    from tactile_flora.baxter import calibrate as cal

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.calibration_path)
    _run(config, out, lambda: cal.calibrate(out))


@baxter_app.command("record")
def baxter_record(config: ConfigOpt) -> None:
    """Record one Baxter episode directory."""
    from tactile_flora.baxter.ros import recorder

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.output_dir)
    _run(config, out, lambda: recorder.record_episode(out))


@baxter_app.command("label")
def baxter_label(config: ConfigOpt) -> None:
    """Label recorded episodes from video only (never shows force)."""
    from tactile_flora.baxter import label

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.output_dir)
    _run(config, out, lambda: label.run_labeler(out))


@baxter_app.command("to-lerobot")
def baxter_to_lerobot(config: ConfigOpt) -> None:
    """Convert episode directories to a LeRobot v3 dataset."""
    from tactile_flora.baxter import to_lerobot

    cfg = load_config(config, BaxterConfig)
    out = Path("data/raw/baxter_lerobot")
    _run(config, out, lambda: to_lerobot.convert(Path(cfg.output_dir), out))
