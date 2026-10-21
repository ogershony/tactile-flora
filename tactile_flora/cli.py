"""The ``tf`` command line: one subcommand per stage.

Owns: argument parsing, the per-command banner (config path, git hash, output path) and exit codes.
Must never: contain stage logic, read YAML outside load_config, or write outside data/.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, get_args

import typer

from tactile_flora.config import (
    BaxterConfig,
    EvalConfig,
    ForceConfig,
    ForceMode,
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
    from tactile_flora.video.readers import AutoReader

    cfg = load_config(config, SourceConfig)
    if cfg.name != source:
        raise typer.BadParameter(
            f"config is for {cfg.name!r}, not {source!r}", param_hint="--source"
        )
    fcfg = load_config(force_config, ForceConfig)
    out = data_root / "verified" / source

    def _entry() -> None:
        summary = run_verify(cfg, fcfg, data_root, AutoReader(Path(cfg.root)), spot_pass=spot_pass)
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


BackendOpt = Annotated[str, typer.Option("--backend", help="Inference backend: stub, hf or vllm.")]
SourcesDirOpt = Annotated[Path, typer.Option("--sources-dir", help="configs/sources directory.")]


@app.command()
def pseudo(
    config: ConfigOpt,
    source: Annotated[str, typer.Option(help="Source name.")],
    backend: BackendOpt = "hf",
    data_root: DataRootOpt = Path("data"),
    sources_dir: SourcesDirOpt = Path("configs/sources"),
    limit: Annotated[int | None, typer.Option(help="Cap on episodes (smoke runs).")] = None,
) -> None:
    """Vision-only FLORA pseudo-labeling (refuses eval-only sources)."""
    from tactile_flora.eval.inference import make_backend
    from tactile_flora.labels import pseudo as ps

    cfg = load_config(config, SourceConfig)
    if cfg.name != source:
        raise typer.BadParameter(
            f"config is for {cfg.name!r}, not {source!r}", param_hint="--source"
        )
    if cfg.eval_only:
        typer.echo(f"refusing: source {source} is eval_only", err=True)
        raise typer.Exit(code=1)
    out = data_root / "verified" / source / "pseudo.parquet"

    def _entry() -> None:
        from tactile_flora.video.readers import SourceRootReader

        be = make_backend(backend, "released", SourceRootReader(sources_dir))
        path = ps.run_pseudo(cfg, data_root, be, seed=cfg.seed, limit=limit)
        typer.echo(f"wrote {path}")

    _run(config, out, _entry)


@app.command()
def train(
    config: ConfigOpt,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Tokenize, print histogram, exit.")
    ] = False,
    force_config: ForceConfigOpt = Path("configs/force.yaml"),
    data_root: DataRootOpt = Path("data"),
    allow_stale_manifest: Annotated[
        bool,
        typer.Option(
            "--allow-stale-manifest",
            help="DEV ONLY: train although the manifest report_version != the current git hash.",
        ),
    ] = False,
) -> None:
    """LoRA SFT on the accepted manifest."""
    from tactile_flora.train import sft

    cfg = load_config(config, TrainConfig)
    fcfg = load_config(force_config, ForceConfig)
    _run(
        config,
        Path(cfg.output_dir) / config.stem,
        lambda: sft.train(
            cfg,
            fcfg,
            data_root,
            dry_run=dry_run,
            run_name=config.stem,
            allow_stale_manifest=allow_stale_manifest,
        ),
    )


def _eval_common(force_config: Path, sources_dir: Path) -> tuple[ForceConfig, Path]:
    return load_config(force_config, ForceConfig), sources_dir


@eval_app.command("run")
def eval_run(
    config: ConfigOpt,
    checkpoint: Annotated[str, typer.Option(help="Checkpoint path or 'released'.")] = "released",
    benchmark: Annotated[str, typer.Option(help="Benchmark name.")] = "phantom_val",
    force_mode: Annotated[str, typer.Option(help="Force mode.")] = "omitted",
    backend: BackendOpt = "hf",
    seed: Annotated[int, typer.Option(help="Seed for the shuffled ablation.")] = 0,
    force_config: ForceConfigOpt = Path("configs/force.yaml"),
    data_root: DataRootOpt = Path("data"),
    sources_dir: SourcesDirOpt = Path("configs/sources"),
) -> None:
    """Evaluate one checkpoint x benchmark x force-mode cell."""
    from tactile_flora.eval import harness
    from tactile_flora.eval.inference import make_backend

    cfg = load_config(config, EvalConfig)
    if benchmark not in cfg.benchmarks:
        raise typer.BadParameter(f"unknown benchmark {benchmark!r}", param_hint="--benchmark")
    if force_mode not in get_args(ForceMode):
        raise typer.BadParameter(f"unknown force mode {force_mode!r}", param_hint="--force-mode")
    fcfg, sdir = _eval_common(force_config, sources_dir)
    out = harness.cell_dir(data_root, checkpoint, benchmark, force_mode)

    def _entry() -> None:
        from tactile_flora.video.readers import SourceRootReader

        be = make_backend(backend, checkpoint, SourceRootReader(sdir))
        cell = harness.run_eval(
            checkpoint, benchmark, force_mode, cfg, fcfg, data_root, be,  # type: ignore[arg-type]
            seed=seed, sources_dir=sdir,
        )  # fmt: skip
        typer.echo(f"wrote {cell}")

    _run(config, out, _entry)


@eval_app.command("grid")
def eval_grid(
    config: ConfigOpt,
    eval_config: Annotated[
        Path, typer.Option("--eval-config", help="Path to configs/eval/benchmarks.yaml.")
    ] = Path("configs/eval/benchmarks.yaml"),
    backend: BackendOpt = "hf",
    force_config: ForceConfigOpt = Path("configs/force.yaml"),
    data_root: DataRootOpt = Path("data"),
    sources_dir: SourcesDirOpt = Path("configs/sources"),
) -> None:
    """Expand the full condition matrix, skipping cells whose predictions exist."""
    from tactile_flora.eval import harness
    from tactile_flora.eval.inference import make_backend

    cfg = load_config(config, GridConfig)
    ecfg = load_config(eval_config, EvalConfig)
    fcfg, sdir = _eval_common(force_config, sources_dir)

    def _entry() -> None:
        from tactile_flora.video.readers import SourceRootReader

        reader = SourceRootReader(sdir)
        cells = harness.run_grid(
            cfg, ecfg, fcfg, data_root, lambda ck: make_backend(backend, ck, reader),
            sources_dir=sdir,
        )  # fmt: skip
        typer.echo(f"ran {len(cells)} cells")

    _run(config, data_root / "eval", _entry)


@eval_app.command("report")
def eval_report(
    config: ConfigOpt,
    data_root: DataRootOpt = Path("data"),
) -> None:
    """Render markdown tables per benchmark."""
    from tactile_flora.eval import harness

    cfg = load_config(config, EvalConfig)

    def _entry() -> None:
        typer.echo(f"wrote {harness.write_report(cfg, data_root)}")

    _run(config, data_root / "eval" / "report.md", _entry)


@eval_app.command("make-roboreward")
def eval_make_roboreward(
    config: ConfigOpt,
    metadata: Annotated[Path, typer.Option(help="RoboReward test/metadata.jsonl.")],
    n: Annotated[int, typer.Option(help="Number of videos.")] = 500,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
    data_root: DataRootOpt = Path("data"),
) -> None:
    """Build the stratified roboreward_500 manifest and windows sidecar (outcome, no force)."""
    from tactile_flora.eval import harness, roboreward
    from tactile_flora.verify import full_git_hash

    cfg = load_config(config, EvalConfig)
    spec = cfg.benchmarks["roboreward_500"]
    out = harness._manifest_path(spec.manifest, data_root)  # pyright: ignore[reportPrivateUsage]

    def _entry() -> None:
        roboreward.make_roboreward(metadata, out, n=n, seed=seed, report_version=full_git_hash())
        typer.echo(f"wrote {out}")

    _run(config, out, _entry)


@baxter_app.command("calibrate")
def baxter_calibrate(config: ConfigOpt) -> None:
    """Fit per-pad weight-to-newton curves (operator presses known weights)."""
    from tactile_flora.baxter import calibrate as cal
    from tactile_flora.baxter.calibration import sensor_warning
    from tactile_flora.baxter.serial_protocol import open_serial

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.calibration_path)
    warning = sensor_warning(cfg)
    if warning:
        typer.echo(warning, err=True)

    def _entry() -> None:
        serial = open_serial(cfg.serial_port, cfg.baud)

        def _prompt(msg: str) -> None:
            input(f"{msg} [enter] ")

        cal.calibrate(cfg, serial, _prompt, out, echo=typer.echo)

    _run(config, out, _entry)


@baxter_app.command("record")
def baxter_record(
    config: ConfigOpt,
    notes: Annotated[str, typer.Option(help="Operator notes for the episode.")] = "",
    timeout_s: Annotated[float, typer.Option(help="Max episode length in seconds.")] = 120.0,
) -> None:
    """Record one Baxter episode directory (refuses on flat/saturated channels)."""
    from tactile_flora.baxter.calibration import CalibrationFile, sensor_warning
    from tactile_flora.baxter.ros import recorder
    from tactile_flora.baxter.ros.transport import RosTransport

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.output_dir) / "raw_episodes"
    warning = sensor_warning(cfg)
    if warning:
        typer.echo(warning, err=True)

    def _entry() -> None:
        version = CalibrationFile.load(cfg.calibration_path).version
        rec = recorder.Recorder(RosTransport("tf_recorder"), cfg, out, version)
        typer.echo(f"recorded {recorder.run_episode(rec, notes, timeout_s)}")

    _run(config, out, _entry)


@baxter_app.command("label")
def baxter_label(
    config: ConfigOpt,
    labeler: Annotated[str, typer.Option(help="Labeler name (writes labels/<name>.json).")] = "",
    merge: Annotated[
        list[str] | None, typer.Option("--merge", help="Two labeler names to merge.")
    ] = None,
) -> None:
    """Label recorded episodes from video only (never shows force); or merge two labelers."""
    from tactile_flora.baxter import label

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.output_dir)
    if merge is not None and len(merge) != 2:
        raise typer.BadParameter("--merge takes exactly two names", param_hint="--merge")
    if merge is None and not labeler:
        raise typer.BadParameter("give --labeler NAME or --merge A --merge B")

    def _entry() -> None:
        player = label.TerminalPlayer()
        if merge is not None:
            path = label.merge_labels(out, merge[0], merge[1], label.player_resolver(player))
        else:
            path = label.run_labeler(out, labeler, player)
        typer.echo(f"wrote {path}")

    _run(config, out, _entry)


@baxter_app.command("to-lerobot")
def baxter_to_lerobot(
    config: ConfigOpt,
    force_config: ForceConfigOpt = Path("configs/force.yaml"),
    episodes: Annotated[
        Path | None,
        typer.Option(help="Recorded episode directories (default output_dir/raw_episodes)."),
    ] = None,
) -> None:
    """Convert episode directories to a LeRobot v3 dataset at output_dir, then run tf verify."""
    from tactile_flora.baxter import to_lerobot

    cfg = load_config(config, BaxterConfig)
    fcfg = load_config(force_config, ForceConfig)
    out = Path(cfg.output_dir)
    src = episodes or out / "raw_episodes"

    def _entry() -> None:
        to_lerobot.convert(src, out, fcfg.resample)
        typer.echo(
            "next: tf verify --config configs/sources/baxter.yaml --source baxter (sync gate)"
        )

    _run(config, out, _entry)
