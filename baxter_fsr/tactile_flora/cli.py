"""The ``tf`` command line: one subcommand per stage.

Owns: argument parsing, the per-command banner (config path, git hash, output path) and exit codes.
Must never: contain stage logic, read YAML outside load_config, or write outside data/.
"""

from __future__ import annotations

import contextlib
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
    record_only: Annotated[
        bool,
        typer.Option(
            "--record-only",
            help="With --spot-pass: record the count and re-score only the spot gate from the "
            "saved checks (no video is decoded, no other check re-run).",
        ),
    ] = False,
) -> None:
    """Run every verification check over a source and write its report."""
    from tactile_flora.verify import record_spot_pass, run_verify
    from tactile_flora.video.readers import AutoReader

    cfg = load_config(config, SourceConfig)
    if cfg.name != source:
        raise typer.BadParameter(
            f"config is for {cfg.name!r}, not {source!r}", param_hint="--source"
        )
    if record_only and spot_pass is None:
        raise typer.BadParameter("--record-only needs --spot-pass N", param_hint="--record-only")
    fcfg = load_config(force_config, ForceConfig)
    out = data_root / "verified" / source

    def _entry() -> None:
        if record_only and spot_pass is not None:
            summary = record_spot_pass(cfg, data_root, spot_pass)
        else:
            summary = run_verify(cfg, fcfg, data_root, AutoReader("."), spot_pass=spot_pass)
        for name, ok in summary.source_verdicts.items():
            typer.echo(f"{name}: {'pass' if ok else 'FAIL'}")
        n_ok = sum(summary.episode_ok.values())
        typer.echo(f"episodes: {n_ok}/{summary.n_episodes} pass; report {summary.report_path}")
        if summary.norm_error:
            typer.echo(f"normalization fit: {summary.norm_error}")
        elif record_only:
            typer.echo("normalization fit: unchanged (record-only)")
        else:
            typer.echo(
                f"normalization fit: {summary.norm_fit_episodes}/{summary.n_episodes} episodes"
            )
            if summary.norm_path:
                from tactile_flora.force import normalize
                from tactile_flora.verify.report import norm_lines

                consts = normalize.load(summary.norm_path)
                for line in norm_lines(consts, cfg.norm_p99_max, cfg.norm_grasp_range)[2:]:
                    typer.echo(line)
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


@manifest_app.command("merge")
def manifest_merge(
    inputs: Annotated[
        list[str],
        typer.Argument(help="Input manifests: names under data/manifest/ or parquet paths."),
    ],
    out: Annotated[str, typer.Option("--out", help="Output name: data/manifest/<out>.parquet.")],
    split: Annotated[
        list[str] | None,
        typer.Option("--split", help="Keep only these splits (repeatable); default: every split."),
    ] = None,
    prompt_kind: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-kind", help="Keep only these prompt kinds (repeatable); default: both."
        ),
    ] = None,
    data_root: DataRootOpt = Path("data"),
) -> None:
    """Concatenate per-source manifests (rows unchanged; per-source caps already applied)."""
    from collections import Counter

    from tactile_flora.verify import manifest as mf

    paths = [
        p if (p := Path(i)).suffix == ".parquet" else data_root / "manifest" / f"{i}.parquet"
        for i in inputs
    ]
    out_path = data_root / "manifest" / f"{out}.parquet"
    typer.echo("inputs:   " + " ".join(str(p) for p in paths))
    typer.echo(f"git hash: {git_hash()}")
    typer.echo(f"output:   {out_path}")
    try:
        rows = mf.merge_manifests(paths, out_path, splits=split, prompt_kinds=prompt_kind)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    counts = Counter((r.source, r.split, r.prompt_kind) for r in rows)
    for (src, sp, kind), n in sorted(counts.items()):
        typer.echo(f"{src} {sp} {kind}: {n}")
    typer.echo(f"rows: {len(rows)}")


BackendOpt = Annotated[str, typer.Option("--backend", help="Inference backend: stub, hf or vllm.")]
SourcesDirOpt = Annotated[Path, typer.Option("--sources-dir", help="configs/sources directory.")]
ModelDirOpt = Annotated[
    str | None,
    typer.Option(
        "--model-dir",
        help="vllm only: local snapshot of the released weights (default "
        "$TACTILE_FLORA_MODEL_DIR, else the Hub id).",
    ),
]


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
    model_dir: ModelDirOpt = None,
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

        be = make_backend(backend, checkpoint, SourceRootReader(sdir), model_dir=model_dir)
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
    model_dir: ModelDirOpt = None,
    checkpoint: Annotated[
        list[str] | None,
        typer.Option(
            "--checkpoint",
            help="Restrict the grid to these checkpoints of the config (repeatable), e.g. to "
            "run one merged model per process.",
        ),
    ] = None,
) -> None:
    """Expand the full condition matrix, skipping cells whose predictions exist."""
    from tactile_flora.eval import harness
    from tactile_flora.eval.inference import make_backend, merged_model_dir

    cfg = load_config(config, GridConfig)
    ecfg = load_config(eval_config, EvalConfig)
    fcfg, sdir = _eval_common(force_config, sources_dir)
    if checkpoint:
        unknown = sorted(set(checkpoint) - set(cfg.checkpoints))
        if unknown:
            raise typer.BadParameter(
                f"not in the grid config: {unknown}", param_hint="--checkpoint"
            )
        cfg = cfg.model_copy(
            update={"checkpoints": [c for c in cfg.checkpoints if c in checkpoint]}
        )

    def _entry() -> None:
        from tactile_flora.video.readers import SourceRootReader

        if backend == "vllm":  # fail before the first cell, not after hours of released cells
            for ck in cfg.checkpoints:
                if ck != "released":
                    merged_model_dir(ck)
        reader = SourceRootReader(sdir)
        cells = harness.run_grid(
            cfg, ecfg, fcfg, data_root,
            lambda ck: make_backend(backend, ck, reader, model_dir=model_dir),
            sources_dir=sdir,
        )  # fmt: skip
        typer.echo(f"ran {len(cells)} cells")

    _run(config, data_root / "eval", _entry)


@eval_app.command("merge-adapter")
def eval_merge_adapter(
    run: Annotated[
        Path, typer.Option("--run", help="Run directory holding adapter/ (data/runs/<name>).")
    ],
    base: Annotated[
        str, typer.Option("--base", help="Base model: Hub id or local snapshot directory.")
    ] = "nvidia/Flora-Rewarder",
    out: Annotated[
        Path | None, typer.Option("--out", help="Output directory (default <run>/merged).")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing output directory.")
    ] = False,
    device_map: Annotated[
        str | None,
        typer.Option("--device-map", help="transformers device_map (default: auto on CUDA, CPU)."),
    ] = None,
) -> None:
    """Merge a run's LoRA adapter into the base weights (bf16 safetensors) for vLLM."""
    from tactile_flora.eval.merge_adapter import merge_adapter

    typer.echo(f"run:      {run}")
    typer.echo(f"git hash: {git_hash()}")
    try:
        merge_adapter(run, base, out, overwrite=overwrite, device_map=device_map, echo=typer.echo)
    except (RuntimeError, FileExistsError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
def baxter_calibrate(
    config: ConfigOpt,
    pads: Annotated[
        str,
        typer.Option(
            help="Pads to calibrate, comma-separated, in order; each is pushed against the "
            "scale on its own (e.g. L_a,L_b, or just L_b to redo one)."
        ),
    ] = "L_a,L_b",
    port: Annotated[
        str | None,
        typer.Option(help="Override serial_port, e.g. /dev/ttyUSB0 or udp://0.0.0.0:9870."),
    ] = None,
) -> None:
    """Fit per-pad counts-to-newton curves by pushing each pad against a kitchen scale."""
    from tactile_flora.baxter import calibrate as cal
    from tactile_flora.baxter.calibration import sensor_warning
    from tactile_flora.baxter.serial_protocol import open_serial

    cfg = load_config(config, BaxterConfig)
    out = Path(cfg.calibration_path)
    warning = sensor_warning(cfg)
    if warning:
        typer.echo(warning, err=True)

    def _entry() -> None:
        serial = open_serial(port or cfg.serial_port, cfg.baud, rate_hz=cfg.sample_rate_hz)

        def _ask(msg: str) -> str:
            return input(f"{msg}: ")

        names = [p.strip() for p in pads.split(",") if p.strip()]
        try:
            cal.calibrate(cfg, serial, _ask, out, pads=names, echo=typer.echo)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    _run(config, out, _entry)


@baxter_app.command("monitor")
def baxter_monitor(
    config: ConfigOpt,
    port: Annotated[
        str | None,
        typer.Option(help="Override serial_port, e.g. /dev/ttyUSB0 or udp://0.0.0.0:9870."),
    ] = None,
    zero: Annotated[
        bool, typer.Option(help="Send Z first (grippers open, pads unloaded).")
    ] = False,
    seconds: Annotated[
        float, typer.Option(help="Stop after this long; 0 runs until Ctrl-C.")
    ] = 0.0,
) -> None:
    """Print live FSR counts (and newtons if the calibration file exists); no ROS needed."""
    from tactile_flora.baxter.calibration import CalibrationFile
    from tactile_flora.baxter.monitor import monitor
    from tactile_flora.baxter.serial_protocol import open_serial

    cfg = load_config(config, BaxterConfig)
    cal_path = Path(cfg.calibration_path)

    def _entry() -> None:
        serial = open_serial(port or cfg.serial_port, cfg.baud, rate_hz=cfg.sample_rate_hz)
        cal = CalibrationFile.load(cal_path) if cal_path.exists() else None
        with contextlib.suppress(KeyboardInterrupt):
            monitor(
                serial, cfg.channels, cal, echo=typer.echo, zero=zero, duration_s=seconds or None
            )

    _run(config, Path("-"), _entry)


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
