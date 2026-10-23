r"""Freeze and submit one v5 tactile arm on the v3 split: force SFT, then force-adapter eval cells.

v5 arms test training-recipe changes (modality dropout, force chart) against the existing v3
tactile arm and the v2 control on the same 251 held-out clips, so only the force adapter is
trained and only its cells are evaluated (``evaluate_hrdexdb_clean.py --worker 2``). Everything
runs on the ``long`` partition so nothing preempts anything on matthew.

Run from the repository root after ``prepare_hrdexdb_v5.sh``-style base assembly:

    .venv/bin/python scripts/cluster/launch_hrdexdb_v5.py \\
        --base data/acceptance/hrdexdb-expanded-v5 --run-name v5a-dropout --tag v5a \\
        --train-config train-dropout.yaml --val-modes real,video_blank --test-modes real
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path

from tactile_flora.config import TrainConfig, load_config
from tactile_flora.verify import full_git_hash, manifest

REPO = Path("/home/ogershony/tactile-flora")
PYTHON = "data/cluster/rocm-venv/bin/python"
TORCHRUN = "data/cluster/rocm-venv/bin/torchrun"
EXPORTS = (
    "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
    "TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1"
)


def sha256(path: Path | str) -> str:
    """Hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(name: str, base: Path, hours: int, partition: str = "long") -> str:
    """Slurm preamble for a GPU step; ``normal`` (4 h cap) is not preempted by other users."""
    return "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH --account=scc",
            f"#SBATCH --qos={partition}",
            f"#SBATCH --partition={partition}",
            "#SBATCH --nodelist=matthew",
            "#SBATCH --gres=mi210:2",
            f"#SBATCH --mem={'128G' if partition == 'long' else '64G'}",
            f"#SBATCH --cpus-per-task={16 if partition == 'long' else 8}",
            f"#SBATCH --time={hours:02d}:00:00",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --output={REPO / base}/jobs/{name.removeprefix('flora-')}-%j.log",
            "set -euo pipefail",
            f"cd {REPO}",
            EXPORTS,
            "",
        ]
    )


def submit(script: Path, dependency: str | None) -> int:
    """Submit one script with sbatch; return its job id."""
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return int(out.strip().split(";")[0])


def main() -> None:
    """Freeze the plan, write the scripts and submit train -> eval-val -> eval-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tag", required=True, help="job-name infix, e.g. v5a")
    parser.add_argument("--train-config", default="train-hrdexdb.yaml")
    parser.add_argument("--control-run", default="expanded-v2-control")
    parser.add_argument("--val-modes", default="real,video_blank")
    parser.add_argument("--test-modes", default="real")
    parser.add_argument("--after", help="Slurm job id the training waits for (afterany).")
    parser.add_argument("--train-hours", type=int, default=8)
    parser.add_argument("--partition", choices=["long", "normal"], default="long")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()
    base = args.base
    plan_path = base / f"training-plan-{args.run_name}.json"
    if plan_path.exists():
        raise FileExistsError(f"plan already frozen: {plan_path}")
    cfg_path = base / "config" / args.train_config
    cfg = load_config(cfg_path, TrainConfig)
    configs = {"force": cfg.model_dump()}
    train_rows = [r for r in manifest.read_manifest(cfg.manifest) if r.split == "train"]
    per_epoch = math.ceil(len(train_rows) / cfg.effective_batch)
    audit = json.loads((base / "representation-audit.json").read_text())
    jobs = base / "jobs"
    jobs.mkdir(exist_ok=True)
    force_cfg = base / "config/force.yaml"
    sft = (
        f"scripts/cluster/run_hrdexdb_sft.py --base {base} --force-config {force_cfg} "
        f"--config {cfg_path} --run-name {args.run_name} --plan {plan_path}"
    )
    hours = min(args.train_hours, 4) if args.partition == "normal" else args.train_hours
    text = header(f"flora-{args.tag}-train", base, hours, args.partition)
    text += f"{TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 {sft}\n"
    paths = {"train": jobs / f"train-{args.run_name}.sbatch"}
    paths["train"].write_text(text)
    for split, modes in [("val", args.val_modes), ("test", args.test_modes)]:
        text = header(f"flora-{args.tag}-{split}", base, 4, args.partition)
        text += (
            f"{PYTHON} -u scripts/cluster/evaluate_hrdexdb_clean.py --base {base} "
            f"--eval-config {base}/config/eval-{split}.yaml --benchmark hrdexdb_aligned_{split} "
            f"--control-run {args.control_run} --force-run {args.run_name} "
            f"--force-config {force_cfg} --force-modes {modes} --worker 2\n"
        )
        paths[f"eval-{split}"] = jobs / f"eval-{split}-{args.run_name}.sbatch"
        paths[f"eval-{split}"].write_text(text)
    core = sorted(p for p in Path("tactile_flora").rglob("*.py"))
    inputs = [
        base / "config/sources/hrdexdb.yaml",
        cfg_path,
        force_cfg,
        base / "config/force-classifier.json",
        base / "verified/hrdexdb/norm.json",
        base / "manifest/holdout_ids.json",
        base / "representation-audit.json",
    ]
    plan = {
        "scope": (
            f"v5 arm {args.run_name}: v3 split, manifests, normalization and classifier; "
            f"recipe change under test: {json.dumps(cfg.modality_dropout.model_dump())}; "
            f"{len(train_rows)} training clips"
        ),
        "base_model": cfg.base_model,
        "configs": configs,
        "force_config": str(force_cfg),
        "manifest_sha256": manifest.manifest_hash(cfg.manifest),
        "validation_manifest_sha256": manifest.manifest_hash(base / "manifest/val.parquet"),
        "test_manifest_sha256": manifest.manifest_hash(base / "manifest/test.parquet"),
        "git_hash": audit["git_hash"],
        "actual_launch_git_hash": full_git_hash(),
        "core_code_sha256": {str(p): sha256(p) for p in core},
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "orchestration_sha256": {
            str(p): sha256(p)
            for p in [
                Path("scripts/cluster/run_hrdexdb_sft.py"),
                Path("scripts/cluster/evaluate_hrdexdb_clean.py"),
                *paths.values(),
            ]
        },
        "slurm_training_job": None,
        "slurm_training_jobs": [],
        "slurm_account": "scc",
        "slurm_qos": args.partition,
        "hardware": "matthew: 2 MI210",
        "selection": (
            "v3 recipe plus the recipe change above and an early stop on the training "
            "score-digit loss; final adapter only; no tuning on held-out clips."
        ),
        "optimizer_steps_per_epoch": per_epoch,
        "evaluation_conditions": {
            args.run_name: sorted(set(args.val_modes.split(",")) | set(args.test_modes.split(",")))
        },
        "frozen_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "slurm_evaluation_jobs": {},
        "previous_experiment": "data/acceptance/hrdexdb-expanded-v3",
    }

    def save() -> None:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    save()
    if args.no_submit:
        print(json.dumps({k: plan[k] for k in ["manifest_sha256", "optimizer_steps_per_epoch"]}))
        return
    train = submit(paths["train"], f"afterany:{args.after}" if args.after else None)
    plan["slurm_training_job"] = train
    plan["slurm_training_jobs"] = [train]
    save()
    val = submit(paths["eval-val"], f"afterok:{train}")
    test = submit(paths["eval-test"], f"afterok:{val}")
    plan["slurm_evaluation_jobs"] = {"val": val, "test": test}
    save()
    pipeline = {"training": train, "development_evaluation": val, "test_evaluation": test}
    (base / f"pipeline-jobs-{args.run_name}.json").write_text(json.dumps(pipeline, indent=2) + "\n")
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    main()
