"""Freeze and submit one v4 fold: force SFT, control SFT, dev eval, test eval, summary.

Unlike the v3 launcher, the two arms are separate Slurm jobs (each must fit the 12 h ``long``
limit on its own) and every GPU step runs on the ``long`` partition, because on matthew a job on
``normal`` preempts a running ``long`` job. The training plan pins core code and input hashes
and lists BOTH training job ids so ``run_hrdexdb_sft.py`` keeps the frozen report version.

Run from the repository root after ``prepare_hrdexdb_folds.py``:

    .venv/bin/python scripts/cluster/launch_hrdexdb_folds.py --fold B
    .venv/bin/python scripts/cluster/launch_hrdexdb_folds.py --fold C --after <fold B summary job>
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


def header(name: str, base: Path, *, gpus: bool, hours: int) -> str:
    """Slurm preamble; GPU steps always go to the long partition (never preempt a sibling)."""
    lines = ["#!/bin/bash", "#SBATCH --account=scc"]
    if gpus:
        lines += [
            "#SBATCH --qos=long",
            "#SBATCH --partition=long",
            "#SBATCH --nodelist=matthew",
            "#SBATCH --gres=mi210:2",
            "#SBATCH --mem=128G",
            "#SBATCH --cpus-per-task=16",
        ]
    else:
        lines += [
            "#SBATCH --qos=normal",
            "#SBATCH --partition=normal",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --mem=4G",
        ]
    lines += [
        f"#SBATCH --time={hours:02d}:00:00",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --output={REPO / base}/jobs/{name.removeprefix('flora-')}-%j.log",
        "set -euo pipefail",
        f"cd {REPO}",
        EXPORTS,
        "",
    ]
    return "\n".join(lines)


def write_scripts(base: Path, tag: str, force_run: str, control_run: str, force_modes: str) -> dict:
    """Write the five batch scripts of one fold; return their paths."""
    jobs = base / "jobs"
    jobs.mkdir(exist_ok=True)
    force_cfg = base / "config/force.yaml"
    sft = f"scripts/cluster/run_hrdexdb_sft.py --base {base} --force-config {force_cfg}"
    launch = f"{TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 {sft}"
    paths = {}
    probe = (
        f"{PYTHON} - <<'ENVIRONMENT'\n"
        "import json,platform,importlib.metadata\n"
        "from pathlib import Path\n"
        "import torch\n"
        "names=['transformers','peft','accelerate','numpy','av']\n"
        "props=torch.cuda.get_device_properties\n"
        "result={'host':platform.node(),'torch':torch.__version__,'hip':torch.version.hip,\n"
        " 'packages':{n:importlib.metadata.version(n) for n in names},\n"
        " 'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),'arch':props(i).gcnArchName,\n"
        "  'total_gib':props(i).total_memory/2**30} for i in range(torch.cuda.device_count())]}\n"
        f"Path('{base}/training-environment.json').write_text(json.dumps(result,indent=2)+'\\n')\n"
        "print(json.dumps(result),flush=True)\n"
        "ENVIRONMENT\n"
    )
    text = header(f"flora-{tag}-force", base, gpus=True, hours=10) + probe
    text += f"{launch} --run-name {force_run}\n"
    paths["train-force"] = jobs / "train-force.sbatch"
    paths["train-force"].write_text(text)
    text = header(f"flora-{tag}-control", base, gpus=True, hours=10)
    text += f"{launch} --config {base}/config/train-control.yaml --run-name {control_run}\n"
    paths["train-control"] = jobs / "train-control.sbatch"
    paths["train-control"].write_text(text)
    for split in ["val", "test"]:
        text = header(f"flora-{tag}-{split}", base, gpus=True, hours=6)
        text += (
            f"{PYTHON} -u scripts/cluster/evaluate_hrdexdb_clean.py --base {base} "
            f"--eval-config {base}/config/eval-{split}.yaml --benchmark hrdexdb_aligned_{split} "
            f"--control-run {control_run} --force-run {force_run} --force-config {force_cfg} "
            f"--force-modes {force_modes}\n"
        )
        paths[f"eval-{split}"] = jobs / f"eval-{split}.sbatch"
        paths[f"eval-{split}"].write_text(text)
    text = header(f"flora-{tag}-report", base, gpus=False, hours=1)
    text += (
        f"{PYTHON} -u scripts/cluster/summarize_hrdexdb_expanded.py --base {base} "
        f"--force-run {force_run} --control-run {control_run}\n"
    )
    paths["summary"] = jobs / "summary.sbatch"
    paths["summary"].write_text(text)
    return paths


def submit(script: Path, dependency: str | None) -> int:
    """Submit one script with sbatch; return its job id."""
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return int(out.strip().split(";")[0])


def main() -> None:
    """Freeze the plan, write the scripts and submit the chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["B", "C"], required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--force-modes", default="real,shuffled")
    parser.add_argument("--after", help="Slurm job id the force training waits for (afterany).")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()
    base = args.base or Path(f"data/acceptance/hrdexdb-expanded-v4-fold{args.fold}")
    tag = f"v4{args.fold}"
    force_run, control_run = f"fold{args.fold}-force", f"fold{args.fold}-control"
    if (base / "training-plan.json").exists():
        raise FileExistsError("training plan already frozen for this base")
    configs = {
        name: load_config(base / f"config/{file}.yaml", TrainConfig).model_dump()
        for name, file in [("force", "train-hrdexdb"), ("control", "train-control")]
    }
    train_rows = [
        r for r in manifest.read_manifest(configs["force"]["manifest"]) if r.split == "train"
    ]
    per_epoch = math.ceil(len(train_rows) / configs["force"]["effective_batch"])
    audit = json.loads((base / "representation-audit.json").read_text())
    paths = write_scripts(base, tag, force_run, control_run, args.force_modes)
    core = sorted(p for p in Path("tactile_flora").rglob("*.py"))
    inputs = [
        base / "config/sources/hrdexdb.yaml",
        base / "config/train-hrdexdb.yaml",
        base / "config/train-control.yaml",
        base / "config/force.yaml",
        base / "config/force-classifier.json",
        base / "verified/hrdexdb/norm.json",
        base / "manifest/holdout_ids.json",
        base / "representation-audit.json",
        Path("data/acceptance/hrdexdb-expanded-v2/download-plan.json"),
        Path("data/acceptance/hrdexdb-expanded-v2/alignment-audit.json"),
    ]
    plan = {
        "scope": (
            f"v4 fold {args.fold}: v3 representation and DFTP trials, rotated object split; "
            f"{len(train_rows)} training clips; held out {audit['held_out_objects']}"
        ),
        "base_model": configs["force"]["base_model"],
        "configs": configs,
        "force_config": str(base / "config/force.yaml"),
        "manifest_sha256": manifest.manifest_hash(configs["force"]["manifest"]),
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
                Path("scripts/cluster/summarize_hrdexdb_expanded.py"),
                Path("scripts/cluster/prepare_hrdexdb_folds.py"),
                *paths.values(),
            ]
        },
        "slurm_training_job": None,
        "slurm_training_jobs": [],
        "slurm_account": "scc",
        "slurm_qos": "long",
        "hardware": "matthew: 2 MI210",
        "selection": (
            "v3 recipe (4 epochs max, seed 17, LoRA r32/a64, lr 2e-5, batch 16) plus an early stop "
            "on the training score-digit loss (mean over 10 steps below "
            f"{configs['force']['early_stop_score_loss']} after at least "
            f"{configs['force']['early_stop_min_epochs']} epochs); final adapters only; no "
            "tuning on held-out clips."
        ),
        "optimizer_steps_per_epoch": per_epoch,
        "evaluation_conditions": {
            "released": ["omitted", "real"],
            control_run: ["omitted"],
            force_run: args.force_modes.split(","),
        },
        "frozen_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "slurm_evaluation_jobs": {},
        "previous_experiment": "data/acceptance/hrdexdb-expanded-v3",
    }
    plan_path = base / "training-plan.json"

    def save() -> None:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    save()
    if args.no_submit:
        print(json.dumps({k: plan[k] for k in ["manifest_sha256", "optimizer_steps_per_epoch"]}))
        return
    force = submit(paths["train-force"], f"afterany:{args.after}" if args.after else None)
    control = submit(paths["train-control"], f"afterok:{force}")
    plan["slurm_training_job"] = force
    plan["slurm_training_jobs"] = [force, control]
    save()  # the trainer checks SLURM_JOB_ID against these before pinning the version
    val = submit(paths["eval-val"], f"afterok:{control}")
    test = submit(paths["eval-test"], f"afterok:{val}")
    report = submit(paths["summary"], f"afterok:{test}")
    plan["slurm_evaluation_jobs"] = {"val": val, "test": test}
    save()
    pipeline = {
        "training": force,
        "training_control": control,
        "development_evaluation": val,
        "test_evaluation": test,
        "summary": report,
        "summary_script_sha256": sha256("scripts/cluster/summarize_hrdexdb_expanded.py"),
        "outputs": ["final-result.json", "final-result.md"],
    }
    (base / "pipeline-jobs.json").write_text(json.dumps(pipeline, indent=2) + "\n")
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    main()
