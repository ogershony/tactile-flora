"""Freeze the v3 training plan, write the batch scripts and submit the dependent job chain.

Mirrors the v2 orchestration: a training job (force then control, two MI210s), a development
evaluation depending on it, a test evaluation depending on that, and the report job last.
The plan pins the manifest-generation commit, core-code and input hashes, so documentation
commits during the run cannot change the learning inputs; see
``run_hrdexdb_sft.planned_report_version``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml

from tactile_flora.config import TrainConfig, load_config
from tactile_flora.verify import full_git_hash, manifest

BASE = Path("data/acceptance/hrdexdb-expanded-v3")
REPO = Path("/home/ogershony/tactile-flora")
PYTHON = "data/cluster/rocm-venv/bin/python"
TORCHRUN = "data/cluster/rocm-venv/bin/torchrun"
EXPORTS = (
    "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
    "TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1"
)


def sha256(path: Path | str) -> str:
    """Hex digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(name: str, base: Path, *, gpus: bool, hours: int, long: bool) -> str:
    """Slurm preamble shared by the chain's scripts."""
    lines = ["#!/bin/bash", "#SBATCH --account=scc"]
    lines += [f"#SBATCH --qos={'long' if long else 'normal'}"]
    lines += [f"#SBATCH --partition={'long' if long else 'normal'}"]
    if gpus:
        lines += [
            "#SBATCH --nodelist=matthew",
            "#SBATCH --gres=mi210:2",
            f"#SBATCH --mem={'128G' if long else '64G'}",
            f"#SBATCH --cpus-per-task={16 if long else 8}",
        ]
    else:
        lines += ["#SBATCH --cpus-per-task=2", "#SBATCH --mem=4G"]
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


def write_scripts(base: Path, force_run: str, control_run: str, force_modes: str) -> dict:
    """Write train/eval-val/eval-test/summary batch scripts; return their paths."""
    jobs = base / "jobs"
    jobs.mkdir(exist_ok=True)
    force_cfg = base / "config/force.yaml"
    train = header("flora-v3-sft", base, gpus=True, hours=12, long=True)
    probe = [
        "import json,platform,importlib.metadata",
        "from pathlib import Path",
        "import torch",
        "names=['transformers','peft','accelerate','numpy','av']",
        "props=torch.cuda.get_device_properties",
        "result={'host':platform.node(),'torch':torch.__version__,'hip':torch.version.hip,",
        " 'packages':{n:importlib.metadata.version(n) for n in names},",
        " 'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),'arch':props(i).gcnArchName,",
        "  'total_gib':props(i).total_memory/2**30} for i in range(torch.cuda.device_count())]}",
        f"Path('{base}/training-environment.json').write_text(json.dumps(result,indent=2)+'\\n')",
        "print(json.dumps(result),flush=True)",
    ]
    sft = f"scripts/cluster/run_hrdexdb_sft.py --base {base} --force-config {force_cfg}"
    launch = f"{TORCHRUN} --standalone --nnodes=1 --nproc-per-node=2 {sft}"
    train += "\n".join(
        [
            f"{PYTHON} - <<'ENVIRONMENT'",
            *probe,
            "ENVIRONMENT",
            f"{launch} --run-name {force_run}",
            f"{launch} --config {base}/config/train-control.yaml --run-name {control_run}",
            "",
        ]
    )
    paths = {"train": jobs / "train.sbatch"}
    paths["train"].write_text(train)
    for split in ["val", "test"]:
        text = header(f"flora-v3-{split}", base, gpus=True, hours=4, long=False)
        text += (
            f"{PYTHON} -u scripts/cluster/evaluate_hrdexdb_clean.py --base {base} "
            f"--eval-config {base}/config/eval-{split}.yaml --benchmark hrdexdb_aligned_{split} "
            f"--control-run {control_run} --force-run {force_run} --force-config {force_cfg} "
            f"--force-modes {force_modes}\n"
        )
        paths[f"eval-{split}"] = jobs / f"eval-{split}.sbatch"
        paths[f"eval-{split}"].write_text(text)
    text = header("flora-v3-report", base, gpus=False, hours=1, long=False)
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
    """Write the plan, the batch scripts and submit the chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--force-run", default="expanded-v3-force")
    parser.add_argument("--control-run", default="expanded-v3-control")
    parser.add_argument("--force-modes", default="real,shuffled,video_blank")
    parser.add_argument("--after", help="Slurm job id the training job waits for (afterany).")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()
    base = args.base
    if (base / "training-plan.json").exists():
        raise FileExistsError("training plan already frozen for this base")
    configs = {
        name: load_config(base / f"config/{file}.yaml", TrainConfig).model_dump()
        for name, file in [("force", "train-hrdexdb"), ("control", "train-control")]
    }
    train_rows = [
        r for r in manifest.read_manifest(configs["force"]["manifest"]) if r.split == "train"
    ]
    epochs = configs["force"]["epochs"]
    per_epoch = math.ceil(len(train_rows) / configs["force"]["effective_batch"])
    audit = json.loads((base / "representation-audit.json").read_text())
    paths = write_scripts(base, args.force_run, args.control_run, args.force_modes)
    core = sorted(p for p in Path("tactile_flora").rglob("*.py"))
    inputs = [
        base / "config/sources/hrdexdb.yaml",
        base / "config/train-hrdexdb.yaml",
        base / "config/train-control.yaml",
        base / "config/force.yaml",
        base / "verified/hrdexdb/norm.json",
        *[p for p in [base / "config/force-classifier.json"] if p.exists()],
        base / "manifest/holdout_ids.json",
        base / "representation-audit.json",
        Path("data/acceptance/hrdexdb-expanded-v2/download-plan.json"),
        Path("data/acceptance/hrdexdb-expanded-v2/alignment-audit.json"),
    ]
    plan = {
        "scope": (
            "v3: corrected HRDexDB force representation (tared, peak-ranked channels, "
            "sign-corrected closure, 5% contact floor) on the frozen v2 splits; "
            f"{len(train_rows)} training clips"
        ),
        "base_model": configs["force"]["base_model"],
        "configs": configs,
        "force_config": str(base / "config/force.yaml"),
        "manifest_sha256": manifest.manifest_hash(configs["force"]["manifest"]),
        "validation_manifest_sha256": manifest.manifest_hash(base / "manifest/val.parquet"),
        "test_manifest_sha256": manifest.manifest_hash(base / "manifest/test.parquet"),
        "git_hash": audit["git_hash"],
        "actual_launch_git_hash": full_git_hash(),
        "report_version_note": (
            "Manifest-generation commit is pinned; actual learning code and inputs are checked "
            "by hashes before each run."
        ),
        "core_code_sha256": {str(p): sha256(p) for p in core},
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "orchestration_sha256": {
            str(p): sha256(p)
            for p in [
                Path("scripts/cluster/run_hrdexdb_sft.py"),
                Path("scripts/cluster/evaluate_hrdexdb_clean.py"),
                Path("scripts/cluster/summarize_hrdexdb_expanded.py"),
                Path("scripts/cluster/prepare_hrdexdb_v3.py"),
                *paths.values(),
            ]
        },
        "slurm_training_job": None,
        "slurm_account": "scc",
        "slurm_qos": "long",
        "hardware": "matthew: 2 MI210",
        "selection": (
            "Fixed four epochs from released base, seed 17; final adapters only; no "
            "tuning/checkpoint selection on development or test. Matched v2 settings; only "
            "the force representation and manifests' norm/report versions differ."
        ),
        "optimizer_steps_per_run": per_epoch * epochs,
        "evaluation_conditions": {
            "released": ["omitted", "real"],
            args.control_run: ["omitted"],
            args.force_run: args.force_modes.split(","),
        },
        "frozen_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "slurm_evaluation_jobs": {},
        "previous_experiment": "data/acceptance/hrdexdb-expanded-v2",
    }
    plan_path = base / "training-plan.json"

    def save() -> None:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    save()
    if args.no_submit:
        print(json.dumps({k: plan[k] for k in ["manifest_sha256", "optimizer_steps_per_run"]}))
        return
    train = submit(paths["train"], f"afterany:{args.after}" if args.after else None)
    plan["slurm_training_job"] = train
    save()  # the launcher checks SLURM_JOB_ID against this before pinning the version
    val = submit(paths["eval-val"], f"afterok:{train}")
    test = submit(paths["eval-test"], f"afterok:{val}")
    report = submit(paths["summary"], f"afterok:{test}")
    plan["slurm_evaluation_jobs"] = {"val": val, "test": test}
    save()
    pipeline = {
        "training": train,
        "development_evaluation": val,
        "test_evaluation": test,
        "summary": report,
        "summary_script_sha256": sha256("scripts/cluster/summarize_hrdexdb_expanded.py"),
        "outputs": ["final-result.json", "final-result.md"],
    }
    (base / "pipeline-jobs.json").write_text(json.dumps(pipeline, indent=2) + "\n")
    print(json.dumps(pipeline, indent=2))
    print(yaml.safe_dump({"plan": str(plan_path)}))


if __name__ == "__main__":
    main()
