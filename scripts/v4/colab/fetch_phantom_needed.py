"""Fetch the PHANTOM episodes the v4 manifests use (train_judge_pf + phantom_fail_test: 87 of the 180 in
data/raw/phantom/manifests/plan.json, 5,070 of 10,513 files), or with --all every episode in the plan,
with hf_hub_download at the plan's pinned revision. Training needs --all: the PHANTOM force inputs
take per-object statistics over the whole success set (verify/windowing.build_source_inputs), so the
87 manifest episodes alone fail with a KeyError (2026-10-24 rebuild). Resumable: files already present
with the planned size are skipped.
"""
import concurrent.futures as cf
import json
import os

import sys

import pandas as pd
from huggingface_hub import hf_hub_download

root = "data/raw/phantom"
plan = json.load(open(f"{root}/manifests/plan.json"))
need = set()
for m in ("data/manifest/train_judge_pf.parquet", "data/manifest/phantom_fail_test.parquet"):
    d = pd.read_parquet(m)
    need |= {e.split("/")[-1] for e in d[d.source == "phantom"].episode_id}
files = plan["files"] if "--all" in sys.argv else [f for f in plan["files"] if f["path"].split("/")[2] in need]
todo = [f["path"] for f in files
        if not (os.path.exists(os.path.join(root, f["path"])) and os.path.getsize(os.path.join(root, f["path"])) == f["size"])]
print("episodes", len(need), "files", len(files), "todo", len(todo), flush=True)


def get(p):
    for _ in range(6):
        try:
            hf_hub_download("armteam/phantom-episodes", p, repo_type="dataset", revision=plan["revision"], local_dir=root)
            return p, True
        except Exception:  # noqa: BLE001
            pass
    return p, False


done = bad = 0
with cf.ThreadPoolExecutor(6) as ex:
    for p, ok in ex.map(get, todo):
        done += 1
        if not ok:
            bad += 1
            print("FAILED", p, flush=True)
        if done % 500 == 0:
            print("done", done, "/", len(todo), flush=True)
print("PHANTOM_FETCH_DONE failed", bad, flush=True)
