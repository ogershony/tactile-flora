#!/bin/bash
# v4 orchestration on the Colab A100 (replaces stage1 + stage2 ordering so the GPU never waits on the
# whole render). CPU passes: A = camera-study subset x 6 cams; B = remaining TRAIN trials x 6 cams;
# C = DEV+TEST trials x the chosen collage. GPU: study phase 1 after A; prep#1 after B (normalization
# and classifier depend on TRAIN only, so prep#2 after C reproduces them and adds DEV/TEST rows);
# parity gate; study phase 2; smoke run. Training starts from stage3_train_eval.sh afterwards.
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY=/content/venv/bin/python; VPY=/content/vllm-venv/bin/python; S=/content/study; MODEL=$TACTILE_FLORA_MODEL_DIR
CANDS=22641005,22645029,22684737,22684755,23263780,25452061
R="/content/v4/hrdexdb_fetch_render.py --episodes data/raw/hrdexdb/metadata/episodes.parquet --decode-threads 2"
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a /content/logs/v4.status; }
mkdir -p $S
$PY - <<'PY'
import json, random
s = json.load(open("configs/v4/hrdexdb_split.json"))
eps = s["episodes"]; study = list(s["camera_study_train_subset"])
random.Random(0).shuffle(study)  # any prefix of the render order is a random TRAIN sample
rest = sorted(e for e, i in eps.items() if i["split"] == "train" and e not in set(study))
devtest = sorted(e for e, i in eps.items() if i["split"] in ("dev", "test"))
for name, ids in (("study", study), ("train_rest", rest), ("devtest", devtest)):
    open(f"/content/study/ids_{name}.txt", "w").write("\n".join(ids) + "\n")
    print(name, len(ids))
PY
# pass A then pass B on the CPU, in the background
( nice -n 10 $PY $R --ids-file $S/ids_study.txt --cams $CANDS --workers 12 > /content/logs/render_a.log 2>&1
  echo PASS_A_DONE >> /content/logs/render_a.log
  nice -n 10 $PY $R --ids-file $S/ids_train_rest.txt --cams $CANDS --workers 12 > /content/logs/render_b.log 2>&1
  echo PASS_B_DONE >> /content/logs/render_b.log ) &
until grep -q PASS_A_DONE /content/logs/render_a.log 2>/dev/null; do sleep 30; done
st pass_a_done
[ -f data/manifest/roboreward_500.parquet ] || nohup nice -n 10 $PY scripts/v4/fetch_roboreward_500.py > /content/logs/fetch_roboreward.log 2>&1 &
if [ ! -f $S/phase1.done ]; then
  $PY scripts/v4/camera_study_jobs.py configs/v4/hrdexdb_split.json data/frames/hrdexdb $S/phase1.jobs 1 $CANDS
  $VPY scripts/v4/flora_native_eval.py --model $MODEL --jobs $S/phase1.jobs --output $S/phase1.jsonl --max-num-seqs 16 > /content/logs/study1.log 2>&1 || { st STUDY1_FAILED; exit 1; }
  touch $S/phase1.done; pkill -9 -f EngineCore
fi
$PY scripts/v4/study_report.py $S/phase1.jsonl "cell96|22641005" > $S/phase1.txt 2>&1
RANKED=$($PY scripts/v4/choose_views.py $S/phase1.jsonl 2>>$S/phase1.txt | head -1); echo $RANKED > $S/ranked.txt
st "study1_done ranked=$RANKED"
if [ ! -f $S/phase2.done ]; then
  $PY scripts/v4/camera_study_jobs.py configs/v4/hrdexdb_split.json data/frames/hrdexdb $S/phase2.jobs 2 $(echo $RANKED | cut -d, -f1-4)
  $VPY scripts/v4/flora_native_eval.py --model $MODEL --jobs $S/phase2.jobs --output $S/phase2.jsonl --max-num-seqs 8 > /content/logs/study2.log 2>&1 || { st STUDY2_FAILED; exit 1; }
  touch $S/phase2.done; pkill -9 -f EngineCore
fi
cat $S/phase1.jsonl $S/phase2.jsonl > $S/all.jsonl
$PY scripts/v4/study_report.py $S/all.jsonl "native|$(echo $RANKED | cut -d, -f1)" > $S/phase2.txt 2>&1
CHOSEN=$($PY scripts/v4/choose_views.py $S/phase1.jsonl $S/phase2.jsonl 2>>$S/phase2.txt | sed -n 2p); echo $CHOSEN > $S/chosen.txt
st "study2_done chosen=$CHOSEN"
# pass C: DEV+TEST with the chosen collage only (CPU, background), after pass B
( until grep -q PASS_B_DONE /content/logs/render_b.log 2>/dev/null; do sleep 30; done
  nice -n 10 $PY $R --ids-file $S/ids_devtest.txt --cams $CHOSEN --workers 12 > /content/logs/render_c.log 2>&1
  echo PASS_C_DONE >> /content/logs/render_c.log ) &
until grep -q PASS_B_DONE /content/logs/render_b.log 2>/dev/null; do sleep 30; done
st pass_b_done
$PY - "$CHOSEN" <<'PY'
import sys, yaml
p = "configs/sources/hrdexdb.yaml"; d = yaml.safe_load(open(p)); d["hrdexdb_cameras"] = sys.argv[1].split(",")
yaml.safe_dump(d, open(p, "w"), sort_keys=False); print("hrdexdb_cameras", d["hrdexdb_cameras"])
PY
$PY scripts/v4/prepare_hrdexdb_v4.py --raw data/raw/hrdexdb --frames data/frames/hrdexdb --out data --cameras $CHOSEN > /content/logs/prep1.log 2>&1 || { st PREP1_FAILED; exit 1; }
$PY scripts/v4/build_v4_manifests.py > /content/logs/build_manifests.log 2>&1 || { st MANIFEST_FAILED; exit 1; }
st prep1_done
$PY scripts/v4/parity_check.py $S/phase1.jsonl 22641005 $MODEL > $S/parity.txt 2>&1 && st parity_ok || st PARITY_FAILED
pkill -9 -f EngineCore
# the parity check set hrdexdb_cameras back to the file's value (the chosen collage)
bash scripts/v4/colab/stage2_smoke.sh "$CHOSEN" > /content/logs/stage2.log 2>&1
st orchestrate_done
