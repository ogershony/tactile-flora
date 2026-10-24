#!/bin/bash
# v4 stage 1 on the Colab A100 runtime: HRDexDB prep, merged manifests, zero-shot camera study on
# the TRAIN subset (released FLORA through NVIDIA's batch_utils), pre-registered view choice.
# Status lines: /content/logs/v4.status. Run from /content/tactile-flora with the v4 patch applied.
set -uo pipefail
cd /content/tactile-flora
export TACTILE_FLORA_MODEL_DIR=/content/models/Flora-Rewarder VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PY=/content/venv/bin/python; VPY=/content/vllm-venv/bin/python; S=/content/study
CANDS=22641005,22645029,22684737,22684755,23263780,25452061
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a /content/logs/v4.status; }
mkdir -p $S
until grep -q "^DONE" /content/logs/render.log 2>/dev/null; do sleep 60; done
until grep -q VLLM_VENV_DONE /content/logs/setup_v4.log 2>/dev/null; do sleep 30; done
st render_and_venv_ready
[ -f data/manifest/roboreward_500.parquet ] || nohup $PY scripts/v4/fetch_roboreward_500.py > /content/logs/fetch_roboreward.log 2>&1 &
if [ ! -f data/manifest/hrdexdb_v4_test.parquet ]; then
  $PY scripts/v4/prepare_hrdexdb_v4.py --raw data/raw/hrdexdb --frames data/frames/hrdexdb --out data > /content/logs/prep_hrdexdb.log 2>&1 || { st PREP_FAILED; tail -20 /content/logs/prep_hrdexdb.log; exit 1; }
fi
$PY scripts/v4/build_v4_manifests.py > /content/logs/build_manifests.log 2>&1 || { st MANIFEST_FAILED; exit 1; }
st prep_done
if [ -n "${CHOSEN:-}" ]; then
  # the collage was chosen by the same rule on the old runtime (docs/v4/plan.md, Phase 2): only the
  # parity reference is needed here, batch_utils on 20 study clips with camera 22641005
  echo "$CHOSEN" > $S/chosen.txt
  $PY scripts/v4/camera_study_jobs.py configs/v4/hrdexdb_split.json data/frames/hrdexdb $S/phase1.jobs 1 22641005
  $PY - <<'PY'
import json
jobs = [json.loads(x) for x in open("/content/study/phase1.jobs") if "|native|22641005" in x][:20]
open("/content/study/parity_ref.jobs", "w").write("".join(json.dumps(j) + "\n" for j in jobs))
PY
  $VPY scripts/v4/flora_native_eval.py --model $TACTILE_FLORA_MODEL_DIR --jobs $S/parity_ref.jobs --output $S/phase1.jsonl --max-num-seqs 8 > /content/logs/study1.log 2>&1
  pkill -9 -f EngineCore
  $PY scripts/v4/parity_check.py $S/phase1.jsonl 22641005 $TACTILE_FLORA_MODEL_DIR > $S/parity.txt 2>&1 && st parity_ok || st PARITY_FAILED
  pkill -9 -f EngineCore
  st "study_skipped chosen=$CHOSEN"
  exit 0
fi
if [ ! -f $S/phase1.done ]; then
  $PY scripts/v4/camera_study_jobs.py configs/v4/hrdexdb_split.json data/frames/hrdexdb $S/phase1.jobs 1 $CANDS
  $VPY scripts/v4/flora_native_eval.py --model $TACTILE_FLORA_MODEL_DIR --jobs $S/phase1.jobs --output $S/phase1.jsonl --max-num-seqs 16 > /content/logs/study1.log 2>&1 || { st STUDY1_FAILED; tail -20 /content/logs/study1.log; exit 1; }
  touch $S/phase1.done
fi
$PY scripts/v4/study_report.py $S/phase1.jsonl "cell96|22641005" > $S/phase1.txt 2>&1
RANKED=$($PY scripts/v4/choose_views.py $S/phase1.jsonl 2>>$S/phase1.txt | head -1)
echo "$RANKED" > $S/ranked.txt
st "study1_done ranked=$RANKED"
# gate: our harness reproduces batch_utils on the same frames (|dE[s]| <= 0.02 on 20 clips)
$PY scripts/v4/parity_check.py $S/phase1.jsonl 22641005 $TACTILE_FLORA_MODEL_DIR > $S/parity.txt 2>&1 && st parity_ok || st PARITY_FAILED
pkill -9 -f EngineCore
if [ ! -f $S/phase2.done ]; then
  $PY scripts/v4/camera_study_jobs.py configs/v4/hrdexdb_split.json data/frames/hrdexdb $S/phase2.jobs 2 $(echo $RANKED | cut -d, -f1-4)
  $VPY scripts/v4/flora_native_eval.py --model $TACTILE_FLORA_MODEL_DIR --jobs $S/phase2.jobs --output $S/phase2.jsonl --max-num-seqs 8 > /content/logs/study2.log 2>&1 || { st STUDY2_FAILED; tail -20 /content/logs/study2.log; exit 1; }
  touch $S/phase2.done
fi
$PY scripts/v4/study_report.py <(cat $S/phase1.jsonl $S/phase2.jsonl) "native|$(echo $RANKED | cut -d, -f1)" > $S/phase2.txt 2>&1
CHOSEN=$($PY scripts/v4/choose_views.py $S/phase1.jsonl $S/phase2.jsonl 2>>$S/phase2.txt | sed -n 2p)
echo "$CHOSEN" > $S/chosen.txt
st "study2_done chosen=$CHOSEN"
