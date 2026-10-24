#!/bin/bash
# Plan fallback after the 3-view smoke OOM (peak 81,150 MiB): the best <= 2-view collage by the
# same rule (22641005+22684737: 0.745 vs 0.723 single view), same 64 frames. Re-preps TRAIN for the
# new camera set, reruns the smoke with the GPU memory sampled, and on success writes a fresh
# status file (history kept in v4.status.run2) and re-arms chain3/4/5. Parity stays valid: it ran
# on 22641005, the first camera of both collages.
set -uo pipefail
cd /content/tactile-flora
S=/content/logs/v4.status
PY=/content/venv/bin/python
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a $S; }
CAMS=22641005,22684737
echo $CAMS > /content/study/chosen.txt
$PY - "$CAMS" <<'PY'
import sys, yaml
p = "configs/sources/hrdexdb.yaml"; d = yaml.safe_load(open(p)); d["hrdexdb_cameras"] = sys.argv[1].split(",")
yaml.safe_dump(d, open(p, "w"), sort_keys=False); print("hrdexdb_cameras", d["hrdexdb_cameras"])
PY
st "fallback_2view chosen=$CAMS"
$PY scripts/v4/prepare_hrdexdb_v4.py --raw data/raw/hrdexdb --frames data/frames/hrdexdb --out data --cameras $CAMS > /content/logs/prep1b.log 2>&1 || { st PREP1B_FAILED; exit 1; }
$PY scripts/v4/build_v4_manifests.py > /content/logs/build_manifests_b.log 2>&1 || { st MANIFESTB_FAILED; exit 1; }
st prep1b_done
(for i in $(seq 1 600); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done > /content/logs/smoke2_mem.log 2>&1 &)
bash scripts/v4/colab/stage2_smoke.sh $CAMS > /content/logs/stage2b.log 2>&1
if tail -n 1 $S | grep -q smoke_exit_0; then
  st "smoke2_peak_mib $(sort -n /content/logs/smoke2_mem.log | tail -n 1)"
  cp $S /content/logs/v4.status.run2
  { grep parity_ok /content/logs/v4.status.run2 | tail -n 1; grep -E 'fallback_2view|smoke_exit_0|smoke2_peak' /content/logs/v4.status.run2; echo "orchestrate_done $(date -u +%FT%TZ)"; } > $S.new && mv $S.new $S
  setsid nohup bash /content/v4/chain3.sh > /dev/null 2>&1 < /dev/null &
  setsid nohup bash /content/v4/chain4.sh > /dev/null 2>&1 < /dev/null &
  setsid nohup bash scripts/v4/colab/chain5.sh > /dev/null 2>&1 < /dev/null &
  echo "chains_rearmed $(date -u +%FT%TZ)" >> $S
else
  st SMOKE2_FAILED
fi
