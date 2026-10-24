#!/bin/bash
# After chain4: a second seed of both arms (force first) on the identical recipe, with the exploratory
# blank-video cells in between. Ordered so the seed-1 force result on HRDexDB test lands before 08:26
# UTC 2026-10-24, when this VM (assigned 08:26 UTC 2026-10-23; the 18:38 container rebuild kept the
# host, now with 4 vCPUs, ~2.27 min per training step) would end if Colab's 24 h limit applies:
# seed-1 DEV in real mode only, then its real-force test cells (HRDexDB first, then REASSEMBLE and
# PHANTOM), then the blank-video cells, then the rest of its stage 4, then seed-1 control. The
# seed-1 configs are copies of the final seed-0 configs (failure weight and local base model already
# set) with only "seed" changed.
S=/content/logs/v4.status
until grep -q -E 'chain4_done|chain4_HELD' $S 2>/dev/null; do sleep 120; done
grep -q chain4_done $S || { echo "chain5_HELD $(date -u +%FT%TZ)" >> $S; exit 0; }
cd /content/tactile-flora
for arm in v4_force v4_control; do
  [ -f configs/train/${arm}_s1.yaml ] || sed 's/^seed: 0$/seed: 1/' configs/train/$arm.yaml > configs/train/${arm}_s1.yaml
  grep -q '^seed: 1$' configs/train/${arm}_s1.yaml || { echo "chain5_CONFIG_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
done
sed 's/MODES=real,omitted,shuffled/MODES=real/' scripts/v4/colab/stage3_train_eval.sh > /content/v4/stage3_s1.sh
grep -q 'MODES=real;;' /content/v4/stage3_s1.sh || { echo "chain5_STAGE3_COPY_FAILED $(date -u +%FT%TZ)" >> $S; exit 1; }
AN() { /content/venv/bin/python scripts/v4/analyze.py data/eval data/hrdexdb_v4_audit.json > /content/results_v4.md 2> /content/logs/analyze.log; echo "analyze_$1_exit_$? $(date -u +%FT%TZ)" >> $S; }
echo "chain5_start $(date -u +%FT%TZ)" >> $S
R=data/runs/v4_force_s1
bash /content/v4/stage3_s1.sh v4_force_s1 >> /content/logs/stage3_s1.log 2>&1
if [ -f $R/merged/config.json ]; then
  bash scripts/v4/colab/stage4_first.sh $R hrdexdb_v4_test; AN force_s1_hrdexdb
  bash scripts/v4/colab/stage4_first.sh $R reassemble_seg_test phantom_fail_test; AN force_s1_real
else
  echo "chain5_v4_force_s1_FAILED $(date -u +%FT%TZ)" >> $S
fi
bash scripts/v4/colab/stage_blank.sh; AN blank   # exploratory blank-video cells of the seed-0 force adapter
[ -f $R/merged/config.json ] && { bash scripts/v4/colab/stage4_test.sh $R >> /content/logs/stage4_s1.log 2>&1; AN force_s1_all; }
bash /content/v4/stage3_s1.sh v4_control_s1 >> /content/logs/stage3_s1.log 2>&1
if [ -f data/runs/v4_control_s1/merged/config.json ]; then
  bash scripts/v4/colab/stage4_test.sh data/runs/v4_control_s1 >> /content/logs/stage4_s1.log 2>&1; AN control_s1
else
  echo "chain5_v4_control_s1_FAILED $(date -u +%FT%TZ)" >> $S
fi
echo "chain5_done $(date -u +%FT%TZ)" >> $S
