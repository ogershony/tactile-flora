#!/bin/bash
# Seed-1 replicate on the rebuilt runtime (2026-10-24; the first runtime ended ~04:15 UTC with the
# seed-0 adapters): force arm first, then control, each trained, scored and analyzed before the next.
# Released FLORA's omitted test cells are rescored here for the seed-1 primary comparison (the seed-0
# cells exist only as the packed export in docs/v4/results). The seed-1 configs are copies of the
# seed-0 configs with only "seed" changed.
S=/content/logs/v4.status
st() { echo "$1 $(date -u +%FT%TZ)" >> $S; }
cd /content/tactile-flora
for arm in v4_force v4_control; do
  [ -f configs/train/${arm}_s1.yaml ] || sed 's/^seed: 0$/seed: 1/' configs/train/$arm.yaml > configs/train/${arm}_s1.yaml
  grep -q '^seed: 1$' configs/train/${arm}_s1.yaml || { st chain_s1_CONFIG_FAILED; exit 1; }
done
# seed-1 stage 3: DEV in real mode only for the force arm, no released DEV cells
sed -e 's/MODES=real,omitted,shuffled/MODES=real/' -e '/^prep2; gpu_free; grid released/d' scripts/v4/colab/stage3_train_eval.sh > /content/v4/stage3_s1.sh
grep -q 'MODES=real;;' /content/v4/stage3_s1.sh && ! grep -q 'grid released' /content/v4/stage3_s1.sh || { st chain_s1_STAGE3_COPY_FAILED; exit 1; }
AN() { /content/venv/bin/python scripts/v4/analyze.py data/eval data/hrdexdb_v4_audit.json > /content/results_v4.md 2> /content/logs/analyze.log; st "analyze_$1_exit_$?"; }
H="hrdexdb_v4_test reassemble_seg_test phantom_fail_test"
C=scripts/v4/colab/stage4_cells.sh
st chain_s1_start
bash /content/v4/stage3_s1.sh v4_force_s1 >> /content/logs/stage3_s1.log 2>&1
R=data/runs/v4_force_s1
if [ -f $R/merged/config.json ]; then
  bash $C $R real hrdexdb_v4_test
  bash $C released omitted hrdexdb_v4_test; AN force_s1_hrdexdb
  bash $C $R real reassemble_seg_test phantom_fail_test
  bash $C released omitted reassemble_seg_test phantom_fail_test; AN force_s1_real
  /content/venv/bin/python scripts/v4/classifier_cells.py hrdexdb_v4_dev hrdexdb_v4_test reassemble_seg_test > /content/logs/classifier_cells.log 2>&1 || st classifier_cells_FAILED
  bash $C $R omitted,shuffled $H; AN force_s1_all
else
  st chain_s1_v4_force_s1_FAILED
fi
bash /content/v4/stage3_s1.sh v4_control_s1 >> /content/logs/stage3_s1.log 2>&1
if [ -f data/runs/v4_control_s1/merged/config.json ]; then
  bash $C data/runs/v4_control_s1 omitted $H; AN control_s1
else
  st chain_s1_v4_control_s1_FAILED
fi
st chain_s1_done
