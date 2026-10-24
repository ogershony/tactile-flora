#!/bin/bash
# After stage 3: a DEV sanity gate (no collapsed or unparsable adapter), then the force-only
# classifier baseline cells, the locked test sets (stage4_test.sh) and the analysis. The gate
# looks only at DEV spread and parse failures, never at a metric, so it cannot steer the test.
S=/content/logs/v4.status
until grep -q -E 'chain3_exit|chain3_HELD' $S 2>/dev/null; do sleep 120; done
cd /content/tactile-flora
PY=/content/venv/bin/python
if ! $PY - > /content/logs/chain4_gate.log 2>&1 <<'PY'
import sys
import pandas as pd
ok = True
for ck, mode in (("v4_force", "real"), ("v4_control", "omitted"), ("released", "omitted")):
    d = pd.read_parquet(f"data/eval/{ck}/hrdexdb_v4_dev/{mode}/predictions.parquet")
    sd, pf = float(d.expected_score.std()), float(d.parse_failed.mean())
    print(ck, mode, len(d), "E[s] std", round(sd, 3), "parse_failed", round(pf, 3))
    ok &= ck == "released" or (sd >= 0.05 and pf <= 0.05)
sys.exit(0 if ok else 1)
PY
then
  echo "chain4_HELD_gate $(date -u +%FT%TZ)" >> $S; exit 0
fi
echo "chain4_gate_ok $(date -u +%FT%TZ)" >> $S
$PY scripts/v4/classifier_cells.py hrdexdb_v4_dev hrdexdb_v4_test reassemble_seg_test > /content/logs/classifier_cells.log 2>&1 || echo "classifier_cells_FAILED $(date -u +%FT%TZ)" >> $S
bash scripts/v4/colab/stage4_test.sh data/runs/v4_force data/runs/v4_control released > /content/logs/stage4.log 2>&1  # force first: a failure there shows ~2 h sooner
$PY scripts/v4/analyze.py data/eval data/hrdexdb_v4_audit.json > /content/results_v4.md 2> /content/logs/analyze.log
echo "chain4_done_$? $(date -u +%FT%TZ)" >> $S
