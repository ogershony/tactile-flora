#!/bin/bash
# Start stage 3 when the orchestrator ends with parity_ok and smoke_exit_0; otherwise hold for a human.
S=/content/logs/v4.status
until grep -q -E 'orchestrate_done|FAILED' $S 2>/dev/null; do sleep 60; done
if grep -q parity_ok $S && grep -q smoke_exit_0 $S && ! grep -q FAILED $S; then
  echo "chain3_start $(date -u +%FT%TZ)" >> $S
  bash /content/tactile-flora/scripts/v4/colab/stage3_train_eval.sh > /content/logs/stage3.log 2>&1
  echo "chain3_exit_$? $(date -u +%FT%TZ)" >> $S
else
  echo "chain3_HELD $(date -u +%FT%TZ)" >> $S
fi
