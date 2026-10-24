#!/bin/bash
# Progress + speed report for the ACT run; also evaluates any new checkpoints on the held-out episodes.
# usage: progress.sh <jobid> <run_dir> [total_steps]
JOB=$1; RUN=$2; TOTAL=${3:-50000}
LOG=$HOME/.cache/lerobot-slurm/$JOB-act-baxter.out
SCRIPTS=$(cd "$(dirname "$0")" && pwd); cd "${BAXTER_ACT_HOME:-$HOME/baxter_teleop}" && source .venv/bin/activate && export HF_TOKEN=$(cat ~/.cache/huggingface/token)
state=$(squeue -j "$JOB" -h -o "%T %M" 2>/dev/null); echo "job $JOB: ${state:-finished}"
last=$(grep "step:" "$LOG" | tail -1)
step=$(grep -oE "step:[0-9.]+K?" <<<"$last" | cut -d: -f2); stepn=$(python -c "s='$step'; print(int(float(s[:-1])*1000) if s.endswith('K') else int(s))")
upd=$(grep -oE "updt_s:[0-9.]+" <<<"$last" | cut -d: -f2); dat=$(grep -oE "data_s:[0-9.]+" <<<"$last" | cut -d: -f2); sps=$(grep -oE "smp/s:[0-9]+" <<<"$last" | cut -d: -f2)
start=$(grep -oE "^INFO [0-9-]+ [0-9:]+" "$LOG" | head -1 | awk '{print $2" "$3}'); t0=$(date -d "$start" +%s)
tnow=$(grep -oE "^INFO [0-9-]+ [0-9:]+" "$LOG" | tail -1 | awk '{print $2" "$3}'); t1=$(date -d "$tnow" +%s)
rate=$(python -c "print(($t1-$t0)/max($stepn,1))")
eta=$(python -c "import datetime; r=$rate; print(datetime.timedelta(seconds=int(($TOTAL-$stepn)*r)), '->', (datetime.datetime.now()+datetime.timedelta(seconds=($TOTAL-$stepn)*r)).strftime('%H:%M'))")
echo "step $stepn/$TOTAL | update ${upd}s + data ${dat}s per step | ${sps} samples/s | avg ${rate:0:5} s/step since start | remaining $eta"
echo "train: $(grep -oE "loss:[0-9.]+|l1_loss:[0-9.]+|kld_loss:[0-9.]+" <<<"$last" | paste -sd' ')"
echo "held-out loss (every 1k): $(grep eval_loss "$LOG" | sed -E 's/.*step ([0-9]+): eval_loss=([0-9.]+)/\1:\2/' | paste -sd' ')"
# evaluate checkpoints not yet in heldout_l1.json (1 GPU, debug partition)
done_ck=$(python -c "import json,sys; print(' '.join(json.load(open('$RUN/heldout_l1.json')).keys()))" 2>/dev/null)
new=0; for d in "$RUN"/checkpoints/[0-9]*/; do s=$((10#$(basename $d))); grep -qw "$s" <<<"$done_ck" || new=1; done
if [ $new = 1 ]; then
  srun -A tritondroids -p debug -w zixian --gres=rtx2080ti:1 -c 2 --mem=8G --time=00:20:00 python "$SCRIPTS/eval_heldout.py" "$RUN" 2>&1 | grep -E "^step|^best"
else
  python -c "
import json; d=json.load(open('$RUN/heldout_l1.json'))
for s in sorted(d,key=int): print(f\"step {int(s):6d}  L1(chunk) {d[s]['l1_chunk_mean']:.4f}  L1(first step) {d[s]['l1_first_step_mean']:.4f}  grip {d[s]['l1_chunk_per_dim']['grip']:.4f}\")"
fi
python "$SCRIPTS/plot_losses.py" "$RUN" "$LOG" >/dev/null 2>&1
