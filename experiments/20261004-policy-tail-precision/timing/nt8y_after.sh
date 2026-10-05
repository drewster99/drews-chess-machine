#!/bin/zsh
# After the tail A-B-B-A bench ends, time 600 steps of nt8y's last checkpoint alone on the GPU (fp32 tail, same params).
B=${0:A:h}
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models/20260701-nT8Y-resume4-replay-latest.safetensors"
grep -q NT8Y_DONE $B/runs.txt 2>/dev/null && { echo "$B/runs.txt already records an nt8y run; move it aside to re-run" >&2; exit 2; }
until grep -q -E "BENCH_DONE|ABORT" $B/runs.txt 2>/dev/null; do sleep 20; done
if pgrep -f "DrewsChessMachine --(replay-corpus|train|probe-model)" >/dev/null; then echo "ABORT nt8y: another job is running" >> $B/runs.txt; exit 2; fi
start=$(date +%s)
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M" --out-model $B/nt8y-fp32.safetensors \
  --parameters $B/bench_params.json --training-step-limit 600 --policy-tail-precision fp32_from_pre_bn > $B/nt8y-fp32.stdout 2>&1; rc=$?
log=$(ls -t ~/Library/Logs/DrewsChessMachine/dcm_log_*.txt | head -1)
echo "nt8y-fp32 rc=$rc secs=$(( $(date +%s)-start )) tail=fp32_from_pre_bn power=$(pmset -g batt | head -1 | grep -o "'.*'") log=$log" >> $B/runs.txt
echo NT8Y_DONE >> $B/runs.txt
