#!/bin/zsh
# B-silu-clip2 (2026-10-07): exact resume of B-silu from its step-18000 checkpoint with grad_clip_max_norm 2 (B-silu used 15), to trainer step 23000 (segment step limit 5000): does a looser fixed cap still prevent the step-20,600 blowup?
set -u
S=${0:A:h}; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2331-4e70c615-p1headact.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $S/chain.log; }
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-lrBsilu-cyc1-replay-step18000.safetensors" --resume-exact --accept-inexact params \
  \
  --out-model "$M/20261006-lrBsilu-clip2-replay-latest.safetensors" --parameters $E/parameters-Bsilu-clip2.json --epochs 12 \
  --training-step-limit 5000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 \
  > $E/train-Bsilu-clip2.stdout 2>&1 &
P=$!
say "LRAB Bsilu-clip2 started pid=$P"
sleep 30
PROBE_BIN="$BIN" PROBE_SEGMENT=1 TRAINER_PID=$P $R/experiments/probe_loop.sh 20261006-lrBsilu-clip2 $E/probes-Bsilu-clip2-seg1.jsonl > $S/lrab_probeBsiluclip2.out 2>&1 &
say "LRAB Bsilu-clip2 probe loop started"
wait $P; say "LRAB Bsilu-clip2 ended rc=$?"
wait
