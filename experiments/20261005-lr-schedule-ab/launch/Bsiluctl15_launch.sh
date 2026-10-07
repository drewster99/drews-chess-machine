#!/bin/zsh
# B-silu-ctl15 (control for B-silu-clip1, 2026-10-06): same exact resume from step 18000 with the ORIGINAL grad_clip_max_norm 15 (parameters-B.json), to trainer step 23000 (segment step limit 5000) — tests whether the step-20,600 blowup reproduces without the cap.
set -u
S=${0:A:h}; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2331-4e70c615-p1headact.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $S/chain.log; }
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-lrBsilu-cyc1-replay-step18000.safetensors" --resume-exact \
  \
  --out-model "$M/20261006-lrBsilu-ctl15-replay-latest.safetensors" --parameters $E/parameters-B.json --epochs 12 \
  --training-step-limit 5000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 \
  > $E/train-Bsilu-ctl15.stdout 2>&1 &
P=$!
say "LRAB Bsilu-ctl15 started pid=$P"
sleep 30
PROBE_BIN="$BIN" PROBE_SEGMENT=1 TRAINER_PID=$P $R/experiments/probe_loop.sh 20261006-lrBsilu-ctl15 $E/probes-Bsilu-ctl15-seg1.jsonl > $S/lrab_probeBsiluctl15.out 2>&1 &
say "LRAB Bsilu-ctl15 probe loop started"
wait $P; say "LRAB Bsilu-ctl15 ended rc=$?"
wait
