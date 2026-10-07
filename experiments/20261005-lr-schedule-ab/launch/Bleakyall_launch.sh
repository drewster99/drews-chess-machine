#!/bin/zsh
# Bleakyall (owner 2026-10-05): arm B's recipe with leaky ReLU at the value-head conv and value FC1 hidden layer.
set -u
S=${0:A:h}; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2331-4e70c615-p1headact.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $S/chain.log; }
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-r7b24-leakyall-fresh.safetensors" \
  --out-model "$M/20261005-lrBleakyall-cyc1-replay-latest.safetensors" --parameters $E/parameters-B.json --epochs 12 \
  --training-step-limit 40000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 \
  > $E/train-Bleakyall.stdout 2>&1 &
P=$!
say "LRAB Bleakyall started pid=$P"
sleep 30
PROBE_BIN="$BIN" TRAINER_PID=$P $R/experiments/probe_loop.sh 20261005-lrBleakyall-cyc1 $E/probes-Bleakyall.jsonl > $S/lrab_probeBleakyall.out 2>&1 &
say "LRAB Bleakyall probe loop started"
wait $P; say "LRAB Bleakyall ended rc=$?"
wait
