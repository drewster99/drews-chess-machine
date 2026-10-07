#!/bin/zsh
# Relative gradient cap V-3 (RELATIVE_GRADIENT_CAP_PLAN.md Part V, owner-required): fresh start on B's recipe
# (B's own start net 20261005-r7b24-fresh, ReLU tower) with the cap in clip mode, k=3, N=1000, W=100, floor 0.5,
# 3000 steps: does the warm-up (hard max only for the first 100 steps) and the early relative cap leave
# early training intact? Compare against B's own first 3000 steps.
set -u
S=${0:A:h}; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$1"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $S/chain.log; }
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-r7b24-fresh.safetensors" \
  --out-model "$M/20261007-lrB-relcapV3-replay-latest.safetensors" --parameters $E/parameters-B-relcap-v3.json --epochs 12 \
  --training-step-limit 3000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 \
  > $E/train-B-relcapV3.stdout 2>&1 &
P=$!
say "LRAB relcapV3 started pid=$P"
sleep 30
PROBE_BIN="$BIN" TRAINER_PID=$P $R/experiments/probe_loop.sh 20261007-lrB-relcapV3 $E/probes-B-relcapV3.jsonl > $S/lrab_probe_relcapV3.out 2>&1 &
say "LRAB relcapV3 probe loop started"
wait $P; say "LRAB relcapV3 ended rc=$?"
wait
