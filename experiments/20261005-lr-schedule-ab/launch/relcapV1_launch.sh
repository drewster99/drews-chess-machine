#!/bin/zsh
# Relative gradient cap V-1 (RELATIVE_GRADIENT_CAP_PLAN.md Part V): per-step measurement, the gate for k.
# Exact resume of B-silu from trainer step 18000, mode log only, k=1, floor 0.01 (lowest allowed, so the
# floor hides nothing above the median), to trainer step 21000. Log-only feeds the hard max (15), so the
# training math is B-silu's; every step above k x median writes a [GRAD-CLIP] ... applied=false line.
set -u
S=${0:A:h}; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$1"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $S/chain.log; }
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-lrBsilu-cyc1-replay-step18000.safetensors" --resume-exact --accept-inexact "$2" \
  --out-model "$M/20261007-lrBsilu-relcapV1-replay-latest.safetensors" --parameters $E/parameters-B-relcap-v1.json --epochs 12 \
  --training-step-limit 3000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 \
  > $E/train-Bsilu-relcapV1.stdout 2>&1 &
P=$!
say "LRAB relcapV1 started pid=$P"
sleep 30
PROBE_BIN="$BIN" PROBE_ABOVE_STEP=18000 TRAINER_PID=$P $R/experiments/probe_loop.sh 20261007-lrBsilu-relcapV1 $E/probes-Bsilu-relcapV1.jsonl > $S/lrab_probe_relcapV1.out 2>&1 &
say "LRAB relcapV1 probe loop started"
wait $P; say "LRAB relcapV1 ended rc=$?"
wait
