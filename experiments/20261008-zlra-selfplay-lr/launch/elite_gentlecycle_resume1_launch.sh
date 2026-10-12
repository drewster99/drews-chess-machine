#!/bin/zsh
# R-elite segment 1 (owner 2026-10-11: "create a less intense cycle"). Exact resume of R-elite from its SIGINT save
# just after trainer step 21,000, with a gentle LR cycle replacing the fixed 0.002: cosine 0.002 -> 0.02 -> 0.002,
# geometric, period 10k steps, not inverted, no decay, unbounded count; momentum fixed at 0.90, no sqrt-batch
# scaling. The cycle's phase is (trainer step - 1,000 warmup) mod 10k, so at step 21,000 it sits at its 0.002
# trough: the LR continues without a jump and peaks at 0.02 at step 26,000, 36,000, ... Build 2503 (GPU fault
# monitor fix). Gaps accepted: params (the LR change), build (2501 -> 2503).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2503-7d945ec7.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elite-latest.safetensors" \
  --resume-exact --accept-inexact params,build \
  --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elite-latest.safetensors" \
  --epochs 1 --enumerate-checkpoints --output $E/results-elite-seg1.json \
  > $E/train-elite-seg1.stdout 2>&1
