#!/bin/zsh
# R-elitecyc (owner 2026-10-11: "create a less intense cycle"): a branch of R-elite from its SIGINT save at trainer
# step 21,006 (20261011-zlra-elite-latest-step21006.safetensors) with a gentle LR cycle replacing the fixed 0.002:
# cosine 0.002 -> 0.02 -> 0.002, geometric, period 10k steps, not inverted, no decay, unbounded count; momentum
# fixed at 0.90, no sqrt-batch scaling. A branch, not --resume-exact: an exact resume restores the checkpoint's own
# schedule (fixed LR) and ignores the configured cycle ([REPLAY-RESUME] WARNING lr_momentum_cycle ... restoring the
# checkpoint's value, 21:34:58). The branch starts at trainer step 0 with zero velocity, warms up over 1,000 steps to
# the cycle's 0.002 trough, peaks at 0.02 at step 6,000, 16,000, ... It continues the elite corpus where R-elite
# stopped (nextGame=2162743, cold buffer refill). New stem so its step files can't collide with R-elite's. Build 2503.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2503-7d945ec7.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elite-latest-step21006.safetensors" \
  --start-game-index 2162743 \
  --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 1 --enumerate-checkpoints --output $E/results-elitecyc.json \
  > $E/train-elitecyc.stdout 2>&1
