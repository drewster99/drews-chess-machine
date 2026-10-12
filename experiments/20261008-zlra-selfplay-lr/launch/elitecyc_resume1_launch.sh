#!/bin/zsh
# R-elitecyc segment 1 (owner 2026-10-12: shut down gently, rebuild, resume): exact resume of segment 0's SIGINT
# abort save at trainer step 45,467 (20261011-zlra-elitecyc-latest-step45467.safetensors, nextGame=6663897) on
# build 2523 (LR keyframes). Same parameters file; its old-form cycle keys convert exactly to keyframes, and the
# checkpoint's schedule converts too, so the configured schedule matches (no `schedule` gap expected).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step45467.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 1 --enumerate-checkpoints --output $E/results-elitecyc-seg1.json \
  > $E/train-elitecyc-seg1.stdout 2>&1
