#!/bin/zsh
# R-elitecyc segment 2 (owner 2026-10-14: shut everything down for a macOS update, resume after): exact resume of
# segment 1's SIGINT abort save at trainer step 126,222 (20261011-zlra-elitecyc-latest-step126222.safetensors) on
# build 2523. The OS update changes the behavior fingerprint: if the run refuses with an `os` gap, add
# `--accept-inexact os` (the only expected gap).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step126222.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 1 --enumerate-checkpoints --output $E/results-elitecyc-seg2.json \
  > $E/train-elitecyc-seg2.stdout 2>&1
