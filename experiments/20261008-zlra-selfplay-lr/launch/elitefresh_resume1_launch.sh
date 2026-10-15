#!/bin/zsh
# R-elitefresh segment 1 (owner 2026-10-14: shut everything down for a macOS update, resume after): exact resume of
# segment 0's SIGINT abort save at trainer step 80,518 (20261012-zlra-elitefresh-latest-step80518.safetensors) on
# build 2523, same parameters file. The seed continues from the checkpoint's lineage (no --seed on a resume). The
# OS update changes the behavior fingerprint: if the run refuses with an `os` gap, add `--accept-inexact os`.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261012-zlra-elitefresh-latest-step80518.safetensors" \
  --resume-exact --parameters $E/parameters-zlra.json --out-model "$M/20261012-zlra-elitefresh-latest.safetensors" \
  --epochs 1 --enumerate-checkpoints --output $E/results-elitefresh-seg1.json \
  > $E/train-elitefresh-seg1.stdout 2>&1
