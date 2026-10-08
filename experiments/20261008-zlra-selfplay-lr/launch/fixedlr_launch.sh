#!/bin/zsh
# R-fixedlr (owner 2026-10-08): GUI self-play, ZlrA's settings except LR fixed 0.01 and momentum fixed 0.90.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --train --start-model "$M/20261008-zlra-ab-fresh.safetensors" --parameters $E/parameters-selfplay-fixedlr.json \
  --seed 20261008 --output $E/results-fixedlr.json > $E/train-fixedlr.stdout 2>&1
