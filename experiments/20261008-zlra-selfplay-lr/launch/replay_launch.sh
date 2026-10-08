#!/bin/zsh
# R-replay (owner 2026-10-08): ZlrA's architecture and parameter snapshot on corpus replay.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-zlra-ab-fresh.safetensors" \
  --out-model "$M/20261008-zlra-replay-latest.safetensors" --parameters $E/parameters-zlra.json \
  --training-step-limit 60000 --enumerate-checkpoints --seed 20261008 --output $E/results-replay.json \
  > $E/train-replay.stdout 2>&1
