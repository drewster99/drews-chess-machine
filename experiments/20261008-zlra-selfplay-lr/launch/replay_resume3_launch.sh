#!/bin/zsh
# R-replay segment 3 (owner 2026-10-10: R-replay keeps running well past 100k, "in perpetuity"): exact resume of
# segment 2's final save at trainer step 100,000, 900,000 more steps (to 1,000,000). Build 2491.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2491-6fe1dcd5.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-zlra-replay-step100000.safetensors" \
  --resume-exact --parameters $E/parameters-zlra.json --out-model "$M/20261008-zlra-replay-latest.safetensors" \
  --training-step-limit 900000 --enumerate-checkpoints --output $E/results-replay-seg3.json \
  > $E/train-replay-seg3.stdout 2>&1
