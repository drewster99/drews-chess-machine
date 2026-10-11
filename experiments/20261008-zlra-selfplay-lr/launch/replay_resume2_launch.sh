#!/bin/zsh
# R-replay segment 2: segment 1 stopped on the 2026-10-10 21:15:26 GPU reset (exit 36, R-fixedlr's arena hang; no
# save after it). Exact resume of its last save, trainer step 67,000, for 33,000 more steps (to 100,000). Build 2491.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2491-6fe1dcd5.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-zlra-replay-step67000.safetensors" \
  --resume-exact --parameters $E/parameters-zlra.json --out-model "$M/20261008-zlra-replay-latest.safetensors" \
  --training-step-limit 33000 --enumerate-checkpoints --output $E/results-replay-seg2.json \
  > $E/train-replay-seg2.stdout 2>&1
