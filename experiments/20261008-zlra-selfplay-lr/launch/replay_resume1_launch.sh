#!/bin/zsh
# R-replay segment 1 (owner 2026-10-09: keep the GPU busy and collect more data): exact resume of the 60,000-step final
# save, 40,000 more steps (to 100,000). Same build, corpus and --parameters file as segment 0 (without it the CLI takes
# the app's saved GUI settings and the resume refuses). Seed and random streams come from the checkpoint's lineage record.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-zlra-replay-step60000.safetensors" \
  --resume-exact --parameters $E/parameters-zlra.json --out-model "$M/20261008-zlra-replay-latest.safetensors" \
  --training-step-limit 40000 --enumerate-checkpoints --output $E/results-replay-seg1.json \
  > $E/train-replay-seg1.stdout 2>&1
