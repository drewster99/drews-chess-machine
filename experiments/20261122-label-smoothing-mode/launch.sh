#!/bin/zsh
# Policy label smoothing mode 0 vs mode 1 (+ no smoothing), owner 2026-11-22. One arm per call: launch.sh A|B|C.
# All three arms start from the same fresh net and seed, so they train on identical batches (compare [BATCH-HASH]);
# the only difference is the policy label smoothing (parameters-<arm>.json). README.md has the design.
set -u
ARM=${1:?arm A, B or C}
X=/Users/andrew/cursor/drews-chess-machine/experiments/20261122-label-smoothing-mode
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2687-d8afa524.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec caffeinate -i "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261122-lsmode-fresh.safetensors" \
  --seed 20261122 --parameters $X/parameters-$ARM.json \
  --out-model "$M/20261122-lsmode-$ARM-latest.safetensors" \
  --training-step-limit 40000 --enumerate-checkpoints --output $X/results-$ARM.json \
  > $X/train-$ARM.stdout 2>&1
