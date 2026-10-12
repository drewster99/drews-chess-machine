#!/bin/zsh
# R-elitefresh (owner 2026-10-12: "an elite corpus run with same params as R-replay", from scratch, beside
# R-elitecyc): R-replay's own start file (20261008-zlra-ab-fresh.safetensors, an untrained network), parameter
# file (parameters-zlra.json: inverted LR cycle 0.001-1, period 10k, decaying to 0.001/1e-6 over 1M steps,
# momentum following) and seed (20261008), fed the elite corpus (op24Gp) instead of w3aA5b. One epoch.
# The old-form schedule keys in parameters-zlra.json convert exactly to keyframes on load (build 2523).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261008-zlra-ab-fresh.safetensors" \
  --out-model "$M/20261012-zlra-elitefresh-latest.safetensors" --parameters $E/parameters-zlra.json \
  --epochs 1 --enumerate-checkpoints --seed 20261008 --output $E/results-elitefresh.json \
  > $E/train-elitefresh.stdout 2>&1
