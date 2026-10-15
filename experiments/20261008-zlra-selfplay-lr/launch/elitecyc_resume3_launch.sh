#!/bin/zsh
# R-elitecyc segment 3 (owner 2026-10-15: "another round of elite corpus through elitecyc ... let's actually do 2 more"):
# exact resume of segment 2's final save at trainer step 127,931 (20261011-zlra-elitecyc-latest-step127931.safetensors,
# corpus position epoch 1, next game 0) on build 2523, as segments 1-2. --epochs counts from the start of the lineage:
# epoch 0 was the partial pass from game 2,162,743 (the branch point), so --epochs 3 trains two more full passes
# (epochs 1 and 2) and stops at the end of epoch 2.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step127931.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 3 --enumerate-checkpoints --output $E/results-elitecyc-seg3.json \
  > $E/train-elitecyc-seg3.stdout 2>&1
