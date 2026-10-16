#!/bin/zsh
# R-elitecyc segment 3, resume after the 2026-10-15 18:25 pause (owner: "stop everything but the lichess bot").
# The paused run's abort save landed at trainer step 134,247 (epoch 1, next corpus game 649,792), written to both
# the rolling 20261011-zlra-elitecyc-latest.safetensors and the enumerated -step134247 copy. This resumes exactly
# from that enumerated checkpoint, continuing the same rolling out-model. --epochs 3 still counts from the lineage
# start (epoch 0 = the partial branch pass), so it trains through the end of epoch 2. Build 2523, as segments 1-3.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step134247.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 3 --enumerate-checkpoints --overwrite-output --output $E/results-elitecyc-seg3.json \
  > $E/train-elitecyc-seg3.stdout 2>&1
