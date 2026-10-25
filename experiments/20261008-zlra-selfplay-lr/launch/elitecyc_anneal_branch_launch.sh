#!/bin/zsh
# R-elitecyc-anneal: branch of R-elitecyc from its abort save at trainer step 384,964 (model 20261025-1-qyUf),
# owner 2026-10-25. Tests why R-elitecyc plateaued (best wide pElo 1779-1786 since ~260k): the LR falls
# geometrically 0.002 -> 0.0002 over 20,000 steps, no cycle, then holds 0.0002 (momentum 0.90, warmup 1,000
# from zero velocity). Above 1786 = LR noise was the limit; flat = model capacity on this corpus.
# Feeds the elite corpus from where R-elitecyc stopped (epoch 2, game 10,696,971). Build 2620 (keyframes).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step384964.safetensors" \
  --start-game-index 10696971 \
  --parameters $E/parameters-zlra-elitecyc-anneal.json --out-model "$M/20261025-zlra-elitecyc-anneal-latest.safetensors" \
  --training-step-limit 30000 --enumerate-checkpoints --output $E/results-elitecyc-anneal.json \
  > $E/train-elitecyc-anneal.stdout 2>&1
