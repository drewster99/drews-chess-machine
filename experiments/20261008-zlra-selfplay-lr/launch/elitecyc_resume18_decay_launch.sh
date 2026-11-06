#!/bin/zsh
# R-elitecyc segment 19: exact resume from the 721,584 abort save (segment 18 stopped by SIGINT at 10:15 on 2026-11-02,
# owner: "stop this run and continue the fixedlr self-play"; owner 2026-11-06: "continue the elitecyc"). Continues to trainer step 1,000,000 (owner
# 2026-10-31). Same build, decaying schedule and rolling out-model as segments 17-18. --epochs 7 so the epoch budget (counted
# from the lineage start; one pass is about 148k steps) does not stop it first; --training-step-limit counts this
# segment's steps, so the run stops at trainer step 1,000,000. No --seed on a resume (the seed continues from the lineage).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=721584
TARGET_STEP=1000000

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step${START_STEP}.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $E/results-elitecyc-seg18.json \
  > $E/train-elitecyc-seg18.stdout 2>&1
