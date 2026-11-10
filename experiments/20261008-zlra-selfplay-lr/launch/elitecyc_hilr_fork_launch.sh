#!/bin/zsh
# R-elitecyc-hilr: the fork of R-elitecyc at trainer step 384,964, where the main line switched from the original
# cycle (parameters-zlra-elite-gentlecycle.json: geometric cosine 0.002 <-> 0.02, period 10,000, momentum 0.90) to the
# decaying keyframe schedule (parameters-zlra-elitecyc-decay.json; segment 14). This fork continues the ORIGINAL
# cycle from the same saved state (owner 2026-11-09: "fork off where we switched to the updated LR regime and in the
# fork, continue with the original"; no shuffle). Question: was the high-LR phase still paying when it was cut?
# Compare with the main line from 384,964 on (held-out NLL, large-set pElo, [DYNAMICS]).
#
# The step-384,964 file was written by segment 13 under the gentle cycle, so an exact resume with that parameter file
# restores the schedule unchanged. New rolling out-model stem so its enumerated checkpoints never collide with the
# main line's. Build 2665+ (per-layer [DYNAMICS] line). Target trainer step 1,000,000 like the main line; --epochs 7
# so the epoch budget (counted from the lineage start, ~148k steps per pass) does not stop it first.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="${DCM_BIN:?set DCM_BIN to the frozen build's executable}"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=384964
TARGET_STEP=1000000

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step${START_STEP}.safetensors" \
  --resume-exact ${DCM_ACCEPT_INEXACT:+--accept-inexact "$DCM_ACCEPT_INEXACT"} \
  --parameters $E/parameters-zlra-elite-gentlecycle.json \
  --out-model "$M/20261109-zlra-elitecyc-hilr-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $E/results-elitecyc-hilr-seg0.json \
  > $E/train-elitecyc-hilr-seg0.stdout 2>&1
