#!/bin/zsh
# Segment 3 of the unlikelihood fork (owner 2026-11-27 "restart training"): exact resume of segment 2's final save at
# trainer step 714,776 (owner stop 2026-11-27), same frozen build, parameters and target (trainer step 1,000,000).
set -u
X=/Users/andrew/cursor/drews-chess-machine/experiments/20261123-unlikelihood-fork
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2701-e5e79ab2.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=714776
TARGET_STEP=1000000
ACCEPT=()
[[ -n "${DCM_ACCEPT_INEXACT:-}" ]] && ACCEPT=(--accept-inexact "$DCM_ACCEPT_INEXACT")
exec caffeinate -i "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261123-zlra-elitecyc-unlik-latest-step${START_STEP}.safetensors" \
  --resume-exact "${ACCEPT[@]}" \
  --parameters $X/parameters-unlikelihood.json \
  --out-model "$M/20261123-zlra-elitecyc-unlik-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $X/results-seg3.json \
  > $X/train-seg3.stdout 2>&1
