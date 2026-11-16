#!/bin/zsh
# R-elitecyc-fp32fix segment 2 (owner 2026-11-16: restart after the 11-14 crash): exact resume of segment 1's
# enumerated save at trainer step 400,000 (segment 1 died in the Mac crash at ~400,780), on a build that plans the fp32 conv split for batch 4096 and refuses larger
# batches (ConvolutionWeightGradientWorkaround; the 9x9 convolutions stay split, the 5x5 stem no longer is: same
# function). Same parameters, corpus and output stem as segment 0 (elitecyc_fp32fix_launch.sh).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/${DCM_FROZEN_BUILD:?set DCM_FROZEN_BUILD to the frozen app name}/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=400000
TARGET_STEP=1000000

accept=()
[ -n "${DCM_ACCEPT_INEXACT:-}" ] && accept=(--accept-inexact "$DCM_ACCEPT_INEXACT")
exec "$BIN" --replay-corpus 20260704-215145-op24Gp \
  --start-model "$M/20261114-zlra-elitecyc-fp32fix-latest-step400000.safetensors" \
  --resume-exact $accept \
  --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261114-zlra-elitecyc-fp32fix-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $E/results-elitecyc-fp32fix-seg2.json \
  > $E/train-elitecyc-fp32fix-seg2.stdout 2>&1
