#!/bin/zsh
# Unlikelihood fork of R-elitecyc (owner 2026-11-22): exact resume of the main line's step-384,964 save with the
# main line's own decay schedule (parameters-zlra-elitecyc-decay.json) and ONE change: negative_advantage_policy_loss
# = unlikelihood (2) instead of complement_ce. bf16 like the main line. No control arm: R-elitecyc's own record from
# 384,964 on is the baseline (owner: GPU time is scarce). The step-384,964 file was written under the gentle cycle,
# so `schedule` is an accepted gap exactly as it was for the main line's segment 14; `params` is the loss change.
# Batches must match R-elitecyc's ([BATCH-HASH] at the same trainer steps). Runs to trainer step 1,000,000 or until
# the owner cancels.
set -u
X=/Users/andrew/cursor/drews-chess-machine/experiments/20261123-unlikelihood-fork
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2701-e5e79ab2.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=384964
TARGET_STEP=1000000
exec caffeinate -i "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step${START_STEP}.safetensors" \
  --resume-exact --accept-inexact "${DCM_ACCEPT_INEXACT:-schedule,params}" \
  --parameters $X/parameters-unlikelihood.json \
  --out-model "$M/20261123-zlra-elitecyc-unlik-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $X/results-seg0.json \
  > $X/train-seg0.stdout 2>&1
