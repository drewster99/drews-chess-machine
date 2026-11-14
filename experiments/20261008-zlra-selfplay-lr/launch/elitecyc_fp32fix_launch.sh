#!/bin/zsh
# R-elitecyc-fp32fix (owner 2026-11-14): R-elitecyc-fp32 restarted from the same fp32 start state (trainer step
# 384,964, derived from 20261011-zlra-elitecyc-latest-step384964.safetensors) on a build with the fp32 conv
# weight-gradient workaround (ConvolutionWeightGradientWorkaround). The first fp32 run (build 2669, stopped at 413,778)
# trained on MPSGraph's wrong fp32 weight gradient for every block's conv1 (analysis-fp32/README.md); its files keep
# their stem (20261112-zlra-elitecyc-fp32-latest-*), this run writes under a new one. Same parameters, corpus, feed and
# accepted gaps as elitecyc_fp32_launch.sh.
# Question unchanged: does fp32 compute change where the annealed line ends up? Compare with R-elitecyc (bf16).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/${DCM_FROZEN_BUILD:?set DCM_FROZEN_BUILD to the frozen app name}/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=384964
TARGET_STEP=1000000

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261112-zlra-elitecyc-fp32-start.safetensors" \
  --resume-exact --accept-inexact "${DCM_ACCEPT_INEXACT:-schedule,compute_data_type,policy_tail_precision}" \
  --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261114-zlra-elitecyc-fp32fix-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $E/results-elitecyc-fp32fix-seg0.json \
  > $E/train-elitecyc-fp32fix-seg0.stdout 2>&1
