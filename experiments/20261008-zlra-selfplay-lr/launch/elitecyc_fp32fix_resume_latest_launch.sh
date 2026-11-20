#!/bin/zsh
# R-elitecyc-fp32fix, every segment from 3 on (owner standing order 2026-11-19: keep fp32fix running, stopping
# for low battery or a conflicting GPU/RAM user and resuming afterwards). Exact resume of the highest-step
# enumerated save under the run's stem (an abort save writes one), on the same frozen build as segments 1-2, so
# the resume is EXACT with no accepted gaps. Each launch takes the next unused segment number for its
# results/stdout files. Same parameters, corpus, stem and target as segment 0 (elitecyc_fp32fix_launch.sh).
# Started and stopped by OpsScripts/fp32fix_supervisor.py; runnable by hand.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2678-2c1e8bb2.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
STEM=20261114-zlra-elitecyc-fp32fix-latest
TARGET_STEP=1000000

START_STEP=$(ls "$M" | sed -nE "s/^${STEM}-step([0-9]+)\.safetensors$/\1/p" | sort -n | tail -1)
[[ -n "$START_STEP" ]] || { echo "no ${STEM}-step*.safetensors in $M" >&2; exit 2; }
(( START_STEP < TARGET_STEP )) || { echo "already at ${START_STEP} >= ${TARGET_STEP}" >&2; exit 3; }
SEG=3
while [[ -e $E/train-elitecyc-fp32fix-seg${SEG}.stdout ]]; do (( SEG++ )); done

echo "segment ${SEG} from trainer step ${START_STEP}" >&2
exec "$BIN" --replay-corpus 20260704-215145-op24Gp \
  --start-model "$M/${STEM}-step${START_STEP}.safetensors" \
  --resume-exact \
  --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/${STEM}.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $E/results-elitecyc-fp32fix-seg${SEG}.json \
  > $E/train-elitecyc-fp32fix-seg${SEG}.stdout 2>&1
