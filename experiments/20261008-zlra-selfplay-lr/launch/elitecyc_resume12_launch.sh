#!/bin/zsh
# R-elitecyc segment 12: resume after the manual stop (SIGINT abort save at trainer step 361,096, epoch 2)
# and the reboot that followed — owner 2026-10-24: "restart the training". Resumes exactly from the newest enumerated checkpoint
# (highest step), continuing the same rolling out-model. --epochs 3 still counts from the lineage start
# (epoch 0 = the partial branch pass). Build 2523, as segments 1-11, so the resume stays EXACT.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"

# Pick the elitecyc enumerated checkpoint with the highest trainer step.
START=$(ls -1 "$M" | grep -E '^20261011-zlra-elitecyc-latest-step[0-9]+\.safetensors$' \
  | sed -E 's/.*step([0-9]+)\.safetensors/\1 &/' | sort -n | tail -1 | awk '{print $2}')
if [[ -z "$START" ]]; then echo "no elitecyc -step checkpoint found in $M" >&2; exit 1; fi
echo "resuming from $START" >&2

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/$START" \
  --resume-exact --parameters $E/parameters-zlra-elite-gentlecycle.json --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 3 --enumerate-checkpoints --overwrite-output --output $E/results-elitecyc-seg12.json \
  > $E/train-elitecyc-seg12.stdout 2>&1
