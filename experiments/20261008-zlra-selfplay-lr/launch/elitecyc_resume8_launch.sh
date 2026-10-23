#!/bin/zsh
# R-elitecyc segment 8: resume after the 2026-10-16 ~20:50 pause (owner: "pause chess trainer, restart in 5 hours").
# The paused run's SIGINT abort save landed near trainer step ~194,4xx, written to both the rolling
# 20261011-zlra-elitecyc-latest.safetensors and its enumerated -step<N> copy. This resumes exactly from the
# newest enumerated checkpoint (highest step number), continuing the same rolling out-model. --epochs 3 still
# counts from the lineage start (epoch 0 = the partial branch pass). Build 2523, as segments 1-3.
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
  --epochs 3 --enumerate-checkpoints --overwrite-output --output $E/results-elitecyc-seg8.json \
  > $E/train-elitecyc-seg8.stdout 2>&1
