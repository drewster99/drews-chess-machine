#!/bin/zsh
# R-elitefresh segment 2: resume after the pause at trainer step ~111,327 (owner 2026-10-17: "continue training
# runs"). Segment 1 stopped on a SIGINT abort inside epoch 0 (done: epochs=0), so the epoch budget was not used.
# This resumes exactly from the newest enumerated elitefresh checkpoint (highest step), continuing the same rolling
# out-model. --epochs 3 matches R-elitecyc seg4's horizon so the cycle-vs-fresh A/B runs over equal corpus passes.
# Build 2523, parameters-zlra.json (fixed LR), as segments 0-1. No --seed on a resume (seed continues from lineage).
# If it refuses with an `os` gap, add `--accept-inexact os`.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"

# Pick the elitefresh enumerated checkpoint with the highest trainer step.
START=$(ls -1 "$M" | grep -E '^20261012-zlra-elitefresh-latest-step[0-9]+\.safetensors$' \
  | sed -E 's/.*step([0-9]+)\.safetensors/\1 &/' | sort -n | tail -1 | awk '{print $2}')
if [[ -z "$START" ]]; then echo "no elitefresh -step checkpoint found in $M" >&2; exit 1; fi
echo "resuming from $START" >&2

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/$START" \
  --resume-exact --parameters $E/parameters-zlra.json --out-model "$M/20261012-zlra-elitefresh-latest.safetensors" \
  --epochs 3 --enumerate-checkpoints --overwrite-output --output $E/results-elitefresh-seg2.json \
  > $E/train-elitefresh-seg2.stdout 2>&1
