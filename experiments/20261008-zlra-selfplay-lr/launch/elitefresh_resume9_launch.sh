#!/bin/zsh
# R-elitefresh segment 9: resume after the manual stop of 2026-10-20 07:30 (abort save at trainer step 196,491,
# epoch 1) — owner 2026-10-23: "continue both runs". Resumes exactly from the newest enumerated elitefresh
# checkpoint (highest step), continuing the same rolling out-model. --epochs 3 as segment 8. Build 2523,
# parameters-zlra.json (fixed LR), as segments 0-8. No --seed on a resume (the seed continues from the lineage).
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
  --epochs 3 --enumerate-checkpoints --overwrite-output --output $E/results-elitefresh-seg9.json \
  > $E/train-elitefresh-seg9.stdout 2>&1
