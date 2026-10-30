#!/bin/zsh
# R-elitefresh segment 11: resume after segment 10 died without a termination line ~19:10 on 2026-10-29 (sleep/wake on
# battery; last save trainer step 278,000 at 18:37). Owner 2026-10-28: "continue elitefresh ... to 700k steps". Resumes exactly
# from the newest enumerated elitefresh checkpoint (highest step), continuing the same rolling out-model. Build 2523 and
# parameters-zlra.json, as segments 0-10. --epochs 5 lifts the 3-pass budget so the run can reach 700,000 (a pass is
# ~149,000 steps); --training-step-limit counts this segment's steps, set so the run stops at trainer step 700,000.
# No --seed on a resume (the seed continues from the lineage).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2523-a242b7ad.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
TARGET_STEP=700000

# Pick the elitefresh enumerated checkpoint with the highest trainer step.
START_STEP=$(ls -1 "$M" | grep -E '^20261012-zlra-elitefresh-latest-step[0-9]+\.safetensors$' \
  | sed -E 's/.*step([0-9]+)\.safetensors/\1/' | sort -n | tail -1)
if [[ -z "$START_STEP" ]]; then echo "no elitefresh -step checkpoint found in $M" >&2; exit 1; fi
if (( START_STEP >= TARGET_STEP )); then echo "already at step $START_STEP (target $TARGET_STEP)" >&2; exit 1; fi
START="20261012-zlra-elitefresh-latest-step${START_STEP}.safetensors"
echo "resuming from $START, step limit $(( TARGET_STEP - START_STEP ))" >&2

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/$START" \
  --resume-exact --parameters $E/parameters-zlra.json --out-model "$M/20261012-zlra-elitefresh-latest.safetensors" \
  --epochs 5 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --overwrite-output --output $E/results-elitefresh-seg12.json \
  > $E/train-elitefresh-seg12.stdout 2>&1
