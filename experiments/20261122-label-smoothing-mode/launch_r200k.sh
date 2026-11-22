#!/bin/zsh
# Segment 1 of the label-smoothing arms (owner 2026-11-22): replay buffer 1,000,000 -> 200,000 positions to cut
# memory (min fill before training 500,000 -> 100,000, same half ratio). Every arm resumes exactly from its own
# step-9,000 checkpoint (the last step all three wrote before they were stopped at 9,733 / 9,875 / 9,833), so all three still train on the same batches. A new out-model stem (-r200k) keeps the
# first segment's files and its abort saves out of the way. Runs on frozen build 2695 (single training executable, streamed shards), so build is an accepted gap. Usage: launch_r200k.sh A|B|C
set -u
ARM=${1:?arm A, B or C}
X=/Users/andrew/cursor/drews-chess-machine/experiments/20261122-label-smoothing-mode
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2695-8e889ffd.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec caffeinate -i "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261122-lsmode-$ARM-latest-step9000.safetensors" \
  --resume-exact --accept-inexact "${DCM_ACCEPT_INEXACT:-params,build}" --parameters $X/parameters-$ARM-r200k.json \
  --out-model "$M/20261122-lsmode-$ARM-r200k-latest.safetensors" \
  --training-step-limit 31000 --enumerate-checkpoints --output $X/results-$ARM-r200k.json \
  > $X/train-$ARM-r200k.stdout 2>&1
