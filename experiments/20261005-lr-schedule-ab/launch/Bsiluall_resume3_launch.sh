#!/bin/zsh
# B-siluall segment 3 (owner 2026-10-10: move every trainer to the build with GPU submission labels and the flight
# recorder): exact resume of segment 2's abort save at trainer step 64,071 (SIGINT), 35,929 more steps (to 100,000).
# Same corpus, epoch budget and --parameters file as segments 0-2; build 2491 (was 2440).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2491-6fe1dcd5.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-lrBsiluall-relcap-replay-step64071.safetensors" \
  --resume-exact --parameters $E/parameters-B-relcap-v3.json --out-model "$M/20261008-lrBsiluall-relcap-replay-latest.safetensors" \
  --epochs 12 --training-step-limit 35929 --enumerate-checkpoints --output $E/results-Bsiluall-seg3.json \
  > $E/train-Bsiluall-seg3.stdout 2>&1
