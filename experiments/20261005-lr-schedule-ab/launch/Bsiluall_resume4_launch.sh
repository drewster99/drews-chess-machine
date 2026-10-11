#!/bin/zsh
# B-siluall segment 4: segment 3 stopped on the 2026-10-10 21:15:26 GPU reset (exit 36, R-fixedlr's arena hang; no
# save after it). Exact resume of its last save, trainer step 74,000, for 26,000 more steps (to 100,000). Build 2491.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2491-6fe1dcd5.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-lrBsiluall-relcap-replay-step74000.safetensors" \
  --resume-exact --parameters $E/parameters-B-relcap-v3.json --out-model "$M/20261008-lrBsiluall-relcap-replay-latest.safetensors" \
  --epochs 12 --training-step-limit 26000 --enumerate-checkpoints --output $E/results-Bsiluall-seg4.json \
  > $E/train-Bsiluall-seg4.stdout 2>&1
