#!/bin/zsh
# B-siluall segment 2 (owner 2026-10-09: keep the GPU busy and collect more data): exact resume of the 60,000-step final
# save, 40,000 more steps (to 100,000). Same build, corpus, epoch budget and --parameters file as segments 0 and 1.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-lrBsiluall-relcap-replay-step60000.safetensors" \
  --resume-exact --parameters $E/parameters-B-relcap-v3.json --out-model "$M/20261008-lrBsiluall-relcap-replay-latest.safetensors" \
  --epochs 12 --training-step-limit 40000 --enumerate-checkpoints --output $E/results-Bsiluall-seg2.json \
  > $E/train-Bsiluall-seg2.stdout 2>&1
