#!/bin/zsh
# B-siluall segment 1 (owner 2026-10-09: "keep them all running"; 60k budget OK): exact resume of the 40,000-step final
# save, 20,000 more steps (to 60,000, R-replay's budget). Same build, corpus, epoch budget and --parameters file as
# segment 0: without it the CLI takes the app's saved settings (the GUI run's), and the resume refuses (params).
# Seed and random streams come from the checkpoint's lineage record.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-lrBsiluall-relcap-replay-step40000.safetensors" \
  --resume-exact --parameters $E/parameters-B-relcap-v3.json --out-model "$M/20261008-lrBsiluall-relcap-replay-latest.safetensors" \
  --epochs 12 --training-step-limit 20000 --enumerate-checkpoints --output $E/results-Bsiluall-seg1.json \
  > $E/train-Bsiluall-seg1.stdout 2>&1
