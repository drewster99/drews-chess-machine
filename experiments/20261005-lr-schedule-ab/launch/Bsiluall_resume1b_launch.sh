#!/bin/zsh
# B-siluall segment 1b: segment 1 halted at trainer step ~45,975 on a NaN gradient with finite losses (LR 0.001,
# gNorm 0.22, no divergence; no save was written after it). Exact resume of the clean 45,000-step autosave,
# 15,000 more steps (to 60,000). Same build, corpus, epoch budget and --parameters file as segments 0 and 1.
# The sampler stream continues from the checkpoint, so step ~45,975 sees the same batch again: a repeat NaN there
# would point at the data or weights, a clean pass at a transient GPU fault.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-lrBsiluall-relcap-replay-step45000.safetensors" \
  --resume-exact --parameters $E/parameters-B-relcap-v3.json --out-model "$M/20261008-lrBsiluall-relcap-replay-latest.safetensors" \
  --epochs 12 --training-step-limit 15000 --enumerate-checkpoints --output $E/results-Bsiluall-seg1b.json \
  > $E/train-Bsiluall-seg1b.stdout 2>&1
