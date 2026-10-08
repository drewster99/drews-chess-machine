#!/bin/zsh
# B-siluall (owner 2026-10-08): B-silu with SiLU in the heads too (policy, value conv, value FC1), on B's LR cycle with the
# relative gradient cap (k = 3, clip). Start net's 61 tensors are byte-identical to B-silu's start.
set -u
S=${0:A:h}; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261008-r7b24-silu-all-fresh.safetensors" \
  --out-model "$M/20261008-lrBsiluall-relcap-replay-latest.safetensors" --parameters $E/parameters-B-relcap-v3.json \
  --epochs 12 --training-step-limit 40000 --enumerate-checkpoints --seed 20261005 --output $E/results-Bsiluall.json \
  > $E/train-Bsiluall.stdout 2>&1
