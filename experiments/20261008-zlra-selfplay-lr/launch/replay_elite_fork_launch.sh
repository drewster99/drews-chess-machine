#!/bin/zsh
# R-elite: a branch of R-replay onto the Lichess Elite corpus (owner 2026-10-11: test whether the May corpus's
# ~1640-mean teachers cap R-replay at ~1610-1650 pElo). Starts from R-replay's SIGINT save at trainer step 86,398
# (20261008-zlra-replay-step86398.safetensors) as a new branch (no --resume-exact: new lineage, new files; the
# Lichess bot keeps following R-replay). Corpus 20260704-215145-op24Gp (14.6M games, mean 2568). Parameters: R-replay's,
# except a fixed LR of 0.002 and momentum 0.90 (no LR/momentum cycle, no sqrt-batch scaling) so the branch isn't
# thrown back by a fresh cycle peak. One full pass over the corpus (--epochs 1). Build 2501.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2501-eae1b55c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261008-zlra-replay-step86398.safetensors" \
  --parameters $E/parameters-zlra-elite-fixedlr.json --out-model "$M/20261011-zlra-elite-latest.safetensors" \
  --epochs 1 --enumerate-checkpoints --output $E/results-elite.json \
  > $E/train-elite.stdout 2>&1
