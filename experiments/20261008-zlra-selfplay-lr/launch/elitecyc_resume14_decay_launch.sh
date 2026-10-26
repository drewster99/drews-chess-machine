# R-elitecyc segment 15: exact resume from the 409,608 SIGINT save (owner stopped segment 14 on 2026-10-26),
# same decaying schedule (parameters-zlra-elitecyc-decay.json), so no inexact gap is expected.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step409608.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 5 --enumerate-checkpoints --output $E/results-elitecyc-seg14.json \
  > $E/train-elitecyc-seg14.stdout 2>&1
