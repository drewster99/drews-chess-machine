# R-elitecyc segment 13: exact resume from the 384,964 abort save with the decaying schedule
# (parameters-zlra-elitecyc-decay.json: cycle 0.02/0.002 to 386k; peak -> 0.001 by 486k; trough -> 0.0001 by 586k).
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step384964.safetensors" \
  --resume-exact --accept-inexact schedule --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 5 --enumerate-checkpoints --output $E/results-elitecyc-seg13.json \
  > $E/train-elitecyc-seg13.stdout 2>&1
