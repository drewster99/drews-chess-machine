# R-elitecyc segment 17: exact resume from the 604,000 checkpoint (segment 16's last save, 18:35 on 2026-10-29; the
# process died without a termination line ~19:10 after a sleep/wake on battery). Same build and decaying schedule
# (parameters-zlra-elitecyc-decay.json) as segment 16, so no inexact gap is expected. Ends by itself after epoch 4.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261011-zlra-elitecyc-latest-step604000.safetensors" \
  --resume-exact --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261011-zlra-elitecyc-latest.safetensors" \
  --epochs 5 --enumerate-checkpoints --output $E/results-elitecyc-seg16.json \
  > $E/train-elitecyc-seg16.stdout 2>&1
