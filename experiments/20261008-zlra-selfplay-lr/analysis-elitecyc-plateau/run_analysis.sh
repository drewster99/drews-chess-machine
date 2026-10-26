# Full analysis of R-elitecyc checkpoints around the plateau, with the anneal's end as reference.
set -u
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
O=/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr/analysis-elitecyc-plateau
for f in 20261011-zlra-elitecyc-latest-step250000 20261011-zlra-elitecyc-latest-step291000 20261011-zlra-elitecyc-latest-step384964 20261025-zlra-elitecyc-anneal-latest-step30000; do
  "$BIN" --describe-model "$M/$f.safetensors" > "$O/$f.describe.json" 2> "$O/$f.describe.err"
  "$BIN" --probe-model "$M/$f.safetensors" --probe-set both --probe-out "$O/$f.probe.jsonl" --probe-positions-out "$O/$f.positions.jsonl" > "$O/$f.probe.stdout" 2>&1
  echo "$f probe exit $?"
  mkdir -p "$O/numerics-$f"
  "$BIN" --analyze-numerics "$M/$f.safetensors" --numerics-out "$O/numerics-$f" > "$O/$f.numerics.stdout" 2>&1
  echo "$f numerics exit $?"
done
