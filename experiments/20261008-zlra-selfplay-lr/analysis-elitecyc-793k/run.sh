#!/bin/bash
# Probe, describe and numerics-audit R-elitecyc checkpoints (same method as ../analysis-elitecyc-plateau).
set -u
B="/Users/andrew/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2620-07adcec1.app/Contents/MacOS/DrewsChessMachine"
M="/Users/andrew/Library/Application Support/DrewsChessMachine/Models"
cd "$(dirname "$0")"
for s in 500000 600000 700000 793000; do
  f="20261011-zlra-elitecyc-latest-step$s"
  "$B" --describe-model "$M/$f.safetensors" > $f.describe.json 2> $f.describe.err
  "$B" --probe-model "$M/$f.safetensors" --probe-set both --probe-out $f.probe.jsonl --probe-positions-out $f.positions.jsonl --probe-out-overwrite > $f.probe.stdout 2>&1
  "$B" --analyze-numerics "$M/$f.safetensors" --numerics-out numerics-$f > $f.numerics.stdout 2>&1
  echo "$f done $?"
done
