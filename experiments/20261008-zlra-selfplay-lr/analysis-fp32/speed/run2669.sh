#!/bin/zsh
# fp32 vs bf16 training speed, one short corpus-replay run per arm, stopped by SIGINT after RUN_SEC.
set -u
S=${0:A:h}
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2669-e9d4f722.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
RUN_SEC=${RUN_SEC:-160}
arms=(
  "v3_16block_3x3-fp32-2669|$S/v3_16block_3x3-fp32.safetensors"
  "zlra9x9-fp32-2669|$M/20261112-zlra-elitecyc-fp32-start.safetensors"
)
for a in $arms; do
  name=${a%%|*}; start=${a#*|}
  echo "== $name $(date +%H:%M:%S)"
  "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$start" --parameters $S/params.json \
    --out-model $S/out-$name.safetensors --training-step-limit 100000 --output $S/results-$name.json \
    > $S/$name.stdout 2>&1 &
  pid=$!
  sleep $RUN_SEC
  kill -INT $pid 2>/dev/null
  wait $pid
  echo "   exit=$? $(date +%H:%M:%S)"
done
echo ALL DONE
