#!/bin/zsh
# Zero-training self-play speed smoke: fp32 / bf16 / fp16, one at a time.
# Each arm launches GUI --train (self-play only; min-positions > capacity so
# the trainer never steps), activates the app so the window attaches and
# self-play starts, runs ~4 min, then reads the [PREFILL] heartbeat.
set -u
DIR=/Users/andrew/cursor/drews-chess-machine/experiments/20261019-inference-dtype-speed
BIN=/Users/andrew/Library/Developer/Xcode/DerivedData/DrewsChessMachine-eyigcdvyyrcsakaqcybzcfgsurbr/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
PARAMS=$DIR/params
MEASURE=240   # seconds of steady-state measurement after activation warmup

run_arm() {
  tag=$1; model=$2
  # make sure nothing else holds the app
  pkill -x DrewsChessMachine 2>/dev/null; sleep 3
  before=$(date +%s)
  "$BIN" --train --start-model "$model" --parameters "$PARAMS" \
    --output "$DIR/arm-$tag.json" --overwrite-output --seed 20261019 \
    > "$DIR/arm-$tag.stdout" 2>&1 &
  pid=$!
  # activate repeatedly across the first ~35s so the window attaches
  for d in 2 3 5 10 15; do sleep $d; /usr/bin/osascript -e 'tell application "Drew'\''s Chess Machine" to activate' >/dev/null 2>&1; done
  sleep $MEASURE
  log=$(find ~/Library/Logs/DrewsChessMachine -name 'dcm_log_*.txt' -newermt "@$before" 2>/dev/null | while read f; do grep -ql 'build=2576' "$f" && echo "$f"; done | head -1)
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 20); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; sleep 2
  echo "================ ARM $tag  model=$(basename $model) ================"
  echo "log=$log"
  if [ -n "$log" ]; then
    echo "--- [PREFILL] heartbeat (last 10) ---"; grep '\[PREFILL\]' "$log" | tail -10
    echo "--- self-play start / champion arch / dtype ---"; grep -nE 'start model loaded as champion|champion.*float|compute_data_type|\[RUN\]' "$log" | head -4
    echo "--- blow-up / errors ---"; grep -iE 'non-finite|nonFinite|\[ALARM\]|error|fatal|refused' "$log" | tail -5
  else
    echo "NO LOG FOUND for build 2576 after $before"
  fi
  echo
}

run_arm fp32 "$DIR/models/zlra-speed-fp32.safetensors"
run_arm bf16 "$DIR/models/zlra-speed-base.safetensors"
run_arm fp16 "$DIR/models/zlra-speed-fp16.safetensors"
pkill -x DrewsChessMachine 2>/dev/null
echo "ALL ARMS DONE"
