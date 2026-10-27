#!/bin/zsh
# bf16 + fp16 self-play speed arms, launched from a real GUI session (window renders).
# fp32 already captured. Each: launch, activate (belt+suspenders), run 200s, stop.
set -u
DIR=/Users/andrew/cursor/drews-chess-machine/experiments/20261019-inference-dtype-speed
BIN=/Users/andrew/Library/Developer/Xcode/DerivedData/DrewsChessMachine-eyigcdvyyrcsakaqcybzcfgsurbr/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
rm -f "$DIR/smoke2.done"
run() {
  tag=$1; model=$2
  pkill -x DrewsChessMachine 2>/dev/null; sleep 3
  "$BIN" --train --start-model "$DIR/models/$model" --parameters "$DIR/params" \
    --output "$DIR/arm-$tag.json" --overwrite-output --seed 20261019 > "$DIR/arm-$tag.stdout" 2>&1 &
  for d in 2 3 5 10; do sleep $d; /usr/bin/osascript -e 'tell application "Drew'\''s Chess Machine" to activate' >/dev/null 2>&1; done
  sleep 180
  pkill -x DrewsChessMachine 2>/dev/null; sleep 3
  echo "$tag done"
}
run bf16 zlra-speed-base.safetensors
run fp16 zlra-speed-fp16.safetensors
echo DONE > "$DIR/smoke2.done"
echo "ALL DONE"
