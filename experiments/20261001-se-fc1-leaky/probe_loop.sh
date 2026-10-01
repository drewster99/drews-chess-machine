#!/bin/zsh
# Probe every enumerated leaky-FC1 checkpoint (every 1000 steps) with --probe-set wide,
# appending one JSON line per checkpoint to probes.jsonl. Exits when the run ends and
# every checkpoint is probed.
E=${0:A:h}
BIN=/private/tmp/claude-501/-Users-andrew-cursor-drews-chess-machine/844f7374-7702-404b-9d25-5c985cb68170/scratchpad/DCM-de0f22b.app/Contents/MacOS/DrewsChessMachine
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
OUT=$E/probes.jsonl; touch $OUT
while true; do
  for s in $(seq 1000 1000 33000); do
    f="$M/20261001-test_SE_scale+bias-fc1leaky-replay-step$s.safetensors"
    [ -f "$f" ] || continue
    grep -q "\"step\":$s," $OUT && continue
    line=$("$BIN" --probe-model "$f" --probe-set wide 2>/dev/null | grep '"pElo"' | head -1)
    [ -n "$line" ] && echo "{\"step\":$s,${line#\{}" >> $OUT && echo "probed $s"
  done
  pgrep -f "DCM-de0f22b.app/Contents/MacOS/DrewsChessMachine --replay-corpus" >/dev/null || { echo LOOP_DONE; break; }
  sleep 60
done
