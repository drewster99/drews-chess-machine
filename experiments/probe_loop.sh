#!/bin/zsh
# Probe every enumerated 1000-step checkpoint of one replay run with --probe-set wide,
# appending one JSON line per checkpoint to the given probes file. Exits once the run
# has ended and every checkpoint present is probed.
# Usage: probe_loop.sh <out-model stem> <probes.jsonl> [step limit, default 33000]
# PROBE_BIN overrides the probe binary (needed for checkpoints a newer architecture
# format writes, which build 2275 cannot read).
STEM=$1; OUT=$2; LIMIT=${3:-33000}
[ -n "$STEM" ] && [ -n "$OUT" ] || { echo "usage: $0 <stem> <probes.jsonl> [limit]" >&2; exit 2; }
BIN="${PROBE_BIN:-$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine}"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
touch $OUT
while true; do
  for s in $(seq 1000 1000 $LIMIT); do
    f="$M/$STEM-replay-step$s.safetensors"
    [ -f "$f" ] || continue
    grep -q "\"step\":$s," $OUT && continue
    line=$("$BIN" --probe-model "$f" --probe-set wide 2>/dev/null | grep '"pElo"' | head -1)
    [ -n "$line" ] && echo "{\"step\":$s,${line#\{}" >> $OUT && echo "probed $s"
  done
  pgrep -f "$STEM-replay-latest" >/dev/null || { echo LOOP_DONE; break; }
  sleep 60
done
