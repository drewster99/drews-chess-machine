#!/bin/zsh
# Runs one --graph-audit config with an outside memory guard: the audit process (only it) is killed if the
# system's free-memory percentage falls below MIN_FREE_PCT. Usage: run_guarded.sh <config.json> <out.jsonl>
set -u
BIN=${DCM_AUDIT_BIN:-$HOME/Library/Developer/Xcode/DerivedData/DrewsChessMachine-eyigcdvyyrcsakaqcybzcfgsurbr/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine}
MIN_FREE_PCT=${MIN_FREE_PCT:-20}
"$BIN" --graph-audit "$1" --graph-audit-out "$2" 2> "${2%.jsonl}.stderr" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_PCT" ]; then
    echo "[guard] free memory ${free}% < ${MIN_FREE_PCT}%: killing audit pid $pid" | tee -a "${2%.jsonl}.stderr"
    kill -9 $pid
  fi
  sleep 3
done
wait $pid; echo "exit=$?"
