#!/bin/zsh
# Block until the fp32fix supervisor appends an event (or dies), print the new lines, exit.
OPS="$HOME/Library/Application Support/DrewsChessMachine/OpsScripts"
EV="$OPS/fp32fix_supervisor.events"
n=$(wc -l < "$EV")
while true; do
  m=$(wc -l < "$EV")
  if (( m > n )); then sleep 2; tail -n +$((n+1)) "$EV"; exit 0; fi
  kill -0 $(cat "$OPS/fp32fix_supervisor.pid") 2>/dev/null || { echo "SUPERVISOR DEAD"; tail -5 "$OPS/fp32fix_supervisor.out"; exit 1; }
  sleep 15
done
