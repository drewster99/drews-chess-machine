#!/bin/zsh
# Block until every running arm has written its checkpoint for trainer step $1 (or an arm's process exits), then
# print what happened and exit. One-shot: the caller re-arms it for the next step.
set -u; setopt null_glob
STEP=$1; M="$HOME/Library/Application Support/DrewsChessMachine/Models"
while true; do
  done_all=1; out=""
  for a in A B C; do
    f=( "$M"/20261122-lsmode-$a-*step$STEP.safetensors )   # null_glob: empty when not written yet
    if (( ${#f} > 0 )); then out+="$a:ok "; elif pgrep -f "lsmode-$a-latest" >/dev/null; then done_all=0; else out+="$a:EXITED "; fi
  done
  if (( done_all )); then echo "step $STEP $out"; exit 0; fi
  sleep 30
done
