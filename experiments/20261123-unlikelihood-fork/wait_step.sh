#!/bin/zsh
# Block until the fork has written its checkpoint for trainer step $1 (or its process exits), print, exit.
set -u; setopt null_glob
STEP=$1; M="$HOME/Library/Application Support/DrewsChessMachine/Models"
while true; do
  f=( "$M"/20261123-zlra-elitecyc-unlik-latest-step$STEP.safetensors )
  if (( ${#f} > 0 )); then echo "step $STEP ok"; exit 0; fi
  pgrep -f "elitecyc-unlik-latest" >/dev/null || { echo "step $STEP EXITED"; exit 0; }
  sleep 30
done
