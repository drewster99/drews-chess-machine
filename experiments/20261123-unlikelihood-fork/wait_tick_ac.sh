#!/bin/zsh
# Wait for the fork's step-$1 checkpoint, then run its tick. Exits early (and says so) if the Mac goes on battery
# (owner 2026-11-26: stop training on battery, resume on AC) or the trainer exits.
S=$1; X=${0:A:h}; M="$HOME/Library/Application Support/DrewsChessMachine/Models"
until [[ -e "$M/20261123-zlra-elitecyc-unlik-latest-step$S.safetensors" ]]; do
  pmset -g batt | grep -q 'Battery Power' && { echo "ON BATTERY $(date)"; exit 0; }
  pgrep -f elitecyc-unlik-latest >/dev/null || { echo "TRAINING EXITED $(date)"; exit 0; }
  sleep 30
done
$X/tick.sh $S 2>&1 | tail -3 | cut -c1-250
