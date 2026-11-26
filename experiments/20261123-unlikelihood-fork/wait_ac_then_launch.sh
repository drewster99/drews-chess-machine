#!/bin/zsh
# Owner 2026-11-26: training stopped on battery; resume segment 2 once on AC power.
X=/Users/andrew/cursor/drews-chess-machine/experiments/20261123-unlikelihood-fork
until pmset -g batt | grep -q "AC Power"; do sleep 60; done
nohup $X/launch-seg2.sh >/dev/null 2>&1 &
sleep 120
echo "AC at $(date); launched seg2"; pgrep -fl elitecyc-unlik-latest-step637457 | cut -c1-120; tail -3 $X/train-seg2.stdout | cut -c1-200
