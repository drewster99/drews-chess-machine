#!/bin/zsh
# A-B-B-A: parent 3f02ae4 vs head-fix da15920, same model/corpus/params, 600 steps each, on AC.
B=${BENCH_DIR:?set BENCH_DIR}
binfor() { for d in ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*; do grep -q "$1" "$d/info.plist" 2>/dev/null && echo "$d/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine"; done; }
PARENT=$(binfor dcm-worktree-3f02ae4); FIX=$(binfor dcm-worktree-da15920)
echo "parent=$PARENT fix=$FIX" > $B/fix_runs.txt
M="$HOME/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors"
for tag in parentA fixA fixB parentB; do case $tag in parent*) BIN=$PARENT;; *) BIN=$FIX;; esac
  start=$(date +%s); "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M" --out-model $B/$tag.safetensors --parameters $B/bench_params.json --training-step-limit 600 > $B/$tag.stdout 2>&1; rc=$?
  log=$(ls -t ~/Library/Logs/DrewsChessMachine/dcm_log_*.txt | head -1); echo "$tag rc=$rc secs=$(( $(date +%s)-start )) power=$(pmset -g batt | head -1 | grep -o "'.*'") log=$log" >> $B/fix_runs.txt
done
echo FIX_DONE >> $B/fix_runs.txt
