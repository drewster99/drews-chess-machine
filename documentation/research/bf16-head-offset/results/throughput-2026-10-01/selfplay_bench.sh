#!/bin/zsh
# A-B-B-A self-play throughput: old (pre-fix 3f02ae4) vs new build, GUI --train, 480 s each.
B=${BENCH_DIR:?set BENCH_DIR}
OLDAPP=~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-bekabztybsbeynfbgeiiuyijnfnt/Build/Products/Release/DrewsChessMachine.app
NEWAPP=~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-cnwhxukmgvpbxrcuohhgxlihzycn/Build/Products/Release/DrewsChessMachine.app
: > $B/sp_runs.txt
for tag in spold1 spnew1 spnew2 spold2; do
  case $tag in spold*) APP=$OLDAPP;; *) APP=$NEWAPP;; esac
  start=$(date +%s)
  open -n -W -a "$APP" --args --train --training-time-limit 480 --output $B/$tag.json
  log=$(ls -t ~/Library/Logs/DrewsChessMachine/dcm_log_*.txt | head -1)
  echo "$tag secs=$(( $(date +%s)-start )) log=$log" >> $B/sp_runs.txt
done
echo SP_DONE >> $B/sp_runs.txt
