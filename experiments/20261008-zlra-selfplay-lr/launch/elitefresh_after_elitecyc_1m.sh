#!/bin/zsh
# Owner 2026-11-01 19:35: "stop elitefresh until elitecyc hits 1 million steps". Waits for the R-elitecyc segment 18
# process to exit, then starts R-elitefresh segment 13 only if R-elitecyc's log shows its final save at trainer step
# 1,000,000. Any other exit (crash, owner stop, sleep death) launches nothing and says why in the watcher log.
set -u
E=/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr
ELITECYC_PID=24076
ELITECYC_LOG="$HOME/Library/Logs/DrewsChessMachine/dcm_log_20261101-193002-2.txt"
echo "$(date) watching R-elitecyc pid $ELITECYC_PID"
while kill -0 $ELITECYC_PID 2>/dev/null; do sleep 60; done
echo "$(date) R-elitecyc pid $ELITECYC_PID exited"
if grep -qE '\[REPLAY\] saved trainer model \(final\).*trainerStep=1000000 ' "$ELITECYC_LOG"; then
  cd $E
  nohup caffeinate -i zsh launch/elitefresh_resume14_launch.sh 2> elitefresh_resume14.launch.err < /dev/null &!
  echo "$(date) R-elitecyc reached 1,000,000; launched R-elitefresh segment 13"
else
  echo "$(date) R-elitecyc did not end with a final save at 1,000,000; R-elitefresh NOT launched"
  grep -E '\[REPLAY\] (saved trainer model|done:)' "$ELITECYC_LOG" | tail -3
fi
