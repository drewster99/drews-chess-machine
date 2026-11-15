#!/bin/zsh
# Runs every matrix/*.json case not yet done (a .jsonl with a case_done or case_error line), in order, each under
# run_guarded.sh. Safe to re-run after an interruption.
set -u
cd ${0:A:h}
for cfg in matrix/*.json; do
  out=${cfg%.json}.jsonl
  if [ -f $out ] && grep -qE '"event":"(case_done|case_error)"' $out; then continue; fi
  [ -f $out ] && mv $out ${out%.jsonl}-interrupted-$(date +%H%M%S).jsonl
  echo "== $(date +%H:%M:%S) $cfg"
  ./run_guarded.sh $cfg $out
done
echo "MATRIX DONE $(date +%H:%M:%S)"
