#!/bin/zsh
# V-3, V-4 (+ exact resume), V-5 (D1, D2, E), V-2 — corpus-replay CLI runs of
# a DCM build that has the training-health alarms (main from 18cac01d on).
# Usage: run_v.sh <DrewsChessMachine binary> <output folder>. Parameter files are read from
# this script's folder; model files, results and logs go to the output folder only.
P=${0:A:h}
BIN=${1:?binary}
V=${2:?output folder}
mkdir -p "$V"
START="/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20261005-r7b24-fresh.safetensors"
CORPUS=20260624-192615-w3aA5b
STATUS=$V/status.txt
diskok() {
  local free=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
  echo "$(date +%H:%M:%S) disk free ${free}G" >> $STATUS
  [ "$free" -ge 15 ]
}
run() {
  local name=$1; shift
  until diskok; do echo "$(date +%H:%M:%S) $name waiting for disk" >> $STATUS; sleep 600; done
  local start=$(date +%s)
  "$BIN" --replay-corpus $CORPUS "$@" --policy-tail-precision fp32_from_pre_bn --seed 20261005 > $V/$name.out 2> $V/$name.err
  local rc=$?
  echo "$(date +%H:%M:%S) $name exit=$rc secs=$(( $(date +%s) - start ))" >> $STATUS
}
: > $STATUS
run V3 --start-model "$START" --out-model $V/V3.safetensors --parameters $P/params-C.json --training-step-limit 600 --output $V/V3-results.json
run V4 --start-model "$START" --out-model $V/V4.safetensors --parameters $P/params-C-stop.json --training-step-limit 600 --output $V/V4-results.json
run V4resume --start-model $V/V4.safetensors --resume-exact --out-model $V/V4resume.safetensors --parameters $P/params-C-stop.json --training-step-limit 1
run V5-D1 --start-model "$START" --out-model $V/V5-D1.safetensors --parameters $P/params-A-off.json --training-step-limit 300
run V5-D2 --start-model "$START" --out-model $V/V5-D2.safetensors --parameters $P/params-A-off.json --training-step-limit 300
run V5-E --start-model "$START" --out-model $V/V5-E.safetensors --parameters $P/params-A-on50.json --training-step-limit 300
run V2 --start-model "$START" --out-model $V/V2.safetensors --parameters $P/params-A.json --training-step-limit 2000 --output $V/V2-results.json
echo "$(date +%H:%M:%S) all done" >> $STATUS
