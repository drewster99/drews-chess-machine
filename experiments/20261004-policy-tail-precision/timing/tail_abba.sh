#!/bin/zsh
# fp32 tail vs mixed policy tail, A-B-B-A per model, 600 steps each, one build (2320), on AC.
B=${0:A:h}
# runs.txt is the record of the 2026-10-04/05 benchmark; a re-run must not append to it.
[[ -e $B/runs.txt ]] && { echo "$B/runs.txt exists (the recorded run); move it aside to re-run" >&2; exit 2; }
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
if [[ ! -f $B/v4fresh.safetensors ]]; then
  # The recorded run minted with a drawn seed; this re-mints the same net (v4fresh.mint.txt) and logs separately.
  "$BIN" --new-model --architecture v4_5block_7x7 --init-seed 2150607837842323118 --out-model $B/v4fresh.safetensors > $B/v4fresh.mint.rerun.txt 2>&1 || { echo "MINT_FAILED rc=$?" >> $B/runs.txt; exit 1; }
fi
typeset -A MODEL
MODEL[r7]="$M/20261002-bench_v5s3_noSE_noReZero-replay-step33000.safetensors"
MODEL[se]="$M/20260929-test_SE_scale+bias-seed2-replay-step7282.safetensors"
MODEL[v4]="$B/v4fresh.safetensors"
for net in r7 se v4; do
  for arm in fp32A mixedA mixedB fp32B; do
    case $arm in fp32*) P=fp32_from_pre_bn;; *) P=mixed_final_projection;; esac
    if pgrep -f "DrewsChessMachine --(replay-corpus|train|probe-model)" >/dev/null; then
      echo "ABORT another DrewsChessMachine job is running before $net-$arm" >> $B/runs.txt; exit 2
    fi
    tag=$net-$arm; start=$(date +%s)
    "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "${MODEL[$net]}" --out-model $B/$tag.safetensors \
      --parameters $B/bench_params.json --training-step-limit 600 --policy-tail-precision $P > $B/$tag.stdout 2>&1; rc=$?
    log=$(ls -t ~/Library/Logs/DrewsChessMachine/dcm_log_*.txt | head -1)
    echo "$tag rc=$rc secs=$(( $(date +%s)-start )) tail=$P power=$(pmset -g batt | head -1 | grep -o "'.*'") log=$log" >> $B/runs.txt
  done
done
echo BENCH_DONE >> $B/runs.txt
