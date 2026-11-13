#!/bin/zsh
# R-elitecyc-fp32 (owner 2026-11-12): R-elitecyc from the same saved state as the hilr fork (trainer step 384,964),
# under exactly R-elitecyc's own continuation (parameters-zlra-elitecyc-decay.json, the decaying keyframe schedule;
# same corpus and feed as elitecyc_resume13_decay_launch.sh), but computing in float32 instead of bfloat16.
#
# Start file: 20261112-zlra-elitecyc-fp32-start.safetensors, a --derive-model --set-compute-dtype float32 copy of
# 20261011-zlra-elitecyc-latest-step384964.safetensors (DERIVE_TRAINER_STATE_DTYPE_PLAN): every tensor bit-exact,
# fp32 masters and velocity included; the lineage record is the source's, so this continues run E709 as its next
# segment. Accepted gaps: `schedule` (the file recorded the gentle cycle; R-elitecyc's own segment 13 accepted the
# same), `compute_data_type` (the trainer state was captured under bf16) and `policy_tail_precision` (fp32 has no policy tail; the bf16 state ran fp32_from_pre_bn). Question: does fp32 compute change
# where the annealed line ends up? Compare with R-elitecyc (bf16) from 384,964 on, step for step.
set -u
R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261008-zlra-selfplay-lr
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2669-e9d4f722.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
START_STEP=384964
TARGET_STEP=1000000

exec "$BIN" --replay-corpus 20260704-215145-op24Gp --start-model "$M/20261112-zlra-elitecyc-fp32-start.safetensors" \
  --resume-exact --accept-inexact "${DCM_ACCEPT_INEXACT:-schedule,compute_data_type,policy_tail_precision}" \
  --parameters $E/parameters-zlra-elitecyc-decay.json \
  --out-model "$M/20261112-zlra-elitecyc-fp32-latest.safetensors" \
  --epochs 7 --training-step-limit $(( TARGET_STEP - START_STEP )) \
  --enumerate-checkpoints --output $E/results-elitecyc-fp32-seg0.json \
  > $E/train-elitecyc-fp32-seg0.stdout 2>&1
