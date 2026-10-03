#!/bin/zsh
# Probe every enumerated checkpoint of one corpus-replay run with --probe-set wide,
# appending one JSON line per checkpoint to the given probes file. Exits once the run
# has ended and every checkpoint present has been probed (or has used up its retries).
#
# Usage: probe_loop.sh [--once] <out-model stem> <probes.jsonl> [step limit]
#
#   --once      one pass over the checkpoints present, no trainer required (re-probing a
#               finished run, e.g. with a newer PROBE_BIN).
#   step limit  optional: checkpoints above it are skipped, and each skip is logged.
#
# Environment:
#   PROBE_BIN               probe binary (needed for checkpoints in an architecture format
#                           the default frozen build cannot read)
#   TRAINER_PID             the trainer's pid, when the launcher knows it; otherwise the
#                           trainer is found by its exact --out-model path
#   PROBE_START_WAIT_SEC    how long to wait for the trainer to appear
#   PROBE_MAX_ATTEMPTS      failed probes of one checkpoint before it is given up on
#   PROBE_SEGMENT           the lineage segment index of the run segment to probe. A
#                           resumed segment names its step files
#                           <stem>-replay-seg<k>-step<N>; the run's first segment (0)
#                           names them <stem>-replay-step<N>. Unset means segment 0.
#
# Identity comes from each checkpoint's safetensors metadata (probe_record.py): the file
# name's step must match the header's training_step and the probe's modelID the header's
# model_id, and the probes file must hold a single model_id. A mismatch stops the loop.
# A failed probe is retried on later passes; its stderr and output are kept under
# <probes>.errors/ and reported, never silently dropped.
set -u
HERE=${0:A:h}
ONCE=0
if [ "${1:-}" = "--once" ]; then ONCE=1; shift; fi
STEM=${1:-}; OUT=${2:-}; LIMIT=${3:-}
[ -n "$STEM" ] && [ -n "$OUT" ] || { echo "usage: $0 [--once] <stem> <probes.jsonl> [step limit]" >&2; exit 2; }
BIN="${PROBE_BIN:-$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine}"
[ -x "$BIN" ] || { echo "probe binary not executable: $BIN" >&2; exit 2; }
# Every record names the build that measured it (scripts/dcm_probe_build.py).
BUILD=$(python3 "$HERE/../scripts/dcm_probe_build.py" "$BIN") || { echo "cannot identify the probe build of $BIN" >&2; exit 2; }
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
ROLLING="$M/$STEM-replay-latest.safetensors"
START_WAIT=${PROBE_START_WAIT_SEC:-600}
MAX_ATTEMPTS=${PROBE_MAX_ATTEMPTS:-3}
SEGMENT=${PROBE_SEGMENT:-0}
[[ "$SEGMENT" == <-> && "$SEGMENT" == (0|[1-9]*) ]] || { echo "PROBE_SEGMENT must be a non-negative integer without leading zeros: $SEGMENT" >&2; exit 2; }
# The marker the app puts before the step number: none for segment 0, -seg<k> after.
if [ "$SEGMENT" = 0 ]; then SEGMENT_PART=""; else SEGMENT_PART="-seg$SEGMENT"; fi
ERRDIR="${OUT:r}.errors"
mkdir -p "$ERRDIR"
touch "$OUT"
typeset -A attempts skipped_logged

# The trainer: an app binary running --replay-corpus with exactly this --out-model path.
# Fixed-string matching (no regex), so stems containing '+' match, and a shell or monitor
# whose command line merely mentions the path does not.
trainer_pids() {
  ps -axww -o pid=,command= | awk -v out=" --out-model $ROLLING " '
    index($0 " ", "/Contents/MacOS/DrewsChessMachine ") && index($0 " ", " --replay-corpus ") && index($0 " ", out) { print $1 }'
}
trainer_alive() {
  if [ -n "${TRAINER_PID:-}" ]; then
    kill -0 "$TRAINER_PID" 2>/dev/null || return 1
    trainer_pids | grep -qx "$TRAINER_PID"
  else
    [ -n "$(trainer_pids)" ]
  fi
}

if [ $ONCE = 0 ]; then
  waited=0
  while true; do
    count=$(trainer_pids | grep -c .)
    [ "$count" -eq 1 ] && break
    [ "$count" -gt 1 ] && { echo "more than one trainer writes $ROLLING:" $(trainer_pids) >&2; exit 3; }
    [ $waited -ge $START_WAIT ] && { echo "no trainer for $ROLLING appeared within ${START_WAIT}s" >&2; exit 3; }
    sleep 5; waited=$((waited + 5))
  done
fi

while true; do
  # Sampled before the pass: a trainer that has exited has already published every
  # checkpoint, so this pass sees its last one and the loop can stop after it.
  alive=0; [ $ONCE = 0 ] && trainer_alive && alive=1
  for f in "$M/$STEM"-replay"$SEGMENT_PART"-step<->.safetensors(Nn); do
    s=${${f:t:r}##*-replay"$SEGMENT_PART"-step}
    if [ -n "$LIMIT" ] && [ "$s" -gt "$LIMIT" ]; then
      [ -z "${skipped_logged[$s]:-}" ] && { echo "skipping step $s (above the step limit $LIMIT)"; skipped_logged[$s]=1; }
      continue
    fi
    grep -q "\"step\":$s," "$OUT" && continue
    [ "${attempts[$s]:-0}" -ge "$MAX_ATTEMPTS" ] && continue
    raw=$("$BIN" --probe-model "$f" --probe-set wide 2>>"$ERRDIR/step$s.stderr"); rc=$?
    rec=$(print -r -- "$raw" | python3 "$HERE/probe_record.py" "$f" "$s" "$OUT" "$BUILD" 2>>"$ERRDIR/step$s.err"); prc=$?
    if [ $prc -eq 0 ]; then
      print -r -- "$rec" >> "$OUT"
      echo "probed $s"
    elif [ $prc -eq 4 ] || [ $prc -eq 5 ]; then   # probe_record.py EXIT_IDENTITY / EXIT_OTHER_RUN
      echo "identity check failed at step $s (see $ERRDIR/step$s.err); stopping" >&2
      exit 4
    else
      attempts[$s]=$(( ${attempts[$s]:-0} + 1 ))
      { echo "--- $(date '+%Y-%m-%d %H:%M:%S') attempt ${attempts[$s]} probe exit $rc record exit $prc"; print -r -- "$raw"; } >> "$ERRDIR/step$s.err"
      echo "probe FAILED step $s attempt ${attempts[$s]}/$MAX_ATTEMPTS (see $ERRDIR/step$s.err)" >&2
    fi
  done
  [ $ONCE = 1 ] && break
  [ $alive = 0 ] && break
  sleep 60
done
echo LOOP_DONE
unprobed=()
for s in ${(k)attempts}; do grep -q "\"step\":$s," "$OUT" || unprobed+=$s; done
if [ ${#unprobed} -gt 0 ]; then
  echo "not probed after failures: step(s) ${(on)unprobed} (see $ERRDIR)" >&2
  exit 1
fi
exit 0
