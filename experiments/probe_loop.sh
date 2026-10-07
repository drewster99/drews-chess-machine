#!/bin/zsh
# Probe every enumerated checkpoint of one corpus-replay run with --probe-set wide,
# appending one JSON line per checkpoint to the given probes file. Exits once the run
# has ended and every checkpoint present has been probed (or has used up its retries).
#
# Usage: probe_loop.sh [--once] <out-model stem> <probes.jsonl> [step limit]
#
#   --once      one pass over the checkpoints present, no trainer required (re-probing a
#               finished run, e.g. with a newer PROBE_BIN).
#   step limit  optional: checkpoints whose name's step is above it are skipped, and each
#               skip is logged. The name's step is the header's training_step: for files
#               from architecture format v11 on that is the TRAINER step (a resumed
#               segment's limit is its start trainer step plus its --training-step-limit);
#               for older corpus-replay files it is the writing segment's own step. A
#               segment-0 probe of a stem that a later segment also wrote into needs its
#               own last trainer step as the limit, or it reaches the later segment's
#               files and stops on the other-run check (exit 4).
#
# Environment:
#   PROBE_BIN               probe binary. It must be a build that reads the files' format:
#                           files written from format v11 on need a v11 build, and the
#                           default frozen build predates v11 (it refuses them, loudly)
#   TRAINER_PID             the trainer's pid, when the launcher knows it; otherwise the
#                           trainer is found by its exact --out-model path. It must be the
#                           app process itself (not a nohup / caffeinate / open wrapper):
#                           a pid that is not the trainer writing this out-model stops the
#                           loop at start (exit 3), instead of reading as a trainer that
#                           has already exited and ending the loop after one pass
#   PROBE_START_WAIT_SEC    how long to wait for the trainer to appear
#   PROBE_MAX_ATTEMPTS      failed probes of one checkpoint before it is given up on
#   PROBE_ABOVE_STEP        the segment's start trainer step (default 0): step files at or
#                           below it are skipped, each skip logged once. From format v11 a
#                           resumed segment names its step files by the trainer step under the
#                           same stem as the segments before it (<stem>-replay-step<T>); those
#                           earlier files carry other model_ids, and without this bound the
#                           loop would reach them and stop on the other-run check (exit 4).
#   PROBE_SEGMENT           LEGACY ONLY: the lineage segment index of a resumed segment
#                           written before format v11, which named its step files
#                           <stem>-replay-seg<k>-step<N> (N its own step). Unset means the
#                           unmarked names <stem>-replay-step<N>. Refused together with
#                           PROBE_ABOVE_STEP: the two describe different naming schemes.
#
# Identity comes from each checkpoint's safetensors metadata (probe_record.py): the file
# name's step must match the header's training_step and the probe's modelID the header's
# model_id, and the probes file must hold a single model_id. A mismatch stops the loop.
# A failed probe is retried on later passes; its stderr and output are kept under
# <probes>.errors/ and reported, never silently dropped. A probe that exits non-zero has
# failed even when it printed a summary first (a crash in teardown, say): its output is
# not recorded, the same rule the dashboard tracker's probe follows.
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
if [ -n "${PROBE_SEGMENT+set}" ] && [ -n "${PROBE_ABOVE_STEP+set}" ]; then
  echo "PROBE_SEGMENT (legacy -seg<k> names) and PROBE_ABOVE_STEP (trainer-step names) cannot be used together" >&2
  exit 2
fi
SEGMENT=${PROBE_SEGMENT:-0}
[[ "$SEGMENT" == <-> && "$SEGMENT" == (0|[1-9]*) ]] || { echo "PROBE_SEGMENT must be a non-negative integer without leading zeros: $SEGMENT" >&2; exit 2; }
ABOVE=${PROBE_ABOVE_STEP:-0}
[[ "$ABOVE" == <-> && "$ABOVE" == (0|[1-9]*) ]] || { echo "PROBE_ABOVE_STEP must be a non-negative integer without leading zeros: $ABOVE" >&2; exit 2; }
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
  if [ -n "${TRAINER_PID:-}" ] && ! trainer_pids | grep -qx "$TRAINER_PID"; then
    echo "TRAINER_PID $TRAINER_PID is not the trainer writing $ROLLING (found: $(trainer_pids))" >&2
    exit 3
  fi
fi

# One failed attempt at probing step $1, for reason $2; the probe's output is in $raw.
failed_attempt() {
  attempts[$1]=$(( ${attempts[$1]:-0} + 1 ))
  { echo "--- $(date '+%Y-%m-%d %H:%M:%S') attempt ${attempts[$1]} $2"; print -r -- "$raw"; } >> "$ERRDIR/step$1.err"
  echo "probe FAILED step $1 attempt ${attempts[$1]}/$MAX_ATTEMPTS ($2; see $ERRDIR/step$1.err)" >&2
}

while true; do
  # Sampled before the pass: a trainer that has exited has already published every
  # checkpoint, so this pass sees its last one and the loop can stop after it.
  alive=0; [ $ONCE = 0 ] && trainer_alive && alive=1
  for f in "$M/$STEM"-replay"$SEGMENT_PART"-step<->.safetensors(Nn); do
    s=${${f:t:r}##*-replay"$SEGMENT_PART"-step}
    if [ "$s" -le "$ABOVE" ]; then
      [ -z "${skipped_logged[$s]:-}" ] && { echo "skipping step $s (at or below PROBE_ABOVE_STEP $ABOVE: an earlier segment's file)"; skipped_logged[$s]=1; }
      continue
    fi
    if [ -n "$LIMIT" ] && [ "$s" -gt "$LIMIT" ]; then
      [ -z "${skipped_logged[$s]:-}" ] && { echo "skipping step $s (above the step limit $LIMIT)"; skipped_logged[$s]=1; }
      continue
    fi
    grep -q "\"step\":$s," "$OUT" && continue
    [ "${attempts[$s]:-0}" -ge "$MAX_ATTEMPTS" ] && continue
    raw=$("$BIN" --probe-model "$f" --probe-set wide 2>>"$ERRDIR/step$s.stderr"); rc=$?
    if [ "$rc" != 0 ]; then
      failed_attempt "$s" "probe exit $rc"
      continue
    fi
    rec=$(print -r -- "$raw" | python3 "$HERE/probe_record.py" "$f" "$s" "$OUT" "$BUILD" 2>>"$ERRDIR/step$s.err"); prc=$?
    if [ "$prc" = 0 ]; then
      print -r -- "$rec" >> "$OUT"
      echo "probed $s"
    elif [ "$prc" = 4 ] || [ "$prc" = 5 ]; then   # probe_record.py EXIT_IDENTITY / EXIT_OTHER_RUN
      echo "identity check failed at step $s (see $ERRDIR/step$s.err); stopping" >&2
      exit 4
    else
      failed_attempt "$s" "probe exit 0 record exit $prc"
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
