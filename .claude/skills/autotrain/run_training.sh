#!/bin/bash
#
# run_training.sh — launch the chess app for a time-limited training run,
# writing results to a JSON file, then verify the output.
#
# Usage:
#   run_training.sh <params-json> <time-limit-seconds> <output-json> [log-file]
#
# Env overrides (only needed if the app's CLI flag names change):
#   DCM_PARAMS_FLAG   (default: --parameters)
#   DCM_TIME_FLAG     (default: --training-time-limit)
#   DCM_OUT_FLAG      (default: --results-output)
#
# Delegates binary discovery to ../../../run_debug.sh at the repo root.

set -u

PROG="$(basename "$0")"

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $PROG <params-json> <time-limit-seconds> <output-json> [log-file]" >&2
    exit 2
fi

PARAMS="$1"
TIME_LIMIT="$2"
OUTPUT="$3"
LOG_FILE="${4:-/dev/null}"

PARAMS_FLAG="${DCM_PARAMS_FLAG:---parameters}"
TIME_FLAG="${DCM_TIME_FLAG:---training-time-limit}"
OUT_FLAG="${DCM_OUT_FLAG:---output}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_DEBUG="$REPO_ROOT/run_debug.sh"

if [ ! -x "$RUN_DEBUG" ]; then
    echo "$PROG: expected $RUN_DEBUG to exist and be executable" >&2
    exit 3
fi

# --- Concurrency guards ---------------------------------------------------
# Exit code 10 is reserved for "another DrewsChessMachine / training run is
# already in flight". Callers (skill step 0) treat that as "skip iteration",
# distinct from a real failure. Guards run before the params-file check so
# that /loop ticks bail out cheaply when the GPU is busy.

if pgrep -x DrewsChessMachine >/dev/null 2>&1; then
    echo "$PROG: DrewsChessMachine is already running — skipping" >&2
    exit 10
fi

LOCK_FILE="$REPO_ROOT/.dcm.training.lock"
if [ -f "$LOCK_FILE" ]; then
    OTHER_PID="$(head -n1 "$LOCK_FILE" 2>/dev/null | tr -dc '0-9')"
    if [ -n "$OTHER_PID" ] && kill -0 "$OTHER_PID" 2>/dev/null; then
        echo "$PROG: training lock held by live PID $OTHER_PID — skipping" >&2
        exit 10
    fi
    echo "$PROG: removing stale lock (pid=${OTHER_PID:-unknown})" >&2
    rm -f "$LOCK_FILE"
fi

echo "$$" > "$LOCK_FILE"
cleanup_lock() { rm -f "$LOCK_FILE"; }
trap cleanup_lock EXIT INT TERM HUP
# -------------------------------------------------------------------------

if [ ! -f "$PARAMS" ]; then
    echo "$PROG: parameters file not found: $PARAMS" >&2
    exit 4
fi

case "$TIME_LIMIT" in
    ''|*[!0-9]*)
        echo "$PROG: time-limit must be a positive integer (got: $TIME_LIMIT)" >&2
        exit 5 ;;
esac

if [ "$TIME_LIMIT" -gt 1800 ]; then
    echo "$PROG: clamping time limit from $TIME_LIMIT to 1800 seconds (hard cap)" >&2
    TIME_LIMIT=1800
fi

# Make sure the target dir exists and remove any stale output.
OUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUT_DIR"
rm -f "$OUTPUT"

echo "$PROG: launching DrewsChessMachine (limit=${TIME_LIMIT}s, params=$PARAMS, out=$OUTPUT)" | tee -a "$LOG_FILE"

# Give the app a safety margin beyond its self-imposed time limit so we catch
# hangs without racing the normal shutdown path.
WATCHDOG=$((TIME_LIMIT + 120))

# Run under the BSD /usr/bin/perl timeout idiom (macOS has no GNU `timeout`).
# If it's already past the internal limit by $WATCHDOG seconds, something is
# wrong — send SIGTERM, and escalate to SIGKILL if still running.
(
    "$RUN_DEBUG" \
        --train \
        "$PARAMS_FLAG" "$PARAMS" \
        "$TIME_FLAG" "$TIME_LIMIT" \
        "$OUT_FLAG" "$OUTPUT"
) >>"$LOG_FILE" 2>&1 &
APP_PID=$!

(
    sleep "$WATCHDOG"
    if kill -0 "$APP_PID" 2>/dev/null; then
        echo "$PROG: watchdog firing after ${WATCHDOG}s, sending SIGTERM" >>"$LOG_FILE"
        kill -TERM "$APP_PID" 2>/dev/null
        sleep 10
        if kill -0 "$APP_PID" 2>/dev/null; then
            echo "$PROG: watchdog escalating to SIGKILL" >>"$LOG_FILE"
            kill -KILL "$APP_PID" 2>/dev/null
        fi
    fi
) &
WATCHDOG_PID=$!

wait "$APP_PID"
APP_EXIT=$?

# Watchdog is no longer needed.
kill "$WATCHDOG_PID" 2>/dev/null
wait "$WATCHDOG_PID" 2>/dev/null

if [ "$APP_EXIT" -ne 0 ]; then
    echo "$PROG: app exited with status $APP_EXIT" >&2
    exit 6
fi

if [ ! -f "$OUTPUT" ]; then
    echo "$PROG: app exited cleanly but output file $OUTPUT was not produced" >&2
    exit 7
fi

if ! /usr/bin/python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$OUTPUT" 2>/dev/null; then
    echo "$PROG: output file $OUTPUT is not valid JSON" >&2
    exit 8
fi

echo "$PROG: run complete, output at $OUTPUT" | tee -a "$LOG_FILE"
exit 0
