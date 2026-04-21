#!/bin/bash
#
# End-to-end test for the --parameters / --output / --train CLI flags.
#
# What this does:
#   1. Builds the app with xcodebuild (DerivedData path resolved from
#      xcodebuild itself so it doesn't assume a user-specific location).
#   2. Writes a parameters JSON with distinctive values.
#   3. Launches the built binary with:
#        --train
#        --parameters <params.json>
#        --output <out.json>
#      The parameters file's `training_time_limit` drives when the
#      process writes the snapshot and exits.
#   4. Waits for the output file + exit, then asserts — via jq —
#      that every CLI override is reflected in:
#        - the first stats[] entry's matching field
#        - the top-level session_id / training_steps / positions_trained
#   5. Emits PASS / FAIL for each assertion and exits non-zero on any
#      mismatch.
#
# Why assertions target stats[0]: the stats logger snapshots every
# configured hyperparameter on each emit, so a single stats entry
# is ground truth for "what the trainer actually ran with". If the
# CLI override failed to take effect, the value in the JSON would
# be the @AppStorage default (or a stale persisted value), which
# is exactly the failure mode this test exists to catch.
#
# Requires: xcodebuild, jq, Xcode command-line tools, and no
# currently-running DrewsChessMachine instance (the app is
# single-window and a duplicate launch will be rejected).
#
# Usage:  tools/test-cli-params.sh
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$REPO_ROOT/DrewsChessMachine/DrewsChessMachine.xcodeproj"
SCHEME="DrewsChessMachine"

TMPDIR=$(mktemp -d -t dcm-cli-params)
trap 'rm -rf "$TMPDIR"' EXIT

PARAMS_JSON="$TMPDIR/params.json"
OUT_JSON="$TMPDIR/out.json"

# Training duration for the test. Short enough to run quickly but
# long enough that at least one [STATS] line emits (bootstrap fires
# on the first step, so ~10s is usually plenty once training starts
# producing). Override with TEST_TRAINING_TIME_LIMIT_SEC env var.
LIMIT_SEC="${TEST_TRAINING_TIME_LIMIT_SEC:-30}"

cat > "$PARAMS_JSON" <<EOF
{
  "entropy_bonus": 2.5e-3,
  "grad_clip_max_norm": 25.0,
  "weight_decay": 5.0e-4,
  "K": 6.0,
  "learning_rate": 7.5e-5,
  "draw_penalty": 0.15,
  "self_play_start_tau": 1.2,
  "self_play_target_tau": 0.35,
  "self_play_tau_decay_per_ply": 0.045,
  "arena_start_tau": 0.95,
  "arena_target_tau": 0.22,
  "arena_tau_decay_per_ply": 0.038,
  "replay_ratio_target": 1.25,
  "replay_ratio_auto_adjust": false,
  "self_play_workers": 6,
  "training_step_delay_ms": 75,
  "training_batch_size": 1024,
  "replay_buffer_capacity": 250000,
  "replay_buffer_min_positions_before_training": 2000,
  "arena_promote_threshold": 0.58,
  "arena_games_per_tournament": 80,
  "arena_auto_interval_sec": 900,
  "candidate_probe_interval_sec": 20,
  "training_time_limit": ${LIMIT_SEC}
}
EOF

echo "== 1. Building app =="
BUILD_DIR=$(xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration Debug -showBuildSettings 2>/dev/null \
    | awk -F ' = ' '/^ *BUILT_PRODUCTS_DIR /{print $2; exit}')
xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration Debug build >/dev/null

APP_BIN="$BUILD_DIR/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine"
if [[ ! -x "$APP_BIN" ]]; then
    echo "FAIL: app binary not found at $APP_BIN"
    exit 2
fi

echo "== 2. Launching $APP_BIN =="
echo "   params: $PARAMS_JSON"
echo "   output: $OUT_JSON"
echo "   limit:  ${LIMIT_SEC}s"

# Give the deadline a generous grace window. The CLI deadline task
# calls exit(0) after writeJSON; if it doesn't, kill the process
# after 2x the limit so the test never hangs.
GRACE=$(( LIMIT_SEC * 2 + 30 ))

# Run in background so we can kill-on-timeout. `--train` builds a
# fresh network and starts Play-and-Train; at `training_time_limit`
# the app writes `$OUT_JSON` and calls exit(0).
"$APP_BIN" --train --parameters "$PARAMS_JSON" --output "$OUT_JSON" &
APP_PID=$!

# Wait for either the output file to appear or the grace window
# to close.
SECONDS_WAITED=0
while [[ $SECONDS_WAITED -lt $GRACE ]]; do
    if [[ -f "$OUT_JSON" ]] && ! kill -0 "$APP_PID" 2>/dev/null; then
        break
    fi
    sleep 1
    SECONDS_WAITED=$(( SECONDS_WAITED + 1 ))
done

if kill -0 "$APP_PID" 2>/dev/null; then
    echo "FAIL: app did not exit within grace window (${GRACE}s); killing"
    kill -9 "$APP_PID" 2>/dev/null || true
    exit 3
fi
wait "$APP_PID" || true

if [[ ! -f "$OUT_JSON" ]]; then
    echo "FAIL: $OUT_JSON was never written"
    exit 4
fi

echo "== 3. Asserting overrides in $OUT_JSON =="

if ! command -v jq >/dev/null 2>&1; then
    echo "FAIL: jq not installed; install with 'brew install jq'"
    exit 5
fi

STATS_COUNT=$(jq '.stats | length' "$OUT_JSON")
if [[ "$STATS_COUNT" -lt 1 ]]; then
    echo "FAIL: stats[] is empty — no [STATS] line fired before exit. Try increasing TEST_TRAINING_TIME_LIMIT_SEC."
    exit 6
fi
echo "   stats entries: $STATS_COUNT"

FAILURES=0
assert() {
    local label="$1"
    local expected="$2"
    local actual="$3"
    # Float comparison with 6-decimal tolerance via awk so
    # floating-point rounding doesn't false-positive.
    local ok
    ok=$(awk -v a="$actual" -v b="$expected" 'BEGIN { print (((a>b?a-b:b-a) < 1e-6) ? "1" : "0") }')
    if [[ "$ok" == "1" ]]; then
        echo "   PASS  $label: $actual"
    else
        echo "   FAIL  $label: expected=$expected actual=$actual"
        FAILURES=$(( FAILURES + 1 ))
    fi
}

FIRST_STATS() { jq ".stats[0].$1" "$OUT_JSON"; }

assert "learning_rate"                 7.5e-5 "$(FIRST_STATS learning_rate)"
assert "entropy_regularization_coeff"  2.5e-3 "$(FIRST_STATS entropy_regularization_coeff)"
assert "grad_clip_max_norm"            25     "$(FIRST_STATS grad_clip_max_norm)"
assert "weight_decay"                  5.0e-4 "$(FIRST_STATS weight_decay)"
assert "policy_scale_k"                6      "$(FIRST_STATS policy_scale_k)"
assert "draw_penalty"                  0.15   "$(FIRST_STATS draw_penalty)"
assert "self_play_start_tau"           1.2    "$(FIRST_STATS self_play_start_tau)"
assert "self_play_floor_tau"           0.35   "$(FIRST_STATS self_play_floor_tau)"
assert "self_play_decay_per_ply"       0.045  "$(FIRST_STATS self_play_decay_per_ply)"
assert "arena_start_tau"               0.95   "$(FIRST_STATS arena_start_tau)"
assert "arena_floor_tau"               0.22   "$(FIRST_STATS arena_floor_tau)"
assert "arena_decay_per_ply"           0.038  "$(FIRST_STATS arena_decay_per_ply)"
assert "ratio_target"                  1.25   "$(FIRST_STATS ratio_target)"
assert "worker_count"                  6      "$(FIRST_STATS worker_count)"
assert "batch_size"                    1024   "$(FIRST_STATS batch_size)"
assert "arena_promote_threshold"       0.58   "$(FIRST_STATS arena_promote_threshold)"
assert "arena_games_per_tournament"    80     "$(FIRST_STATS arena_games_per_tournament)"
# replay_ratio_auto_adjust is a bool; jq emits true/false — check textually.
RRAA=$(jq -r '.stats[0].ratio_auto_adjust' "$OUT_JSON")
if [[ "$RRAA" == "false" ]]; then
    echo "   PASS  ratio_auto_adjust: $RRAA"
else
    echo "   FAIL  ratio_auto_adjust: expected=false actual=$RRAA"
    FAILURES=$(( FAILURES + 1 ))
fi
BUFCAP=$(jq '.stats[0].buffer_capacity' "$OUT_JSON")
if [[ "$BUFCAP" == "250000" ]]; then
    echo "   PASS  buffer_capacity: $BUFCAP"
else
    echo "   FAIL  buffer_capacity: expected=250000 actual=$BUFCAP"
    FAILURES=$(( FAILURES + 1 ))
fi

# Top-level mirror assertions.
SESS=$(jq -r '.session_id' "$OUT_JSON")
if [[ -z "$SESS" || "$SESS" == "null" ]]; then
    echo "   FAIL  session_id: empty"
    FAILURES=$(( FAILURES + 1 ))
else
    echo "   PASS  session_id: $SESS"
fi

STEPS=$(jq '.training_steps' "$OUT_JSON")
if [[ "$STEPS" == "null" ]] || [[ "$STEPS" -lt 1 ]]; then
    echo "   FAIL  training_steps must be >= 1; got $STEPS"
    FAILURES=$(( FAILURES + 1 ))
else
    echo "   PASS  training_steps: $STEPS"
fi

if [[ "$FAILURES" -ne 0 ]]; then
    echo "== $FAILURES assertion(s) failed =="
    exit 7
fi

echo "== All CLI parameter overrides verified =="
