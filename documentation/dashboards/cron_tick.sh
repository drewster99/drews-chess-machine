#!/bin/bash
# DCM per-mark monitoring tick — driven by the user's crontab (every 5 min), NOT by
# the assistant. Auto-detects the active replay-corpus run and updates its CSV +
# dashboard (freeze checkpoint -> GPU probe -> log-backfill -> render). No-op when
# nothing is training. Decouples checkpoint preservation + pElo tracking from any
# interactive session, so a monitoring gap can never silently drop pElo history again.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"
DIR="/Users/andrew/Documents/drews-chess-machine/documentation/dashboards"
LOG="$DIR/cron_tick.log"
LOCK="/tmp/dcm_cron_tick.lock"
LOCK_OWNER="$LOCK/owner"

# Lock = a directory (mkdir is atomic) so overlapping fires can't collide on the
# CSV/render. The holder records its identity in $LOCK_OWNER as "<pid> <start
# time>". The start time is what makes the identity unique: a bare PID can be
# reused by an unrelated process after the holder dies, which would wedge the
# lock forever (or, read the other way, let a stale lock look alive).
#
# A lock is only ever cleared when its recorded owner is provably gone. It is
# NOT cleared by age alone: a slow tick (a long GPU probe) can legitimately hold
# it for longer than any fixed timeout, and clearing it then lets two ticks
# rewrite the same CSV/registry concurrently -- one silently losing the other's
# rows. The only age-based clear left is for a lock with NO owner record, which
# can only be one whose holder died between mkdir and writing the record.
process_identity() {
  # Prints "<pid> <start time>" for a live process, nothing (and status 1) if gone.
  process_start_time=$(ps -o lstart= -p "$1" 2>/dev/null) || return 1
  [ -n "$process_start_time" ] || return 1
  printf '%s %s\n' "$1" "$process_start_time"
}

claim_lock() {
  mkdir "$LOCK" 2>/dev/null || return 1
  if ! process_identity $$ > "$LOCK_OWNER"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') cron_tick: could not record lock owner; releasing" >> "$LOG"
    rm -f "$LOCK_OWNER"
    rmdir "$LOCK"
    exit 1
  fi
}

release_lock() {
  # Only remove the lock if it is still ours (never someone else's re-acquired lock).
  if [ "$(cat "$LOCK_OWNER" 2>/dev/null)" = "$(process_identity $$)" ]; then
    rm -f "$LOCK_OWNER"
    rmdir "$LOCK" 2>/dev/null
  fi
}

if ! claim_lock; then
  recorded_owner=$(cat "$LOCK_OWNER" 2>/dev/null)
  if [ -n "$recorded_owner" ]; then
    recorded_pid=${recorded_owner%% *}
    if kill -0 "$recorded_pid" 2>/dev/null \
       && [ "$(process_identity "$recorded_pid")" = "$recorded_owner" ]; then
      exit 0   # held by a live tick
    fi
  elif [ -z "$(find "$LOCK" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then
    exit 0     # no owner record yet: the holder may be between mkdir and writing it
  fi
  # Stale. Break it by renaming (atomic, so only one breaker can win), then make
  # sure what we moved is the lock we judged stale and not one re-acquired since.
  stale_lock="$LOCK.stale.$$"
  mv "$LOCK" "$stale_lock" 2>/dev/null || exit 0
  if [ "$(cat "$stale_lock/owner" 2>/dev/null)" != "$recorded_owner" ]; then
    mv "$stale_lock" "$LOCK" 2>/dev/null
    exit 0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') cron_tick: cleared stale lock (owner: ${recorded_owner:-none recorded})" >> "$LOG"
  rm -f "$stale_lock/owner"
  rmdir "$stale_lock"
  claim_lock || exit 0
fi
trap release_lock EXIT

cd "$DIR" || exit 1
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  /usr/bin/python3 tick.py auto 2>&1 | grep -vE '^\|'   # log status lines, skip the wide table
} >> "$LOG" 2>&1

# Keep the log bounded.
tail -n 3000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
