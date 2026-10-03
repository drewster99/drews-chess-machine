#!/bin/bash
# DCM per-mark monitoring tick — driven by the user's crontab (every 5 min), NOT by
# the assistant. Auto-detects the active replay-corpus run and updates its CSV +
# dashboard (freeze checkpoint -> GPU probe -> log-backfill -> render). No-op when
# nothing is training. Decouples checkpoint preservation + pElo tracking from any
# interactive session. Every tick leaves a line in the log, including a tick that
# found the lock held, so a stalled tick shows up as a growing run of "skipped"
# lines (and an [ALARM] once it has held the lock too long), never as silence.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"

# The dashboards folder is this script's own folder, so the tick works from any
# checkout of the repository. Until it is known there is no log to write to, so a
# failure to find it goes to stderr (cron mail) and the system log (`logger`).
report_unusable_directory() {
  echo "cron_tick: $1" >&2
  logger -t dcm_cron_tick "$1"
  exit 1
}
SCRIPT_REFERENCE="${BASH_SOURCE[0]}"
DIR="$(cd "$(dirname "$SCRIPT_REFERENCE")" && pwd -P)" \
  || report_unusable_directory "cannot resolve the script's folder from '$SCRIPT_REFERENCE'"
[ -f "$DIR/tick.py" ] && [ -f "$DIR/registry.json" ] \
  || report_unusable_directory "'$DIR' is not the dashboards folder (no tick.py / registry.json); run this file with bash"
LOG="$DIR/cron_tick.log"
LOCK="/tmp/dcm_cron_tick.lock"
LOCK_OWNER="$LOCK/owner"
# A lock held longer than this raises an [ALARM] line on every skipped tick. Nothing
# is killed: a long hold may be a slow but legitimate probe batch.
TICK_HELD_ALARM_SECONDS=1800
# The log is moved aside under a new timestamped name (never overwriting one) once it
# grows past this size.
LOG_ROTATE_BYTES=1048576

log_line() { echo "$(date '+%Y-%m-%d %H:%M:%S') cron_tick: $*" >> "$LOG"; }

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

lock_age_seconds() {
  # The lock folder's modification time is when its owner record was written.
  lock_mtime=$(stat -f %m "$LOCK" 2>/dev/null) || return 1
  echo $(( $(date +%s) - lock_mtime ))
}

claim_lock() {
  mkdir "$LOCK" 2>/dev/null || return 1
  if ! process_identity $$ > "$LOCK_OWNER"; then
    log_line "could not record lock owner; releasing"
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
  lock_age=$(lock_age_seconds) || lock_age=unknown
  if [ -n "$recorded_owner" ]; then
    recorded_pid=${recorded_owner%% *}
    if kill -0 "$recorded_pid" 2>/dev/null \
       && [ "$(process_identity "$recorded_pid")" = "$recorded_owner" ]; then
      log_line "skipped: the lock is held by a live tick (owner: $recorded_owner, held ${lock_age}s)"
      if [ "$lock_age" != unknown ] && [ "$lock_age" -gt "$TICK_HELD_ALARM_SECONDS" ]; then
        tick_group=$(ps -o pgid= -p "$recorded_pid" | tr -d ' ')
        alarm="[ALARM] tick pid $recorded_pid has held the lock ${lock_age}s (over ${TICK_HELD_ALARM_SECONDS}s); not killed. Its process group: $(ps -o pid=,etime=,command= -g "$tick_group" | tr '\n' ';')"
        log_line "$alarm"
        echo "cron_tick: $alarm" >&2
      fi
      exit 0
    fi
  elif [ -z "$(find "$LOCK" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then
    log_line "skipped: the lock has no owner record yet (age ${lock_age}s); its holder may be starting"
    exit 0
  fi
  # Stale. Break it by renaming (atomic, so only one breaker can win), then make
  # sure what we moved is the lock we judged stale and not one re-acquired since.
  stale_lock="$LOCK.stale.$$"
  mv "$LOCK" "$stale_lock" 2>/dev/null || { log_line "skipped: another tick is breaking the stale lock"; exit 0; }
  if [ "$(cat "$stale_lock/owner" 2>/dev/null)" != "$recorded_owner" ]; then
    mv "$stale_lock" "$LOCK" 2>/dev/null
    log_line "skipped: the lock changed hands while being judged stale"
    exit 0
  fi
  log_line "cleared stale lock (owner: ${recorded_owner:-none recorded})"
  rm -f "$stale_lock/owner"
  rmdir "$stale_lock"
  claim_lock || { log_line "skipped: another tick claimed the lock after it was cleared"; exit 0; }
fi
trap release_lock EXIT

cd "$DIR" || { log_line "cannot enter $DIR"; exit 1; }
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  /usr/bin/python3 tick.py auto 2>&1 | grep -vE '^\|'   # log status lines, skip the wide table
  tick_status=${PIPESTATUS[0]}
  [ "$tick_status" -eq 0 ] || echo "$(date '+%Y-%m-%d %H:%M:%S') cron_tick: tick.py exited with status $tick_status"
} >> "$LOG" 2>&1

# Keep the live log bounded by moving it aside (an append racing the move lands in
# whichever file it opened; nothing is truncated or overwritten).
log_size=$(stat -f %z "$LOG" 2>/dev/null) || log_size=0
if [ "$log_size" -gt "$LOG_ROTATE_BYTES" ]; then
  rotated="$DIR/cron_tick-$(date '+%Y%m%d-%H%M%S').log"
  mv -n "$LOG" "$rotated"
  if [ -e "$LOG" ]; then
    log_line "log rotation skipped: $rotated already exists"
  fi
fi
# A failed tick also exits non-zero, so cron reports it beyond the log line.
exit "$tick_status"
