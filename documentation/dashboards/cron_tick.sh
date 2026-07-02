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

# Stale-lock guard: a prior run that died leaves the lock dir; clear it if >15 min old.
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then
  rmdir "$LOCK" 2>/dev/null
fi
# Atomic lock (mkdir is atomic) so overlapping fires can't collide on the CSV/render.
mkdir "$LOCK" 2>/dev/null || exit 0
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$DIR" || exit 1
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  /usr/bin/python3 tick.py auto 2>&1 | grep -vE '^\|'   # log status lines, skip the wide table
} >> "$LOG" 2>&1

# Keep the log bounded.
tail -n 3000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
