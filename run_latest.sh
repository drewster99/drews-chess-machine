#!/bin/bash
PROG="$(basename "$0")"
# Refuse to launch a second DrewsChessMachine instance — macOS LaunchServices
# would normally refocus the running app, but execing the binary directly
# bypasses that and silently spawns a concurrent process, which wastes the
# GPU and skews any ongoing training. Override with DCM_FORCE_LAUNCH=1 if
# you truly need two instances.
if [ "${DCM_FORCE_LAUNCH:-0}" != "1" ] && pgrep -x DrewsChessMachine >/dev/null 2>&1; then
    echo "$PROG: DrewsChessMachine is already running (set DCM_FORCE_LAUNCH=1 to override)" >&2
    exit 10
fi
# Pick the newer of Release vs Debug by mtime — whichever build
# configuration was most recently produced wins. nullglob so a
# missing configuration silently drops out of the candidate list
# rather than feeding an unmatched literal pattern into stat.
shopt -s nullglob
candidates=()
for path in ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Debug/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine \
            ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine; do
    [ -x "$path" ] && candidates+=("$path")
done
if [ ${#candidates[@]} -eq 0 ]; then
    echo "$PROG: no DrewsChessMachine binary found in Debug or Release" >&2
    exit 1
fi
NEWEST=""
NEWEST_MTIME=0
for path in "${candidates[@]}"; do
    mtime=$(stat -f %m "$path" 2>/dev/null || echo 0)
    if [ "$mtime" -gt "$NEWEST_MTIME" ]; then
        NEWEST_MTIME="$mtime"
        NEWEST="$path"
    fi
done
echo "$PROG: launching $NEWEST" >&2
exec "$NEWEST" "$@"
