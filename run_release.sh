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
exec ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine $@
