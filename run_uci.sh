#!/bin/sh
# UCI engine for cutechess and other GUIs. run_latest.sh prints which binary
# and build it launches to stderr; stdout stays the UCI protocol.
export DCM_FORCE_LAUNCH=1
exec /Users/andrew/cursor/drews-chess-machine/run_latest.sh --uci "$@"
