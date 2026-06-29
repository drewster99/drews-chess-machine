#!/bin/sh
export DCM_FORCE_LAUNCH=1
export DCM_QUIET=1
exec /Users/andrew/cursor/drews-chess-machine/run_latest.sh --uci "$@"
