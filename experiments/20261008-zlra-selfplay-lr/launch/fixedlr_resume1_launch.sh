#!/bin/zsh
# R-fixedlr segment 1 (owner 2026-10-10: move every trainer to the build with GPU submission labels and the flight
# recorder). --train can't resume a session, so segment 0 (build 2440, --train) was ended with SIGUSR2 at 11:33:45
# (save 20261010-163339-20261008-7-peGM-sigusr2.dcmsession, buffer omitted; the screen was locked, so the "Include
# replay buffer" toggle was out of reach) and build 2491 was opened plainly: the launch sheet auto-resumes the
# LastSessionPointer's session after 30 s and starts Play-and-Train with the session's own parameters (fixed LR 0.01,
# momentum 0.90, arenas every 400 s). Resume verdict: NOT EXACT: buffer (refilled from new games). No results.json:
# a plain GUI run writes none.
set -u
E=/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr
APP="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2491-6fe1dcd5.app"
open -n "$APP" --stdout $E/train-fixedlr-seg1.stdout --stderr $E/train-fixedlr-seg1.stdout
