#!/bin/zsh
# R-fixedlr segment 2: segment 1 (build 2491) halted training on the 2026-10-10 21:15:26 GPU reset (its own arena
# hang; trainer tainted, no save after it) and was quit by the owner at 22:32. Build 2501 (GPU fault monitor fix) was
# opened plainly at 22:44: the launch sheet auto-resumed the last good save, 20261010-210325-20261008-7-peGM-promote
# .dcmsession (16:03 local), with the session's own parameters. Verdict: NOT EXACT: buffer. Arena code unchanged
# (owner). R-fixedlr runs on indefinitely (owner).
set -u
E=/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr
APP="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2501-eae1b55c.app"
open -n "$APP" --stdout $E/train-fixedlr-seg2.stdout --stderr $E/train-fixedlr-seg2.stdout
