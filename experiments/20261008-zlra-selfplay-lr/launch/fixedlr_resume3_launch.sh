#!/bin/zsh
# R-fixedlr resume (owner 2026-11-02 10:15 CST: "stop this run and continue the fixedlr self-play"). The last save is
# 20261025-011951-20261008-7-peGM-sigusr2.dcmsession (trainer step 109,221, build 2590), which is the
# LastSessionPointer target: the app's launch sheet auto-resumes it after its 30 s countdown. Same build as the save.
# Started 10:16 (open -n -g, then open -g to materialize the window); resumed 10:17:45 as lineage segment 5
# (run A2599216), [RESUME] NOT EXACT: buffer. Log dcm_log_20261102-101606.
set -u
E=/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr
APP="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2590-e42bfe04.app"
open -n -g "$APP" --stdout $E/train-fixedlr-seg3.stdout --stderr $E/train-fixedlr-seg3.stdout
