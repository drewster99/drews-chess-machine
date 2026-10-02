#!/bin/sh
# Full-model dead / stuck / always-on analysis. CPU only (numpy + pandas); reads
# checkpoints from ~/Library/Application Support/DrewsChessMachine/Models and
# writes everything under ../results. Uses the latest enumerated leaky-FC1
# checkpoint present at run time.
set -e
cd "$(dirname "$0")"
python3 layout_check.py
python3 units.py
python3 input_features.py
python3 summarize.py
