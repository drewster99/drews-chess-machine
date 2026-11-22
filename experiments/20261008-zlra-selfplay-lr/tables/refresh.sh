#!/bin/zsh
# Regenerate the training step tables and copy them to documentation/artifacts/ (the repo copy of the published
# artifact https://claude.ai/artifact/JBDkj3xWshHd6RARZ1HwLe). Publishing the artifact itself is a separate step.
set -e
cd "${0:A:h}"
python3 tables_html.py
cp training_tables.html ../../../documentation/artifacts/dcm-training-step-tables.html
