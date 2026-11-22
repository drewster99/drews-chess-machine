#!/bin/zsh
# One 1000-step tick: report the arms' table through step $1, refresh both dashboards, commit and push.
# Publishing the two artifacts is a separate step (the session does it).
set -eu
STEP=$1; R=/Users/andrew/cursor/drews-chess-machine; X=$R/experiments/20261122-label-smoothing-mode
python3 $X/report.py 1000 $STEP
$R/experiments/20261008-zlra-selfplay-lr/tables/refresh.sh >/dev/null
B="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2687-d8afa524.app/Contents/MacOS/DrewsChessMachine"
DCM_BIN="$B" DCM_BINARY="$B" /usr/bin/python3 $R/documentation/dashboards/refresh_dashboard.py 2>&1 | grep -i 'new segment\|new run\|error' || true
cd $R
F=(dcm_master.html documentation/artifacts/dcm-master-dashboard.html documentation/artifacts/dcm-training-step-tables.html
   documentation/dashboards/dcm_master_hosted.html documentation/dashboards/data/lsmodeA.csv documentation/dashboards/data/lsmodeB.csv
   documentation/dashboards/data/lsmodeC.csv experiments/20261008-zlra-selfplay-lr/tables/training_tables.html
   experiments/20261008-zlra-selfplay-lr/tables/wide_pelo_record.json
   experiments/20261122-label-smoothing-mode/large_probe_A.jsonl experiments/20261122-label-smoothing-mode/large_probe_B.jsonl
   experiments/20261122-label-smoothing-mode/large_probe_C.jsonl)
git add "${F[@]}"
git commit -qm "Label-smoothing mode arms: step $STEP probes; dashboards refreshed" -- "${F[@]}" && git push -q
git log --oneline -1
