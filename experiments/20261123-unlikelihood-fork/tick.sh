#!/bin/zsh
# One 1000-step tick of the unlikelihood fork: comparison table row(s) vs R-elitecyc, dashboards refreshed, committed.
set -eu
STEP=$1; R=/Users/andrew/cursor/drews-chess-machine; X=$R/experiments/20261123-unlikelihood-fork
python3 $X/table.py ${2:-$STEP} $STEP
$R/experiments/20261008-zlra-selfplay-lr/tables/refresh.sh >/dev/null
B="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2701-e5e79ab2.app/Contents/MacOS/DrewsChessMachine"
DCM_BIN="$B" DCM_BINARY="$B" /usr/bin/python3 $R/documentation/dashboards/refresh_dashboard.py 2>&1 | grep -i 'new segment\|new run\|error' || true
cd $R
F=(dcm_master.html documentation/artifacts/dcm-master-dashboard.html documentation/artifacts/dcm-training-step-tables.html
   documentation/dashboards/dcm_master_hosted.html documentation/dashboards/data/relitecyc_unlik.csv
   experiments/20261008-zlra-selfplay-lr/tables/training_tables.html experiments/20261008-zlra-selfplay-lr/tables/wide_pelo_record.json
   experiments/20261123-unlikelihood-fork/large_probe.jsonl)
for f in "${F[@]}"; do [[ -e $f ]] && git add "$f"; done
git commit -qm "R-elitecyc-unlik $STEP: large-set probe; dashboards refreshed" -- $(for f in "${F[@]}"; do [[ -e $f ]] && echo $f; done) && git push -q
git log --oneline -1
