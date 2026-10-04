#!/bin/zsh
# Probe the untrained starting nets of the 2026-09-29 – 10-03 architecture and label
# smoothing runs (R1–R16) with --probe-set wide, the probe the runs' 1k checkpoints get,
# and append one JSON line per net to step0-probes.jsonl: the step-0 row of their tables.
# Each net is probed with the build its run's probes used (2275; R9's format-6 net needs
# 2290). A net already in the output file is skipped, so the script can be re-run.
set -u
HERE=${0:A:h}
OUT=$HERE/step0-probes.jsonl
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
FB="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds"
B2275="$FB/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine"
B2290="$FB/DCM-2290-9a36f9f.app/Contents/MacOS/DrewsChessMachine"
# runs|start net|binary
NETS=(
  "R1 R10 R12|20260929-test_SE_scale+bias-fresh|$B2275"
  "R2 R11|20260929-test_SE_scale+bias-seed2-fresh|$B2275"
  "R3|20260929-test_SE_attenuate-only-fresh|$B2275"
  "R4|20261001-test_SE_scale+bias-fc1leaky-fresh|$B2275"
  "R5|20260929-test_SE_none-fresh|$B2275"
  "R6|20260929-test_SE_none-seed2-fresh|$B2275"
  "R7|20261002-bench_v5s3_noSE_noReZero-fresh|$B2275"
  "R8|20261002-bench_v5s3_noSE_noReZero-seed2-fresh|$B2275"
  "R9|20260929-test_SE_none-rz0cap1-fresh|$B2290"
  "R13|20261003-fatty216-b2275-fresh|$B2275"
  "R14|20261003-skinny48-b2275-fresh|$B2275"
  "R15|20261003-fatty224s3-b2275-fresh|$B2275"
  "R16|20261004-fatconv98-b2275-fresh|$B2275"
)
for entry in $NETS; do
  runs=${entry%%|*}; rest=${entry#*|}; net=${rest%%|*}; bin=${rest#*|}
  if [ -f "$OUT" ] && grep -q "\"net\": \"$net\"" "$OUT"; then
    echo "skip $net (already probed)"; continue
  fi
  f="$M/$net.safetensors"
  [ -f "$f" ] || { echo "MISSING $f" >&2; exit 1; }
  raw=$("$bin" --probe-model "$f" --probe-set wide 2>/dev/null); rc=$?
  if [ $rc != 0 ]; then echo "probe of $net exited $rc" >&2; exit 1; fi
  build=$(basename "${bin:h:h:h}" .app)
  print -r -- "$raw" | python3 -c '
import json, sys
runs, net, build = sys.argv[1], sys.argv[2], sys.argv[3]
lines = [l for l in sys.stdin.read().splitlines() if l.strip().startswith("{")]
if len(lines) != 1:
    sys.exit(f"{net}: expected one JSON summary line, got {len(lines)}")
p = json.loads(lines[0])
print(json.dumps({"step": 0, "runs": runs, "net": net, "modelID": p.get("modelID"),
                  "pElo": p.get("pElo"), "nll": p.get("nll"), "build": build}))
' "$runs" "$net" "$build" >> "$OUT" || exit 1
  echo "probed $net ($runs)"
done
