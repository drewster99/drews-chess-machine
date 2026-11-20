#!/bin/zsh
# Probe one R-elitecyc-fp32fix enumerated checkpoint on the large puzzle set and compare with R-elitecyc (bf16) at the same step.
set -u
STEP=$1
O="$HOME/Library/Application Support/DrewsChessMachine/OpsScripts"
B="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2687-d8afa524.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
X=/Users/andrew/cursor/drews-chess-machine/experiments
OUT=$X/20261008-zlra-selfplay-lr/analysis-fp32/fp32fix_large_probe_step${STEP}.jsonl
[[ -e $OUT ]] || "$B" --probe-model "$M/20261114-zlra-elitecyc-fp32fix-latest-step${STEP}.safetensors" \
  --probe-set-file "$O/lichess_probes_large.json" --probe-out "$OUT" > /dev/null 2>"$O/probe_${STEP}.err" || { echo "probe failed: $(tail -3 "$O/probe_${STEP}.err")"; exit 1; }
python3 - "$OUT" "$X/20261108-elitecyc-heldout/large_probe.jsonl" "$STEP" <<'P'
import json, sys, re
f = json.loads(open(sys.argv[1]).readline()); step = sys.argv[3]
b = next((json.loads(l) for l in open(sys.argv[2]) if re.search(rf"-step{step}\.safetensors", json.loads(l)["model"])), None)
fmt = lambda d: f"pElo {d['pElo']:.1f} nll {d['nll']:.4f} top1 {d['argmaxCorrect']} top5 {d['top5Correct']}"
print(f"step {step}: fp32fix {fmt(f)}")
print(f"step {step}: bf16    {fmt(b)}  | fp32-bf16 pElo {f['pElo']-b['pElo']:+.1f} nll {f['nll']-b['nll']:+.4f}" if b else "bf16: no checkpoint at this step")
P
