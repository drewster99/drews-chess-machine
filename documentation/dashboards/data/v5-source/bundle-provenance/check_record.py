"""Evaluate the newest probe of a run segment against the standing records.

Emits ONE line on stdout, prefixed RECORD when a new pElo high or NLL low was set,
so the monitor's consumer can decide whether to push. Records are rewritten only
when actually beaten, via a temp file + atomic replace, so an interrupted write
cannot truncate records.json and lose the run's history.

Usage:  python3 -I check_record.py <jsonl-basename> <cumulative-offset>
        (defaults to run 3's file/offset for backward compatibility)
"""
import json
import os
import re
import sys

MON = os.path.dirname(os.path.abspath(__file__))

# Segment offsets: cum step = offset + raw step. See archive/MANIFEST.md section 1.
JSONL = sys.argv[1] if len(sys.argv) > 1 else "new_ckpts_run3.jsonl"
OFFSET = int(sys.argv[2]) if len(sys.argv) > 2 else 605116

records_path = os.path.join(MON, "records.json")
records = json.load(open(records_path))

with open(os.path.join(MON, JSONL)) as fh:
    latest = json.loads([l for l in fh if l.strip()][-1])

raw = int(re.search(r"step(\d+)", latest["model"]).group(1))
cum_k = (OFFSET + raw) / 1000.0
pelo = latest["pElo"]
nll = latest["nll"]
top1 = 100.0 * latest["argmaxCorrect"] / latest["n"]
logit = latest["policy_logit_abs_max"]

beaten = []
if pelo > records["best_pelo"]:
    beaten.append("pElo %.1f (was %.1f @%.1fk)" % (pelo, records["best_pelo"], records["best_pelo_step"]))
    records["best_pelo"], records["best_pelo_step"] = pelo, cum_k
if nll < records["low_nll"]:
    beaten.append("NLL %.4f (was %.4f @%.1fk)" % (nll, records["low_nll"], records["low_nll_step"]))
    records["low_nll"], records["low_nll_step"] = nll, cum_k

if beaten:
    tmp = records_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(records, fh, indent=2)
    os.replace(tmp, records_path)

status = "RECORD " + " and ".join(beaten) if beaten else "ok"
print("cum %.1fk (raw %d) pElo %.1f top1 %.1f NLL %.4f logit %.1f | %s"
      % (cum_k, raw, pelo, top1, nll, logit, status))
sys.exit(0)
