#!/usr/bin/env python3
"""Emit the full continuous per-checkpoint table for the v5 corpus-replay run.

Run 1 (PID 17249, ended step 268506) and run 2 (PID 7022, counter restarted at 1)
are stitched onto one cumulative step axis: cum = 268506 + run2_step. Probe values
come from the two .jsonl files; loss/pLoss/vLoss/gNorm come from the [REPLAY] log
line at that exact step in the matching session log.
"""
import glob
import json
import os
import re

MON = os.path.dirname(os.path.abspath(__file__))
OFFSET = 268506  # run 1's final step; run 2's counter restarts at 1
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")

BASE_ROW = ("base", 1584.2, 47.8, 2.097, None, None, None, None, 14.86)
HDR = "| step | pElo | top1% | NLL | loss | pLoss | vLoss | gNorm | logitAbsMax |"
SEP = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
REPEAT_HEADER_BEFORE_LAST = 20


def load_probes(filename, offset):
    """step -> probe dict, keyed by cumulative step."""
    out = {}
    path = os.path.join(MON, filename)
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = re.search(r"step(\d+)", d["model"])
        if m:
            out[int(m.group(1)) + offset] = d
    return out


def load_log(pattern, offset):
    """step -> (loss, pLoss, vLoss, gNorm), keyed by cumulative step."""
    out = {}
    for path in glob.glob(os.path.join(LOGS, pattern)):
        for line in open(path, errors="ignore"):
            if "[REPLAY] step=" not in line:
                continue

            def field(key):
                m = re.search(key + r"=([-0-9.]+)", line)
                return float(m.group(1)) if m else None

            step = field("step")
            if step is None:
                continue
            out[int(step) + offset] = (
                field("loss"), field("pLoss"), field("vLoss"), field("gNorm"))
    return out


def fmt(value, places):
    return "—" if value is None else ("%." + str(places) + "f") % value


def render(row):
    step, pelo, top1, nll, loss, ploss, vloss, gnorm, logit = row
    return "| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        step, pelo, top1, fmt(nll, 3), fmt(loss, 3), fmt(ploss, 3),
        fmt(vloss, 3), fmt(gnorm, 2), fmt(logit, 2))


probes = load_probes("new_ckpts.jsonl", 0)
probes.update(load_probes("new_ckpts_run2.jsonl", OFFSET))

logs = load_log("dcm_log_20260702-*.txt", 0)
logs.update(load_log("dcm_log_20260713-*.txt", OFFSET))

rows = [BASE_ROW]
for cum in sorted(probes):
    d = probes[cum]
    top1 = 100 * d["argmaxCorrect"] / d["n"]
    loss, ploss, vloss, gnorm = logs.get(cum, (None,) * 4)
    label = "%gk" % round(cum / 1000, 1)
    rows.append((label, round(d["pElo"]), round(top1, 1), round(d["nll"], 3),
                 loss, ploss, vloss, gnorm, round(d["policy_logit_abs_max"], 2)))

out = [HDR, SEP]
repeat_at = len(rows) - REPEAT_HEADER_BEFORE_LAST
for i, row in enumerate(rows):
    if i == repeat_at:
        out += [HDR, SEP]
    out.append(render(row))
print("\n".join(out))

# Summary line: n, best, latest, logit trend.
records = json.load(open(os.path.join(MON, "records.json")))
best_cum = max(probes, key=lambda k: probes[k]["pElo"])
last_cum = max(probes)
best, last = probes[best_cum], probes[last_cum]
print()
print("%d checkpoints · best %gk = %d · latest %gk = %d · logitAbsMax %.1f · "
      "records: pElo %.0f @%dk, NLL %.4f @%dk" % (
          len(rows), round(best_cum / 1000, 1), round(best["pElo"]),
          round(last_cum / 1000, 1), round(last["pElo"]),
          last["policy_logit_abs_max"],
          records["best_pelo"], records["best_pelo_step"],
          records["low_nll"], records["low_nll_step"]))
