#!/usr/bin/env python3
"""Append new in-training wide-probe marks to selfplay_probe/<run>.csv.

The app logs a `[TACTICAL-LICHESS] tick set=wide` line every ~25 steps carrying
pElo + NLL for the CURRENT trainer weights on the 4,435-position puzzle set.
That is the only pElo trajectory a self-play run has (there are no per-1000-step
frozen checkpoints to probe), so it is the source `selfplay.py` merges into the
run CSV.

Append-only and idempotent: reads the highest `step` already recorded for the
target segment and only scans forward from there, so a tick costs one pass over
the live log instead of re-reading the whole multi-hundred-MB lineage.

The `segment` column is the index of the log within the run's registry `logs`
list. It is REQUIRED for correctness, not decoration: a lineage that restarted
from step 1 more than once reuses the same raw step numbers under different
cumulative bases, and without the column selfplay.py cannot tell them apart.

Only rows whose `model=` matches the run's base ModelID are kept, so a foreign
model interleaved in a shared log cannot contaminate the curve.

Usage:  python3 selfplay_probe_append.py <run> [--segment N] [--spacing 250]
        (segment defaults to the LAST entry in the run's `logs` list)
"""
import os, re, csv, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
LINE = re.compile(r"\[TACTICAL-LICHESS\] tick set=wide step=(\d+) .*?NLL=([\d.]+) pElo=(-?\d+) model=(\S+)")
FIELDS = ["step", "pElo", "nll", "segment"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--segment", type=int, default=None)
    ap.add_argument("--spacing", type=int, default=250,
                    help="minimum step gap between kept marks (the probe fires ~every 25)")
    a = ap.parse_args()

    reg = json.load(open(os.path.join(HERE, "selfplay_registry.json")))
    if a.run not in reg["runs"]:
        sys.exit(f"unknown run {a.run!r}; known: {', '.join(reg['runs'])}")
    cfg = reg["runs"][a.run]
    logs = cfg["logs"]
    seg = a.segment if a.segment is not None else len(logs) - 1
    if not 0 <= seg < len(logs):
        sys.exit(f"segment {seg} out of range for {len(logs)} logs")
    log = os.path.join(LOGDIR, logs[seg])
    if not os.path.exists(log):
        sys.exit(f"log not found: {log}")
    base = cfg.get("base_modelID")

    path = os.path.join(HERE, "selfplay_probe", f"{a.run}.csv")
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    last = max((int(r["step"]) for r in rows if r.get("segment") == str(seg)), default=-1)

    added = 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not rows:
            w.writeheader()
        for line in open(log, errors="replace"):
            if "set=wide" not in line:
                continue
            m = LINE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step <= last or step - last < a.spacing:
                continue
            if base and not m.group(4).startswith(base):
                continue
            w.writerow({"step": step, "pElo": m.group(3), "nll": m.group(2), "segment": seg})
            last = step
            added += 1
    print(f"{a.run} seg{seg} ({logs[seg]}): +{added} marks, now through step {last}")


if __name__ == "__main__":
    main()
