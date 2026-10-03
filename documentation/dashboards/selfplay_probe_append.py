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

"Append" is logical, not an `open(path, "a")`: this CSV is a source of truth (the
marks are not reconstructable once the log rotates away), and an in-place append
interrupted mid-row leaves a torn last line that the NEXT append would then glue
a new row onto. So the existing bytes are copied verbatim into a temp file, the
new rows are written after them, and the result replaces the original atomically
(see _atomic_write.py). The prefix is copied as text read with newline="" and
written with newline="", so the existing rows keep their exact bytes, CRLF row
terminators included.

The `segment` column is the index of the log within the run's registry `logs`
list. It is REQUIRED for correctness, not decoration: a lineage that restarted
from step 1 more than once reuses the same raw step numbers under different
cumulative bases, and without the column selfplay.py cannot tell them apart.

Only rows whose `model=` matches the run's base ModelID are kept, so a foreign
model interleaved in a shared log cannot contaminate the curve.

Usage:  python3 selfplay_probe_append.py <run> [--segment N] [--spacing 250]
        (segment defaults to the LAST entry in the run's `logs` list)
"""
import os, re, io, csv, sys, json, argparse
# Crash-safe, compare-and-swap replace of selfplay_probe/<run>.csv that may only extend it.
from _guarded_csv import read_text, replace_text_if_unchanged, snapshot_of

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
    # Last step of this launch on the kept chain (registry `log_kept_to`); marks
    # past it come from a tail a later resume abandoned.
    kept_to = cfg.get("log_kept_to", {}).get(logs[seg])

    path = os.path.join(HERE, "selfplay_probe", f"{a.run}.csv")
    if os.path.exists(path):
        existing_text, snapshot = read_text(path)
    else:
        existing_text, snapshot = "", snapshot_of(path)
    rows = list(csv.DictReader(io.StringIO(existing_text, newline="")))
    last = max((int(r["step"]) for r in rows if r.get("segment") == str(seg)), default=-1)

    new_rows = []
    for line in open(log, errors="replace"):
        if "set=wide" not in line:
            continue
        m = LINE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if step <= last or (last >= 0 and step - last < a.spacing):
            continue
        if base and not m.group(4).startswith(base):
            continue
        if kept_to is not None and step > kept_to:
            continue
        new_rows.append({"step": step, "pElo": m.group(3), "nll": m.group(2), "segment": seg})
        last = step

    # Rewrite only when there is something to add (or the file does not exist
    # yet / is empty and needs its header), so an idle tick leaves the file alone.
    if new_rows or not existing_text:
        buffer = io.StringIO(newline="")
        buffer.write(existing_text)
        w = csv.DictWriter(buffer, fieldnames=FIELDS)
        if not existing_text:
            w.writeheader()
        for row in new_rows:
            w.writerow(row)
        replace_text_if_unchanged(path, buffer.getvalue(), snapshot, must_extend=True)
    print(f"{a.run} seg{seg} ({logs[seg]}): +{len(new_rows)} marks, now through step {last}")


if __name__ == "__main__":
    main()
