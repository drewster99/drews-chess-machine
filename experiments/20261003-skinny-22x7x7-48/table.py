#!/usr/bin/env python3
"""Per-1000-step table for the depth sweep at the no-SE/no-ReZero budget: fatty
(1 x 7x7 @216), skinny (22 x 7x7 @48) and the baseline Avg(R7,R8) — the mean of the two
no-SE/no-ReZero seeds (3 x 7x7 @128) at each step, blank where either seed lacks the
step. Columns: three pElo, then three NLL, in that order.

Probe files are read through `probe_record.arm_points` with each run's model_id, so a
file holding another run's records is refused rather than tabulated; an arm whose run
has not written a checkpoint yet is declared NOT_STARTED."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(HERE, "..")
sys.path.insert(0, EXPERIMENTS)
import probe_record  # noqa: E402

NO_REZERO = os.path.join(EXPERIMENTS, "20261002-noSE-noReZero")
R7 = (os.path.join(NO_REZERO, "probes.jsonl"), "20261002-2-5tKN", "R7 no SE, no ReZero s1")
R8 = (os.path.join(NO_REZERO, "probes-seed2.jsonl"), "20261002-4-T79u", "R8 no SE, no ReZero s2")
# The run model_ids are filled in from each run's log once it has started.
FATTY = (os.path.join(EXPERIMENTS, "20261003-fatty-1x7x7-216", "probes.jsonl"), probe_record.NOT_STARTED, "fatty")
SKINNY = (os.path.join(HERE, "probes.jsonl"), probe_record.NOT_STARTED, "skinny")


STEP0 = os.path.join(EXPERIMENTS, "step0-probes.jsonl")


def step0(start_model_id):
    """(pElo, nll) of the untrained start net, from `probe_step0.sh`'s output."""
    import json
    if not os.path.exists(STEP0):
        return None
    for line in open(STEP0):
        if line.strip():
            rec = json.loads(line)
            if rec["modelID"] == start_model_id:
                return (rec["pElo"], rec["nll"])
    return None


START_IDS = {"fatty": "20261004-8-2Sao", "skinny": "20261004-9-czp5",
             "R7 no SE, no ReZero s1": "20261002-1-bh2u", "R8 no SE, no ReZero s2": "20261002-3-x4gI"}


def points(arm):
    path, model_id, label = arm
    pts = dict(probe_record.arm_points(path, model_id, label) or {})
    zero = step0(START_IDS[label])
    if zero is not None:
        pts[0] = zero
    return pts


def mean_of(a, b, step, index):
    if step not in a or step not in b:
        return None
    x, y = a[step][index], b[step][index]
    if x is None or y is None:
        return None
    return (x + y) / 2


def main():
    fatty, skinny, r7, r8 = points(FATTY), points(SKINNY), points(R7), points(R8)
    last = max(list(fatty) + list(skinny))
    if last < 0:
        print("no fatty or skinny probe yet")
        return
    print("| step | pElo fatty | pElo skinny | pElo Avg(R7,R8) | NLL fatty | NLL skinny | NLL Avg(R7,R8) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for step in range(0, last + 1000, 1000):
        avg_pelo, avg_nll = mean_of(r7, r8, step, 0), mean_of(r7, r8, step, 1)
        row = [f"{step:,}",
               probe_record.pelo_cell(fatty, step), probe_record.pelo_cell(skinny, step),
               "" if avg_pelo is None else f"{avg_pelo:.1f}",
               probe_record.nll_cell(fatty, step), probe_record.nll_cell(skinny, step),
               "" if avg_nll is None else f"{avg_nll:.4f}"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
