#!/usr/bin/env python3
"""Per-1000-step table for the label-smoothing arms against their shared baseline.

Baseline: the SE experiment's ReLU scale+bias seed 1 (`se_sb`, policy ε 0.1, value
ε 0.013), from the dashboard CSV. Arm C (policy ε 0.03) and arm D (value ε 0) start
from the same fresh net and come from their probes.jsonl (`--probe-set wide`). A
cell is blank where a run never reached the step.

Buffer plies/game is the average game length in the 500k-position replay buffer at
that step, from the baseline's [REPLAY] lines; every arm replays the same corpus in
the same order with the same feed per step.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "20261001-se-fc1-leaky"))
import table as leaky_table  # noqa: E402  (csv_points, buffer_plies_per_game, BASELINE_LOG)


def probe_points(path):
    import json
    points = {}
    if os.path.exists(path):
        for line in open(path):
            record = json.loads(line)
            points[record["step"]] = (record["pElo"], record["nll"])
    return points


def main():
    arms = [("baseline ε 0.1 / 0.013", leaky_table.csv_points("se_sb")),
            ("C policy ε 0.03", probe_points(os.path.join(HERE, "probes.jsonl"))),
            ("D value ε 0", probe_points(os.path.join(HERE, "..", "20261002-label-smoothing-D", "probes.jsonl")))]
    plies = leaky_table.buffer_plies_per_game(leaky_table.BASELINE_LOG)
    last = max(max(p) for _, p in arms if p)
    header = ["step", "buffer plies/game"] + [f"pElo {l}" for l, _ in arms] + [f"NLL {l}" for l, _ in arms]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for step in range(1000, last + 1, 1000):
        row = [f"{step:,}", f"{plies[step]:.1f}" if step in plies else ""]
        row += [f"{p[step][0]:.1f}" if step in p else "" for _, p in arms]
        row += [f"{p[step][1]:.4f}" if step in p else "" for _, p in arms]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
