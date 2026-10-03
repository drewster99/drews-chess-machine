#!/usr/bin/env python3
"""Per-1000-step table for the label-smoothing arms against their shared baseline.

Baselines: the SE experiment's ReLU scale+bias seed 1 (`se_sb`, policy ε 0.1, value
ε 0.013) and seed 2 (`se_sb2`, same settings; the comparator for C seed 2), from the
dashboard CSVs. Arm C (policy ε 0.03) and arm D (value ε 0) start
from the same fresh net and come from their probes.jsonl (`--probe-set wide`). A
cell is blank where a run never reached the step.

Buffer plies/game is the average game length in the 500k-position replay buffer at
that step, from the baseline's [REPLAY] lines; every arm replays the same corpus in
the same order with the same feed per step.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import probe_record  # noqa: E402
from table_common import (BASELINE_LOG, buffer_plies_per_game, csv_points, csv_probe_builds,  # noqa: E402
                          print_probe_builds)

# (label, probes file, model_id); NOT_STARTED until the run's first checkpoint exists.
PROBE_ARMS = [
    ("C policy ε 0.03", os.path.join(HERE, "probes.jsonl"), "20261002-5-WkQG"),
    ("C seed 2", os.path.join(HERE, "probes-seed2.jsonl"), "20261003-22-oLbF"),
    ("D value ε 0", os.path.join(HERE, "..", "20261002-label-smoothing-D", "probes.jsonl"), "20261003-21-yEjN"),
]


def main():
    arms = [("baseline ε 0.1 / 0.013", csv_points("se_sb")),
            ("baseline seed 2", csv_points("se_sb2"))]
    builds = [("baseline ε 0.1 / 0.013", csv_probe_builds("se_sb")),
              ("baseline seed 2", csv_probe_builds("se_sb2"))]
    not_started = []
    for label, path, model_id in PROBE_ARMS:
        points = probe_record.arm_points(path, model_id, label)
        if points is None:
            not_started.append(label)
        else:
            arms.append((label, points))
            builds.append((label, probe_record.probe_builds(path)))
    plies = buffer_plies_per_game(BASELINE_LOG)
    last = max(max(p) for _, p in arms if p)
    header = ["step", "buffer plies/game"] + [f"pElo {l}" for l, _ in arms] + [f"NLL {l}" for l, _ in arms]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for step in range(1000, last + 1, 1000):
        row = [f"{step:,}", f"{plies[step]:.1f}" if step in plies else ""]
        row += [probe_record.pelo_cell(p, step) for _, p in arms]
        row += [f"{p[step][1]:.4f}" if step in p else "" for _, p in arms]
        print("| " + " | ".join(row) + " |")
    print_probe_builds(builds)
    if not_started:
        print(f"\nNot started: {', '.join(not_started)}.")


if __name__ == "__main__":
    main()
