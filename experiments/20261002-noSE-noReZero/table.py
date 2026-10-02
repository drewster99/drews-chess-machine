#!/usr/bin/env python3
"""Per-1000-step table: no-SE with ReZero (SE experiment seeds 1 and 2) vs no-SE
without ReZero (this run). pElo / NLL from `--probe-set wide` probes; blank where
a run never reached the step.

Buffer plies/game is the average game length in the 500k-position replay buffer
at that step: 500,000 / (games fed since the buffer's oldest position was fed),
from the SE experiment's ReLU scale+bias run's [REPLAY] lines (complete to 33k).
Every arm replays the same corpus in the same order with the same feed per step,
so it is the same for every arm; the script stops if this run's log disagrees
where both exist."""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "documentation", "dashboards", "data")
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
BUFFER = 500_000
BASELINE_LOG = "dcm_log_20260929-150727.txt"
NO_REZERO_LOG = "dcm_log_20261002-011124.txt"


def csv_points(run):
    points = {}
    with open(os.path.join(DATA, f"{run}.csv")) as handle:
        for row in csv.DictReader(handle):
            if row.get("pElo"):
                points[int(float(row["cum_step"]))] = (float(row["pElo"]), float(row["nll"]))
    return points


def probe_points(file_name):
    points = {}
    path = os.path.join(HERE, file_name)
    if os.path.exists(path):
        for line in open(path):
            record = json.loads(line)
            points[record["step"]] = (record["pElo"], record["nll"])
    return points


def buffer_plies_per_game(log_name):
    feed = []
    pattern = re.compile(r"\[REPLAY\] step=(\d+) .* plies=(\d+) games=(\d+)")
    for line in open(os.path.join(LOGS, log_name), errors="replace"):
        match = pattern.search(line)
        if match:
            feed.append(tuple(int(x) for x in match.groups()))
    result = {}
    for step, plies, games in feed:
        if step % 1000 or plies < BUFFER:
            continue
        older = [f for f in feed if f[1] <= plies - BUFFER]
        if older:
            _, old_plies, old_games = older[-1]
            result[step] = (plies - old_plies) / (games - old_games)
    return result


def main():
    arms = [("no SE + ReZero s1", csv_points("se_none")), ("no SE, no ReZero s1", probe_points("probes.jsonl")),
            ("no SE, no ReZero s2", probe_points("probes-seed2.jsonl")),
            ("no SE + ReZero s2", csv_points("se_none2"))]
    plies = buffer_plies_per_game(BASELINE_LOG)
    for step, value in buffer_plies_per_game(NO_REZERO_LOG).items():
        if step in plies and abs(plies[step] - value) > 0.05:
            raise SystemExit(f"buffer plies/game differs at {step}: baseline {plies[step]:.2f} vs no-ReZero {value:.2f}")
    last = max((max(p) for _, p in arms if p), default=0)
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
