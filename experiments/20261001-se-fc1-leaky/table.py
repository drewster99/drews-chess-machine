#!/usr/bin/env python3
"""Per-1000-step table: the three SE-experiment seed-1 arms vs leaky-FC1.

pElo / NLL for the SE arms come from the dashboard CSVs (their enumerated
checkpoints probed with `--probe-set wide`); the leaky-FC1 arm's come from
probes.jsonl (written by probe_loop.sh, same probe set). A cell is blank where
a run never reached that step.

Buffer plies/game is the average game length in the 500k-position replay
buffer at that step: 500,000 / (games fed since the buffer's oldest position
was fed), from the ReLU scale+bias run's [REPLAY] lines (complete to 33k). All
arms replay the same corpus in the same order with the same feed per step, so
it is the same for every arm; the script stops if the leaky run's log
disagrees where both exist.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "documentation", "dashboards", "data")
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
BUFFER = 500_000
BASELINE_LOG = "dcm_log_20260929-150727.txt"
LEAKY_LOG = "dcm_log_20261001-151822.txt"

SE_ARMS = [("ReLU scale+bias", "se_sb"), ("ReLU attenuate-only", "se_att"), ("ReLU no SE", "se_none")]


def csv_points(run):
    points = {}
    with open(os.path.join(DATA, f"{run}.csv")) as handle:
        for row in csv.DictReader(handle):
            if row.get("pElo"):
                points[int(float(row["cum_step"]))] = (float(row["pElo"]), float(row["nll"]))
    return points


def leaky_points():
    points = {}
    path = os.path.join(HERE, "probes.jsonl")
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


def cell(points, step, index, fmt):
    return fmt.format(points[step][index]) if step in points else ""


def main():
    # The leaky-FC1 arm sits right after its direct comparator, ReLU scale+bias.
    se = [(label, csv_points(run)) for label, run in SE_ARMS]
    arms = [se[0], ("leaky FC1 scale+bias", leaky_points())] + se[1:]
    plies = buffer_plies_per_game(BASELINE_LOG)
    leaky_plies = buffer_plies_per_game(LEAKY_LOG)
    for step, value in leaky_plies.items():
        if step in plies and abs(plies[step] - value) > 0.05:
            raise SystemExit(f"buffer plies/game differs at {step}: ReLU {plies[step]:.2f} vs leaky {value:.2f}")
    last = max(max(points) for _, points in arms if points)
    header = ["step", "buffer plies/game"] + [f"pElo {label}" for label, _ in arms] + [f"NLL {label}" for label, _ in arms]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for step in range(1000, last + 1, 1000):
        row = [f"{step:,}", f"{plies[step]:.1f}" if step in plies else ""]
        row += [cell(points, step, 0, "{:.1f}") for _, points in arms]
        row += [cell(points, step, 1, "{:.4f}") for _, points in arms]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
