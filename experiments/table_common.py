"""Helpers the experiment table scripts share: dashboard CSV points and the replay
buffer's average game length. One copy, so the tables cannot drift apart."""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "documentation", "dashboards", "data")
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
BUFFER = 500_000
# The SE experiment's ReLU scale+bias seed-1 run: its [REPLAY] lines are complete to
# 33k, and every replay arm feeds the same corpus in the same order with the same
# feed per step, so its buffer game length serves every arm.
BASELINE_LOG = "dcm_log_20260929-150727.txt"


def csv_points(run):
    """{cum_step: (pElo, nll)} for the dashboard CSV rows of `run` that carry a pElo."""
    points = {}
    with open(os.path.join(DATA, f"{run}.csv")) as handle:
        for row in csv.DictReader(handle):
            if row.get("pElo"):
                points[int(float(row["cum_step"]))] = (float(row["pElo"]), float(row["nll"]))
    return points


def buffer_plies_per_game(log_name):
    """{step: average plies per game in the replay buffer} at every 1000-step mark."""
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
