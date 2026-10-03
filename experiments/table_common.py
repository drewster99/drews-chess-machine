"""Helpers the experiment table scripts share: dashboard CSV points and the replay
buffer's average game length. One copy, so the tables cannot drift apart."""
import bisect
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
    """{step: average plies per game in the replay buffer} at every 1000-step mark.

    At a mark where `plies` positions have been fed, the buffer holds the last BUFFER
    of them, so its average game length is BUFFER / (games fed since the feed stood at
    plies - BUFFER). `[REPLAY]` lines arrive every 50 steps (about 420k plies apart), so
    the games count at that boundary is interpolated linearly between the two lines that
    bracket it. A mark whose boundary falls before the first logged line has no value."""
    feed = []
    pattern = re.compile(r"\[REPLAY\] step=(\d+) .* plies=(\d+) games=(\d+)")
    for line in open(os.path.join(LOGS, log_name), errors="replace"):
        match = pattern.search(line)
        if match:
            feed.append(tuple(int(x) for x in match.groups()))
    plies_axis = [plies for _, plies, _ in feed]
    if plies_axis != sorted(plies_axis):
        raise ValueError(f"{log_name}: [REPLAY] plies= values are not increasing; more than one run in this log?")
    result = {}
    for step, plies, games in feed:
        if step % 1000:
            continue
        boundary = plies - BUFFER
        games_at_boundary = games_fed_at(feed, plies_axis, boundary)
        if games_at_boundary is not None:
            result[step] = BUFFER / (games - games_at_boundary)
    return result


def games_fed_at(feed, plies_axis, plies):
    """Games fed when the feed stood at `plies`, interpolated between bracketing lines;
    None when `plies` lies outside the logged range."""
    index = bisect.bisect_left(plies_axis, plies)
    if index < len(plies_axis) and plies_axis[index] == plies:
        return feed[index][2]
    if index == 0 or index == len(plies_axis):
        return None
    (_, p0, g0), (_, p1, g1) = feed[index - 1], feed[index]
    return g0 + (g1 - g0) * (plies - p0) / (p1 - p0)
