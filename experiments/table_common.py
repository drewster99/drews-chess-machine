"""Helpers the experiment table scripts share: dashboard CSV points and the replay
buffer's average game length. One copy, so the tables cannot drift apart."""
import bisect
import csv
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documentation", "dashboards"))
from dcm_probe_build import UNRECORDED  # noqa: E402
from _schema import NON_FINITE_PELO_NOTE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "documentation", "dashboards", "data")
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
BUFFER = 500_000
# The SE experiment's ReLU scale+bias seed-1 run: its [REPLAY] lines are complete to
# 33k, and every replay arm feeds the same corpus in the same order with the same
# feed per step, so its buffer game length serves every arm.
BASELINE_LOG = "dcm_log_20260929-150727.txt"


def _is_probed(row):
    """Whether a dashboard CSV row holds a probe measurement: a pElo, or the marker of
    a probe that measured a non-finite pElo (whose pElo cell is blank). A row that was
    only backfilled from the log has neither and is not a measurement."""
    return bool(row.get("pElo")) or NON_FINITE_PELO_NOTE in (row.get("note") or "")


def csv_points(run):
    """{cum_step: (pElo or None, nll or None)} for the dashboard CSV rows of `run` that
    hold a probe measurement, in the shape `probe_record.load_probe_points` gives a
    probes.jsonl arm: pElo None is a measured non-finite value (shown as such by
    `probe_record.pelo_cell`), never a step the run did not reach; nll None is a
    blank nll cell."""
    points = {}
    with open(os.path.join(DATA, f"{run}.csv")) as handle:
        for row in csv.DictReader(handle):
            if _is_probed(row):
                pelo = float(row["pElo"]) if row.get("pElo") else None
                nll = float(row["nll"]) if row.get("nll") else None
                points[int(float(row["cum_step"]))] = (pelo, nll)
    return points


def csv_probe_builds(run):
    """The probe builds behind a dashboard CSV's measurements, the rows `csv_points`
    reads (UNRECORDED for rows written before the tracker recorded them)."""
    builds = set()
    with open(os.path.join(DATA, f"{run}.csv")) as handle:
        for row in csv.DictReader(handle):
            if _is_probed(row):
                builds.add(row.get("probe_build") or UNRECORDED)
    return builds


def print_probe_builds(arms_builds):
    """After a table: which builds produced each pElo column, and a warning when the
    columns do not share one recorded build (a difference between them then includes
    any offset between builds)."""
    print("\nProbe builds: " + "; ".join(f"{label}: {', '.join(sorted(builds))}" for label, builds in arms_builds))
    distinct = set().union(*(builds for _, builds in arms_builds)) if arms_builds else set()
    if len(distinct) > 1 or UNRECORDED in distinct:
        print("Columns are not all from one recorded probe build; a pElo difference between columns "
              "includes any offset between builds.")


def buffer_plies_per_game(log_name):
    """{step: average plies per game in the replay buffer} at every 1000-step mark.

    At a mark where `plies` positions have been fed, the buffer holds the last BUFFER
    of them, so its average game length is BUFFER / (games fed since the feed stood at
    plies - BUFFER). `[REPLAY]` lines are many plies apart, so the games count at that
    boundary is interpolated linearly between the two lines that bracket it. A mark whose boundary falls before the first logged line has no value."""
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


def buffer_plies_per_game_by_trainer_step(log_name):
    """{trainer step: average plies per game in the replay buffer} at every trainer step
    that is a multiple of 1000 — the steps a run saves at, and the steps new probe
    records carry as `step` (from architecture format v11 a checkpoint's name and
    `training_step` are the trainer step), so the two join directly.

    The same buffer arithmetic as `buffer_plies_per_game`, over the `[REPLAY]` lines
    that carry `trainerStep=` (lines from builds before that field are not read). Kept
    separate from `buffer_plies_per_game`, whose segment-step keys the dated experiment
    tables use: for a resumed segment the two keys differ by the segment's start."""
    feed = []
    steps = []
    pattern = re.compile(r"\[REPLAY\] step=\d+ .* plies=(\d+) games=(\d+)")
    trainer_pattern = re.compile(r" trainerStep=(\d+)")
    with open(os.path.join(LOGS, log_name), errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            trainer = trainer_pattern.search(line) if match else None
            if match and trainer:
                plies, games = int(match.group(1)), int(match.group(2))
                feed.append((int(trainer.group(1)), plies, games))
                steps.append(int(trainer.group(1)))
    plies_axis = [plies for _, plies, _ in feed]
    if plies_axis != sorted(plies_axis) or steps != sorted(steps):
        raise ValueError(f"{log_name}: [REPLAY] plies= or trainerStep= values are not increasing; "
                         f"more than one run in this log?")
    result = {}
    for trainer_step, plies, games in feed:
        if trainer_step % 1000:
            continue
        games_at_boundary = games_fed_at(feed, plies_axis, plies - BUFFER)
        if games_at_boundary is not None:
            result[trainer_step] = BUFFER / (games - games_at_boundary)
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
