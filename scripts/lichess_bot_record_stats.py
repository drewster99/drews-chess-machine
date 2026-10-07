#!/usr/bin/env python3
"""Independent check of the Lichess bot's Record card statistics.

A separate implementation of documentation/plans-active/
LICHESS_BOT_RECORD_STATS_PLAN.md section 3, over the bot's game records
(LichessBot/Games/**/*.json). It never writes anything: it reads the records
and prints the numbers the app logs on every index recompute, in the same
shape as the app's line

    [LICHESS-BOT] record stats (index): games=... W-D-L=... score=...
        perf=... rating=... brier@20=... ms=...

so the two can be compared (plan section 8.1): games, W-D-L, score and the
rating change must agree exactly, Perf within 1, Brier at move 20 to three
decimals (the records hold Float probabilities).

Usage:
    scripts/lichess_bot_record_stats.py [--games DIR] [--now ISO8601] [--tz ZONE]
                                        [--filter all|rated|casual]

--games defaults to ~/Library/Application Support/DrewsChessMachine/
LichessBot/Games. --now defaults to the current time, --tz to the system
time zone (periods are anchored on each game's start in that zone; the week
starts on Monday unless --sunday-weeks is given, matching the system locale
the app uses).
"""

import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

HELD = 0.80
CHECKPOINTS = (10, 20, 40)


def system_zone_name():
    """The system's IANA zone (macOS: /etc/localtime links into zoneinfo).
    A named zone, not a fixed offset, so DST boundaries come out right."""
    target = os.path.realpath("/etc/localtime")
    marker = "/zoneinfo/"
    if marker not in target:
        sys.exit("cannot tell the system time zone from %s; pass --tz" % target)
    return target.split(marker, 1)[1]


def parse_date(text):
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_records(directory):
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        with open(path, "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def expected(rating, opponent):
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400.0))


def solve(points, score):
    """R with sum E(R, p_i) = score; half-point convention at the ends."""
    n = len(points)
    if n == 0:
        return None, "none"
    kind = "estimate"
    if score >= n:
        score, kind = n - 0.5, "atLeast"
    elif score <= 0:
        score, kind = 0.5, "atMost"
    p = score / n
    offset = 400 * math.log10(p / (1 - p))
    lo, hi = min(points) + offset, max(points) + offset
    if hi == lo:
        return lo, kind
    while hi - lo >= 0.01:
        mid = (lo + hi) / 2
        if sum(expected(mid, x) for x in points) < score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, kind


def format_estimate(value, kind):
    if kind == "none":
        return "-"
    text = str(int(round(value)))
    return {"estimate": text, "atLeast": ">=" + text, "atMost": "<=" + text}[kind]


def period_starts(now, zone, sunday_weeks):
    local = now.astimezone(zone)
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    weekday = local.weekday()  # Monday = 0
    back = (weekday + 1) % 7 if sunday_weeks else weekday
    week = day - dt.timedelta(days=back)
    month = day.replace(day=1)
    year = day.replace(month=1, day=1)
    return {
        "lastHour": now - dt.timedelta(hours=1),
        "today": day,
        "thisWeek": week,
        "thisMonth": month,
        "thisYear": year,
        "allTime": None,
    }


def in_period(record, start):
    return start is None or parse_date(record["createdAt"]) >= start


def brier_at(records, move_number):
    total, games = 0.0, 0
    for record in records:
        score = record["outcome"].get("ourScore")
        if score is None:
            continue
        if record["setup"]["variant"] != "standard":
            continue
        color = record["ourColor"]
        ply = 2 * (move_number - 1) + (0 if color == "white" else 1)
        move = next((m for m in record["moves"] if m["ply"] == ply and m["ours"]), None)
        if move is None or move.get("decision") is None:
            continue
        d = move["decision"]
        o = (1.0 if score == 1 else 0.0, 1.0 if score == 0.5 else 0.0, 1.0 if score == 0 else 0.0)
        total += (d["win"] - o[0]) ** 2 + (d["draw"] - o[1]) ** 2 + (d["loss"] - o[2]) ** 2
        games += 1
    return (total / games if games else None), games


def summarize(records):
    scored = [r for r in records if r["outcome"].get("ourScore") is not None]
    wins = sum(1 for r in scored if r["outcome"]["ourScore"] == 1)
    draws = sum(1 for r in scored if r["outcome"]["ourScore"] == 0.5)
    losses = sum(1 for r in scored if r["outcome"]["ourScore"] == 0)
    score = (wins + 0.5 * draws) / len(scored) if scored else None
    rated_opponents = [r for r in scored if r["opponent"].get("ratingBefore") is not None]
    perf, kind = solve([r["opponent"]["ratingBefore"] for r in rated_opponents],
                       sum(r["outcome"]["ourScore"] for r in rated_opponents))
    rated = [r for r in scored if r["setup"]["rated"]]
    with_diff = [r for r in rated if r["us"].get("ratingDiff") is not None]
    if not rated:
        rating = "-"
    elif not with_diff:
        rating = "-*"
    else:
        total = sum(r["us"]["ratingDiff"] for r in with_diff)
        rating = ("+%d" % total if total > 0 else str(total)) + ("*" if len(with_diff) < len(rated) else "")
    brier, brier_games = brier_at(records, 20)
    return {
        "games": len(scored),
        "wdl": "%d-%d-%d" % (wins, draws, losses),
        "score": "-" if score is None else "%.1f%%" % (100 * score),
        "perf": format_estimate(perf, kind),
        "rating": rating,
        "brier20": "-" if brier is None else "%.3f" % brier,
        "brier20_games": brier_games,
        "not_counted": len(records) - len(scored),
        "rated_without_diff": len(rated) - len(with_diff),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", default=os.path.expanduser("~/Library/Application Support/DrewsChessMachine/LichessBot/Games"))
    parser.add_argument("--now", help="ISO 8601 time to compute at (default: now)")
    parser.add_argument("--tz", help="IANA time zone (default: the system's)")
    parser.add_argument("--sunday-weeks", action="store_true", help="weeks start on Sunday")
    parser.add_argument("--filter", choices=("all", "rated", "casual"), default="all")
    args = parser.parse_args()

    zone = ZoneInfo(args.tz or system_zone_name())
    now = parse_date(args.now) if args.now else dt.datetime.now(dt.timezone.utc)
    records = load_records(args.games)
    if args.filter == "rated":
        records = [r for r in records if r["setup"]["rated"]]
    elif args.filter == "casual":
        records = [r for r in records if not r["setup"]["rated"]]
    starts = period_starts(now, zone, args.sunday_weeks)
    print("records: %d (filter %s, now %s, zone %s)" % (len(records), args.filter, now.isoformat(), zone))
    for period, start in starts.items():
        subset = [r for r in records if in_period(r, start)]
        s = summarize(subset)
        print("%-9s games=%d W-D-L=%s score=%s perf=%s rating=%s brier@20=%s (n=%d) not-counted=%d rated-without-diff=%d"
              % (period, s["games"], s["wdl"], s["score"], s["perf"], s["rating"], s["brier20"], s["brier20_games"],
                 s["not_counted"], s["rated_without_diff"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
