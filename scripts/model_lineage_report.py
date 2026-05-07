#!/usr/bin/env python3
"""model_lineage_report.py — scan all session logs for one model lineage.

Reports where a trainer/candidate/champion ID appears across log files so you
can tell whether a failure was created in the current session or inherited.

Usage:
  python3 scripts/model_lineage_report.py 20260506-5-aoTz-2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path


LOG_GLOB = os.path.expanduser("~/Library/Logs/DrewsChessMachine/dcm_log_*.txt")
NUM_RE = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def parse_stats_fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ["steps", "pEnt", "gNorm", "pLogitAbsMax", "legalMass", "pEntLegal"]:
        m = re.search(rf"{re.escape(key)}={NUM_RE}", line)
        if m:
            out[key] = m.group(1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id", help="trainer/candidate/champion/model id to scan for")
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(LOG_GLOB))
    if not paths:
        print("no logs found", file=sys.stderr)
        return 1

    hits = 0
    for path in paths:
        first_stats = None
        last_stats = None
        first_alarm = None
        arena_hits = []
        param_lines = []
        with path.open("r", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if args.model_id not in line:
                    continue
                if "[STATS] elapsed=" in line:
                    fields = parse_stats_fields(line)
                    stamp = line.split("  [", 1)[0]
                    record = (stamp, fields)
                    if first_stats is None:
                        first_stats = record
                    last_stats = record
                elif "[ALARM]" in line and first_alarm is None:
                    first_alarm = line
                elif "[ARENA]" in line and " kv " in line:
                    arena_hits.append(line)
                elif "[PARAM]" in line:
                    param_lines.append(line)

        if first_stats is None and not arena_hits and not first_alarm and not param_lines:
            continue

        hits += 1
        print(path.name)
        if first_stats:
            print(
                "  first stats:",
                first_stats[0],
                "step=" + first_stats[1].get("steps", "?"),
                "pEnt=" + first_stats[1].get("pEnt", "?"),
                "gNorm=" + first_stats[1].get("gNorm", "?"),
                "pLogitAbsMax=" + first_stats[1].get("pLogitAbsMax", "?"),
                "legalMass=" + first_stats[1].get("legalMass", "?"),
            )
        if last_stats and last_stats is not first_stats:
            print(
                "  last stats: ",
                last_stats[0],
                "step=" + last_stats[1].get("steps", "?"),
                "pEnt=" + last_stats[1].get("pEnt", "?"),
                "gNorm=" + last_stats[1].get("gNorm", "?"),
                "pLogitAbsMax=" + last_stats[1].get("pLogitAbsMax", "?"),
                "legalMass=" + last_stats[1].get("legalMass", "?"),
            )
        if arena_hits:
            print("  arena lines:")
            for line in arena_hits[-3:]:
                print("   ", line)
        if param_lines:
            print("  param lines:")
            for line in param_lines[-3:]:
                print("   ", line)
        if first_alarm:
            print("  first alarm:")
            print("   ", first_alarm)
        print()

    if hits == 0:
        print(f"no log references found for {args.model_id}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
