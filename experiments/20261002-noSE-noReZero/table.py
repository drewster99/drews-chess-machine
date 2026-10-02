#!/usr/bin/env python3
"""Per-1000-step table: no-SE with ReZero (SE experiment seeds 1 and 2) vs no-SE
without ReZero (this run). pElo / NLL from `--probe-set wide` probes; blank where
a run never reached the step."""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "documentation", "dashboards", "data")


def csv_points(run):
    points = {}
    with open(os.path.join(DATA, f"{run}.csv")) as handle:
        for row in csv.DictReader(handle):
            if row.get("pElo"):
                points[int(float(row["cum_step"]))] = (float(row["pElo"]), float(row["nll"]))
    return points


def probe_points():
    points = {}
    path = os.path.join(HERE, "probes.jsonl")
    if os.path.exists(path):
        for line in open(path):
            record = json.loads(line)
            points[record["step"]] = (record["pElo"], record["nll"])
    return points


def main():
    arms = [("no SE + ReZero s1", csv_points("se_none")), ("no SE, no ReZero", probe_points()),
            ("no SE + ReZero s2", csv_points("se_none2"))]
    last = max((max(p) for _, p in arms if p), default=0)
    header = ["step"] + [f"pElo {l}" for l, _ in arms] + [f"NLL {l}" for l, _ in arms]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for step in range(1000, last + 1, 1000):
        row = [f"{step:,}"]
        row += [f"{p[step][0]:.1f}" if step in p else "" for _, p in arms]
        row += [f"{p[step][1]:.4f}" if step in p else "" for _, p in arms]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
