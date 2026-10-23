#!/usr/bin/env python3
"""Markdown tables from the --coreml-benchmark reports in a results folder
(positions/hour, millions, median of the rounds; batch-1 latency). Prints to stdout.

  python3 summarize.py [results folder, default results/]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("20260630-mini-YkKk-resume2-replay-step250803", "mini YkKk (0.24M, 2 x 64 3x3 SE, trained)"),
    ("20261008-zlra-ab-fresh", "zlra shape (8.27M, 3 x 128 9x9, untrained)"),
    ("bench-v4_5block_7x7-untrained", "v4_5block_7x7 preset (8.45M, 5 x 128 7x7 SE, untrained)"),
    ("bench-10x256-untrained", "10 x 256 3x3 SE (12.9M, untrained)"),
]


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    for stem, title in MODELS:
        report = json.load(open(os.path.join(folder, f"{stem}.json")))
        arms = [a["label"] for a in report["arms"]]
        cells = {(c["arm"], c["batch_size"], c["in_flight"]): c for c in report["cells"]}
        print(f"### {title}\n")
        print("| Arm | Planned placement | Load s | Top-1 agreement | Batch-1 median ms |")
        print("|---|---|---|---|---|")
        for arm in report["arms"]:
            placement = arm.get("placement")
            if placement:
                total = sum(placement["estimated_cost"].values()) or 1
                where = ", ".join(f"{d} {v / total:.0%}" for d, v in
                                  sorted(placement["estimated_cost"].items(), key=lambda kv: -kv[1]))
            else:
                where = "MPSGraph (GPU)"
            parity = arm.get("parity")
            agreement = (f"{min(p['top_move_agreements'] / (p['positions'] - p['positions_without_legal_moves']) for p in parity):.2%}"
                         if parity else "reference")
            latency = cells[(arm["label"], 1, 1)]["latency_median_ms"]
            print(f"| {arm['label']} | {where} | {arm['load_seconds']:.2f} | {agreement} | {latency:.3f} |")
        print()
        for in_flight in report["in_flight"]:
            print(f"Positions/hour (millions), in flight {in_flight}; spread > 1.25 marked *\n")
            print("| Batch | " + " | ".join(arms) + " |")
            print("|---" * (len(arms) + 1) + "|")
            for batch in report["batch_sizes"]:
                row = []
                for arm in arms:
                    cell = cells[(arm, batch, in_flight)]
                    mark = "*" if cell["round_spread"] > 1.25 else ""
                    row.append(f"{cell['positions_per_second_median'] * 3600 / 1e6:.1f}{mark}")
                # The winner by the measured rate, not the rounded text: two arms can round alike.
                best = max(range(len(arms)), key=lambda i: cells[(arms[i], batch, in_flight)]["positions_per_second_median"])
                row[best] = f"**{row[best]}**"
                print(f"| {batch} | " + " | ".join(row) + " |")
            print()


if __name__ == "__main__":
    main()
