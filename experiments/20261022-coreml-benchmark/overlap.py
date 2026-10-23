#!/usr/bin/env python3
"""Overlap tables for OVERLAP.md: in flight 4 against in flight 1 for the
MPSGraph arms of a --coreml-benchmark results folder, then the comparisons at
equal positions in flight. Markdown to stdout. Throughput is millions of
positions/hour (median round); * marks a round spread above 1.25.

  python3 overlap.py [results folder, default results-run2/]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("20260630-mini-YkKk-resume2-replay-step250803", "mini YkKk (0.24M)"),
    ("20261008-zlra-ab-fresh", "zlra shape (8.27M)"),
    ("bench-v4_5block_7x7-untrained", "v4_5block_7x7 preset (8.45M)"),
    ("bench-10x256-untrained", "10 x 256 (12.9M)"),
]
REFERENCE = "mps-bfloat16"


def mph(cell):
    return cell["positions_per_second_median"] * 3600 / 1e6


def shown(cell):
    return f"{mph(cell):.1f}{'*' if cell['round_spread'] > 1.25 else ''}"


def load(folder):
    for stem, title in MODELS:
        path = os.path.join(folder, f"{stem}.json")
        if os.path.exists(path):
            report = json.load(open(path))
            cells = {(c["arm"], c["batch_size"], c["in_flight"]): c for c in report["cells"]}
            yield title, report, cells


def per_batch(folder):
    for title, report, cells in load(folder):
        others = [a["label"] for a in report["arms"] if a["engine"] == "mpsgraph" and a["label"] != REFERENCE]
        coreml = [a["label"] for a in report["arms"] if a["engine"] == "coreml"]
        print(f"#### {title}\n")
        print(f"| Batch | {REFERENCE} in flight 1 | in flight 4 | gain | ms/call in flight 1 | ms/call in flight 4 | "
              + " | ".join(f"{a} gain" for a in others) + " | best Core ML in flight 4 ÷ bf16 in flight 4 |")
        print("|---" * (7 + len(others)) + "|")
        for batch in report["batch_sizes"]:
            one, four = cells[(REFERENCE, batch, 1)], cells[(REFERENCE, batch, 4)]
            row = [str(batch), shown(one), shown(four), f"{mph(four) / mph(one) - 1:+.0%}",
                   f"{one['latency_median_ms']:.2f}", f"{four['latency_median_ms']:.2f}"]
            row += [f"{mph(cells[(a, batch, 4)]) / mph(cells[(a, batch, 1)]) - 1:+.0%}" for a in others]
            best = max(coreml, key=lambda a: mph(cells[(a, batch, 4)]))
            row.append(f"{mph(cells[(best, batch, 4)]) / mph(four):.2f} ({best})")
            print("| " + " | ".join(row) + " |")
        print()


def equal_in_flight(folder, pairs):
    print(f"| Model | " + " | ".join(f"4 × {b4} ÷ 1 × {b1}" for b4, b1 in pairs) + " |")
    print("|---" * (len(pairs) + 1) + "|")
    for title, report, cells in load(folder):
        row = [title]
        for b4, b1 in pairs:
            four, one = cells[(REFERENCE, b4, 4)], cells[(REFERENCE, b1, 1)]
            row.append(f"{shown(four)} ÷ {shown(one)} = {mph(four) / mph(one):.2f}")
        print("| " + " | ".join(row) + " |")
    print()


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results-run2")
    per_batch(folder)
    batches = json.load(open(os.path.join(folder, f"{MODELS[1][0]}.json")))["batch_sizes"]
    pairs = [(64, 250), (128, 512), (250, 1024)] if 250 in batches else [(64, 256), (128, 512), (256, 1024)]
    print(f"Equal positions in flight, {REFERENCE}:\n")
    equal_in_flight(folder, pairs)


if __name__ == "__main__":
    main()
