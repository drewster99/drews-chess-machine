#!/usr/bin/env python3
"""Charts for the label smoothing arms (C policy ε 0.03, seeds 1 and 2; D value ε 0)
against the baseline (ReLU scale+bias, policy ε 0.1 / value ε 0.013, seeds 1 and 2):
pElo and NLL by step for every run, and each arm minus the seed-1 baseline.

Reads the same columns as `table.py` (its `table_arms`), so chart and table always show
the same runs. Writes `charts/label-smoothing-*.svg` in this folder.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import charts_common as cc  # noqa: E402
from table import table_arms  # noqa: E402

BASELINE = "baseline ε 0.1 / 0.013"
STYLE = {  # label → (color, dash)
    BASELINE: (cc.PALETTE[0], ""),
    "baseline seed 2": (cc.PALETTE[0], cc.SEED2_DASH),
    "C policy ε 0.03": (cc.PALETTE[1], ""),
    "C seed 2": (cc.PALETTE[1], cc.SEED2_DASH),
    "D value ε 0": (cc.PALETTE[2], ""),
}


def main():
    arms, _, _ = table_arms()
    by_label = dict(arms)
    for label in by_label:
        if label not in STYLE:
            raise SystemExit(f"{label}: no chart style; add it to STYLE")

    def lines(index):
        return [cc.series(label, cc.metric_points(points, index), *STYLE[label]) for label, points in arms]

    def gaps(index):
        return [cc.series(f"{label} − baseline", cc.difference_points(points, by_label[BASELINE], index),
                          *STYLE[label]) for label, points in arms if not label.startswith("baseline")]

    written = [
        cc.write_chart(HERE, "label-smoothing-pelo.svg", cc.render(
            "pElo by step (solid = seed 1, dashed = seed 2; higher is better)",
            [dict(h=260, ylabel="pElo", series=lines(0))])),
        cc.write_chart(HERE, "label-smoothing-nll.svg", cc.render(
            "NLL by step (lower is better)",
            [dict(h=260, ylabel="NLL", series=lines(1))])),
        cc.write_chart(HERE, "label-smoothing-vs-baseline.svg", cc.render(
            "Each arm minus the seed-1 baseline at the same step",
            [dict(h=200, ylabel="pElo difference (positive = arm ahead)", series=gaps(0), zero=True),
             dict(h=200, ylabel="NLL difference (negative = arm ahead)", series=gaps(1), zero=True)])),
    ]
    print("\n".join(written))


if __name__ == "__main__":
    main()
