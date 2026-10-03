#!/usr/bin/env python3
"""Charts for the no-SE ReZero comparison: no SE with ReZero (seeds 1 and 2), no SE
without ReZero (seeds 1 and 2) and zero-init ReZero: pElo and NLL by step for every
run, and each arm minus "no SE + ReZero s1".

Reads the same columns as `table.py` (its `table_arms`), so chart and table always show
the same runs. Writes `charts/rezero-*.svg` in this folder.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import charts_common as cc  # noqa: E402
from table import table_arms  # noqa: E402

BASELINE = "no SE + ReZero s1"
STYLE = {  # label → (color, dash)
    BASELINE: (cc.PALETTE[0], ""),
    "no SE + ReZero s2": (cc.PALETTE[0], cc.SEED2_DASH),
    "no SE, no ReZero s1": (cc.PALETTE[3], ""),
    "no SE, no ReZero s2": (cc.PALETTE[3], cc.SEED2_DASH),
    "zero-init ReZero": (cc.PALETTE[5], ""),
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
        return [cc.series(f"{label} − ReZero s1", cc.difference_points(points, by_label[BASELINE], index),
                          *STYLE[label]) for label, points in arms if label != BASELINE]

    written = [
        cc.write_chart(HERE, "rezero-pelo.svg", cc.render(
            "pElo by step (solid = seed 1, dashed = seed 2; higher is better)",
            [dict(h=260, ylabel="pElo", series=lines(0))])),
        cc.write_chart(HERE, "rezero-nll.svg", cc.render(
            "NLL by step (lower is better)",
            [dict(h=260, ylabel="NLL", series=lines(1))])),
        cc.write_chart(HERE, "rezero-vs-baseline.svg", cc.render(
            "Each run minus no SE + ReZero (seed 1) at the same step",
            [dict(h=200, ylabel="pElo difference (positive = run ahead)", series=gaps(0), zero=True),
             dict(h=200, ylabel="NLL difference (negative = run ahead)", series=gaps(1), zero=True)])),
    ]
    print("\n".join(written))


if __name__ == "__main__":
    main()
