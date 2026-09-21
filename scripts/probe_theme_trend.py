#!/usr/bin/env python3
"""Per-theme + top1/top5 trend from the in-training wide puzzle probe.

The app logs `[TACTICAL-LICHESS] tick set=wide` every ~25 steps with aggregate
accuracy (argmax, top5, avgProb, avgRank, NLL, pElo) AND a per-theme n/total
breakdown. Themes are not epistemically equal: mateIn1/mateIn2 have a unique
machine-verifiable answer, while opening/middlegame encode human preference.
So a DIFFERENTIAL decline (convention themes fall, forced themes hold) reads as
stylistic drift, while a UNIFORM decline including forced themes reads as decay.

Bins the run into equal-step buckets and reports the mean of each series per
bucket, so single-probe noise (~+-7 pElo tick to tick) averages out.

Usage: python3 scripts/probe_theme_trend.py <run> [--bins 10]
       run = a key in documentation/dashboards/selfplay_registry.json
"""
import os, re, sys, json, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "..", "documentation", "dashboards", "selfplay_registry.json")
LOGDIR = os.path.expanduser("~/Library/Logs/DrewsChessMachine")

HEAD = re.compile(r"\[TACTICAL-LICHESS\] tick set=wide step=(\d+) argmax=(\d+)/(\d+)\S* "
                  r"top5=(\d+)/(\d+)\S* avgProb=([\d.]+) avgRank=([\d.]+) NLL=([\d.]+) "
                  r"pElo=(-?\d+) model=(\S+)")
THEME = re.compile(r"lichess([A-Za-z0-9]+)=(\d+)/(\d+)")
# Forced/objective themes: a unique machine-verifiable answer exists.
FORCED = {"MateIn1", "MateIn2", "HangingPiece", "Promotion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--bins", type=int, default=10)
    a = ap.parse_args()
    cfg = json.load(open(REG))["runs"][a.run]
    base = cfg.get("base_modelID")

    marks = []   # (cum_step, dict)
    step_base = 0; rmax = 0
    for log in cfg["logs"]:
        p = os.path.join(LOGDIR, log)
        if not os.path.exists(p): continue
        decided = False
        for line in open(p, errors="replace"):
            if "set=wide" not in line: continue
            m = HEAD.search(line)
            if not m or (base and not m.group(10).startswith(base)): continue
            raw = int(m.group(1))
            if not decided:
                if rmax > 500 and raw <= 500: step_base = rmax
                decided = True
            cum = raw + step_base; rmax = max(rmax, cum)
            d = {"argmax": int(m.group(2)) / int(m.group(3)),
                 "top5": int(m.group(4)) / int(m.group(5)),
                 "avgProb": float(m.group(6)), "avgRank": float(m.group(7)),
                 "NLL": float(m.group(8)), "pElo": float(m.group(9))}
            for t, n, tot in THEME.findall(line):
                if int(tot): d["T:" + t] = int(n) / int(tot)
            marks.append((cum, d))
    if not marks: sys.exit("no wide probe marks found")
    marks.sort(key=lambda x: x[0])
    lo, hi = marks[0][0], marks[-1][0]
    width = max(1, (hi - lo) // a.bins)

    bins = defaultdict(lambda: defaultdict(list))
    for cum, d in marks:
        b = min(a.bins - 1, (cum - lo) // width)
        for k, v in d.items(): bins[b][k].append(v)

    def col(b, k):
        v = bins[b].get(k)
        return sum(v) / len(v) if v else None

    keys = sorted({k for _, d in marks for k in d if k.startswith("T:")})
    idx = sorted(bins)
    centers = [lo + int((b + 0.5) * width) for b in idx]

    print(f"run {a.run}: {len(marks)} wide marks, cum steps {lo}..{hi}, {a.bins} bins of ~{width} steps\n")
    hdr = f"{'series':<22}" + "".join(f"{c//1000:>8}k" for c in centers) + f"{'Δ first→last':>14}"
    print(hdr); print("-" * len(hdr))

    def row(label, k, scale=1.0, fmt="{:>9.3f}"):
        vals = [col(b, k) for b in idx]
        first = next((v for v in vals if v is not None), None)
        last = next((v for v in reversed(vals) if v is not None), None)
        cells = "".join(("{:>9}".format("—") if v is None else fmt.format(v * scale)) for v in vals)
        delta = "—" if (first is None or last is None) else f"{(last-first)*scale:+.3f}"
        print(f"{label:<22}{cells}{delta:>14}")

    for k, lab in [("argmax", "argmax %"), ("top5", "top5 %")]:
        row(lab, k, 100.0, "{:>9.2f}")
    for k in ["avgProb", "avgRank", "NLL"]:
        row(k, k)
    row("pElo", "pElo", 1.0, "{:>9.1f}")
    print()
    print("FORCED themes (unique verifiable answer):")
    for k in keys:
        if k[2:] in FORCED: row("  " + k[2:], k, 100.0, "{:>9.2f}")
    print("CONVENTION / pattern themes:")
    for k in keys:
        if k[2:] not in FORCED: row("  " + k[2:], k, 100.0, "{:>9.2f}")


if __name__ == "__main__":
    main()
