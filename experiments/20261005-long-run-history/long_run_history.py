#!/usr/bin/env python3
"""What the long corpus-replay runs (v5, qeu8, nt8y, coxw) actually trained with, how their heads
were built and behaved, and nt8y against R7/R8 at matched wall time (from step times measured on this Mac).

Sources: documentation/dashboards/registry.json (segments, architecture strings) and data/<run>.csv
(probe pElo / NLL and training-line means), the session logs named by the registry under
~/Library/Logs/DrewsChessMachine (a log that is missing is reported, never guessed), and the
R7/R8 probe table (experiments/20261004-fatconv-1x15x15-98/table.py).

Usage: python3 experiments/20261005-long-run-history/long_run_history.py > experiments/20261005-long-run-history/results.md
"""
import csv
import json
import os
import re
import statistics as st
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DASH = os.path.join(REPO, "documentation", "dashboards")
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
RUNS = ["v5", "qeu8", "nt8y", "coxw"]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def registry():
    r = json.load(open(os.path.join(DASH, "registry.json")))
    return r.get("runs", r)


def csv_rows(run):
    return list(csv.DictReader(open(os.path.join(DASH, "data", f"{run}.csv"))))


def section_recipes(reg):
    print("## 1. Training settings per segment ([REPLAY-HPARAMS], first line of each surviving log)\n")
    for run in RUNS:
        print(f"### {run}")
        for seg in reg[run]["segments"]:
            path = os.path.join(LOGS, seg["log"])
            base = seg.get("cumstep_base")
            if not os.path.exists(path):
                print(f"- cum {base:,}: {seg['log']} — log missing; settings unknown")
                continue
            hp = cycle = None
            lrs = {}
            for line in open(path, errors="replace"):
                if hp is None and "[REPLAY-HPARAMS]" in line:
                    hp = line.split("[REPLAY-HPARAMS]", 1)[1].strip()
                if cycle is None and "[REPLAY-CYCLE]" in line:
                    cycle = line.split("[REPLAY-CYCLE]", 1)[1].strip()
                m = re.search(r"\[REPLAY\] step=\d+ .* lr=([0-9.e-]+)", line)
                if m:
                    lrs[m.group(1)] = lrs.get(m.group(1), 0) + 1
            top = ", ".join(f"lr={k} ×{v}" for k, v in sorted(lrs.items(), key=lambda kv: -kv[1])[:3])
            print(f"- cum {base:,}: {seg['log']}\n  - HPARAMS: {hp}\n  - CYCLE: {cycle or 'none logged'}\n  - logged step LRs: {top}")
        print()


def section_heads(reg):
    print("## 2. Head configurations (registry arch_heads)\n")
    for run in RUNS + ["qeu8b1128", "se_none"]:
        print(f"- {run}: {reg[run].get('arch_heads')}")
    print()


def section_head_health():
    print("## 3. Head behaviour by step window (means of the dashboard CSV rows in the window)\n")
    cols = ["pElo", "nll", "vLoss", "pLoss", "pIllM", "gNorm", "pLogit_mean", "pLogit_peak"]
    print("| run | window | rows | " + " | ".join(cols) + " |")
    print("|---|---|---:|" + "---:|" * len(cols))
    for run in ["nt8y", "qeu8", "v5"]:
        rows = csv_rows(run)
        for lo, hi in [(100000, 150000), (200000, 250000), (250000, 312748), (400000, 450000), (600000, 650000), (800000, 860000)]:
            w = [r for r in rows if lo <= float(r["cum_step"]) <= hi]
            if not w:
                continue
            cells = []
            for c in cols:
                v = [num(r.get(c)) for r in w]
                v = [x for x in v if x is not None]
                cells.append(f"{st.mean(v):.4g}" if v else "")
            print(f"| {run} | {lo // 1000}–{hi // 1000}k | {len(w)} | " + " | ".join(cells) + " |")
    print()


# Measured on this Mac (M5 Max), alone on the GPU, build 2320, fp32 policy tail, steps 200-550 of
# 600: R7 33k 816.1 / 808.1 ms/step (fp32 A / B), nt8y's last checkpoint 443.1 ms/step
# (experiments/20261004-policy-tail-precision/timing/, summary E-0011).
R7_MS_PER_STEP = (816.1 + 808.1) / 2
NT8Y_MS_PER_STEP = 443.1


def section_equal_time():
    ratio = R7_MS_PER_STEP / NT8Y_MS_PER_STEP
    print(f"## 4. nt8y at matched wall time: nt8y step = R7/R8 step × {ratio:.3f} (measured step times; window means of probes)\n")
    nt = {int(float(r["cum_step"])): float(r["pElo"]) for r in csv_rows("nt8y") if num(r.get("pElo")) is not None}
    table = subprocess.run(["python3", os.path.join(REPO, "experiments", "20261004-fatconv-1x15x15-98", "table.py")],
                           capture_output=True, text=True, check=True).stdout
    avg = {}
    for line in table.splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if c and c[0].replace(",", "").isdigit() and c[4]:
            avg[int(c[0].replace(",", ""))] = float(c[4])

    def window(d, s, w):
        v = [p for k, p in d.items() if abs(k - s) <= w]
        return (st.mean(v), len(v)) if v else None

    f = lambda x: f"{x[0]:.1f} (n={x[1]})" if x else "no probes"
    print("| R7/R8 step | R7/R8 avg (±w, n) | nt8y step | nt8y (±2k, n) | nt8y − R7/R8 |")
    print("|---:|---:|---:|---:|---:|")
    for s in [2000, 5000, 10000, 15000, 20000, 25000, 30000, 33000]:
        w = 1000 if s <= 5000 else 2000
        t = round(s * ratio)
        a, b = window(avg, s, w), window(nt, t, 2000)
        d = f"{b[0] - a[0]:+.1f}" if a and b else ""
        print(f"| {s:,} | {f(a)} | {t:,} | {f(b)} | {d} |")
    print()


if __name__ == "__main__":
    reg = registry()
    section_recipes(reg)
    section_heads(reg)
    section_head_health()
    section_equal_time()
