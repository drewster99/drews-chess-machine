#!/usr/bin/env python3
"""Render the per-run tensor tables of TENSOR-STATS.md from data/tensor_stats.csv.

Everything between the two marker lines in TENSOR-STATS.md is replaced; the
hand-written findings above the first marker are left untouched. Run
tensor_stats.py first.

Usage: python3 experiments/20260929-se-style-ab/tensor_report.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "tensor_stats.csv")
MD_PATH = os.path.join(HERE, "TENSOR-STATS.md")
BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"

RUN_ORDER = ["se_sb", "se_att", "se_none", "se_zb1", "se_sb2", "se_att2", "se_none2", "se_zb2"]


def fmt(x):
    v = float(x)
    if v == 0:
        return "0"
    return f"{v:+.4g}"


def label(row):
    return row["tensor"] if row["part"] == "all" else f"{row['tensor']} [{row['part']}]"


def main():
    with open(CSV_PATH) as fh:
        rows = list(csv.DictReader(fh))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    out = [BEGIN, ""]
    out.append("## Checkpoints covered")
    out.append("")
    out.append("| run | arm | seed | checkpoints | steps | trained ModelID |")
    out.append("|---|---|---:|---:|---|---|")
    for run in RUN_ORDER:
        rs = by_run[run]
        steps = sorted({int(r["training_step"]) for r in rs})
        ids = sorted({r["model_id"] for r in rs if int(r["training_step"]) > 0})
        out.append(f"| `{run}` | {rs[0]['arm']} | {rs[0]['seed']} | {len(steps)} | "
                   f"0, {steps[1]:,} … {steps[-1]:,} | `{', '.join(ids)}` |")
    out.append("")

    out.append("## Every tensor, fresh vs final, per run")
    out.append("")
    out.append("Network parameters and BN running statistics. `[gamma]` / `[beta]` rows split a "
               "scale+bias SE fc2 tensor into its γ half (rows 0–127) and β half (rows 128–255). "
               "All checkpoints, including the intermediate ones, are in "
               "[data/tensor_stats.csv](data/tensor_stats.csv).")
    for run in RUN_ORDER:
        rs = by_run[run]
        final_step = max(int(r["training_step"]) for r in rs)
        fresh = {label(r): r for r in rs if int(r["training_step"]) == 0}
        final = {label(r): r for r in rs
                 if int(r["training_step"]) == final_step and r["kind"] != "optimizer"}
        out.append("")
        out.append(f"### `{run}` — {rs[0]['arm']}, seed {rs[0]['seed']} (step 0 → {final_step:,})")
        out.append("")
        out.append("| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | "
                   "final max | final std | final abs max |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, f in final.items():
            a = fresh[name]
            out.append(f"| `{name}` | {f['shape']} | {fmt(a['mean'])} | {fmt(a['min'])} | "
                       f"{fmt(a['max'])} | {fmt(f['mean'])} | {fmt(f['min'])} | {fmt(f['max'])} | "
                       f"{fmt(f['std'])} | {fmt(f['abs_max'])} |")

    out.append("")
    out.append("## Optimizer velocity at the final checkpoint")
    out.append("")
    out.append("Only checkpoints from build 2259 onward carry optimizer state (seed 2 and zero-β). "
               "`zero frac` is the fraction of elements whose momentum velocity is exactly 0.")
    for run in RUN_ORDER:
        rs = by_run[run]
        final_step = max(int(r["training_step"]) for r in rs)
        opt = [r for r in rs if int(r["training_step"]) == final_step and r["kind"] == "optimizer"]
        if not opt:
            continue
        out.append("")
        out.append(f"### `{run}` (step {final_step:,})")
        out.append("")
        out.append("| tensor | count | mean | min | max | rms | zero frac |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in opt:
            out.append(f"| `{label(r)}` | {int(r['count']):,} | {fmt(r['mean'])} | {fmt(r['min'])} | "
                       f"{fmt(r['max'])} | {fmt(r['rms'])} | {float(r['zero_frac']):.4f} |")
    out.append("")
    out.append(END)

    with open(MD_PATH) as fh:
        text = fh.read()
    if BEGIN not in text or END not in text:
        raise ValueError(f"{MD_PATH} is missing the generated-table markers")
    head = text[:text.index(BEGIN)]
    tail = text[text.index(END) + len(END):]
    with open(MD_PATH, "w") as fh:
        fh.write(head + "\n".join(out) + tail)
    print(f"wrote tables to {MD_PATH}")


if __name__ == "__main__":
    main()
