#!/usr/bin/env python3
"""Position-by-position comparison of two checkpoints on the same probe battery.

Usage: compare.py <A positions.jsonl> <B positions.jsonl> [--set wide] [--label-a A] [--label-b B]

Inputs are `--probe-positions-out` files (one JSON line per position; `.gz` is
read transparently). Positions
are paired by `index` within the set, and the pairing is checked by name.

Reports, B relative to A:
  - mean bookmove NLL, the paired difference, a 95% bootstrap interval
    (10,000 resamples of positions, fixed seed) and the paired t statistic;
  - top-1 accuracy and McNemar's exact test on the discordant positions;
  - calibration: positions bucketed by top-1 probability, with accuracy per
    bucket and the expected calibration error (ECE);
  - confidence: mean legal-masked entropy, the share of positions with top-1
    probability above 0.9, and confident errors (top-1 wrong at p > 0.8);
  - all of the above split by legal-move count.

The probe battery is Lichess puzzles: each position has a single correct move,
so "several good moves" cannot be identified here; over-confidence shows as
confident errors and as calibration above the diagonal.
"""
import argparse
import gzip
import json
import math

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261002
CALIBRATION_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
LEGAL_BUCKETS = [(1, 10), (11, 25), (26, 40), (41, 1000)]


def load(path, set_label):
    rows = {}
    model_ids = set()
    opener = gzip.open if path.endswith(".gz") else open
    for line in opener(path, "rt"):
        record = json.loads(line)
        if record["set"] != set_label:
            continue
        model_ids.add(record["modelID"])
        if record["index"] in rows:
            raise SystemExit(f"{path}: index {record['index']} appears twice in set {set_label}")
        rows[record["index"]] = record
    if len(model_ids) != 1:
        raise SystemExit(f"{path}: expected one model, found {sorted(model_ids)}")
    return rows, model_ids.pop()


def binomial_two_sided(k, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def calibration(conf, correct):
    rows, ece = [], 0.0
    for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, min(hi, 1.0), 0, None, None))
            continue
        mean_conf = float(conf[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += n / len(conf) * abs(accuracy - mean_conf)
        rows.append((lo, min(hi, 1.0), n, mean_conf, accuracy))
    return rows, ece


def summarize(label, nll, conf, correct, entropy):
    wrong = ~correct
    return {
        "label": label,
        "nll": float(nll.mean()),
        "top1": float(correct.mean()),
        "entropy": float(entropy.mean()),
        "share_top1_over_0.9": float((conf > 0.9).mean()),
        "confident_errors": int((wrong & (conf > 0.8)).sum()),
        "mean_top1_prob_when_wrong": float(conf[wrong].mean()) if wrong.any() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--set", default="wide")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    rows_a, id_a = load(args.a, args.set)
    rows_b, id_b = load(args.b, args.set)
    if sorted(rows_a) != sorted(rows_b):
        raise SystemExit("the two files do not cover the same positions")
    indices = sorted(rows_a)
    for i in indices:
        if rows_a[i]["name"] != rows_b[i]["name"]:
            raise SystemExit(f"index {i}: names differ ({rows_a[i]['name']!r} vs {rows_b[i]['name']!r})")

    def required(rows, key, label):
        """Every position's value for `key`; a position missing it (an errored probe, or an
        expected move the probe found illegal) stops the comparison instead of becoming a
        made-up zero or NaN."""
        missing = [i for i in indices if key not in rows[i]]
        if missing:
            raise SystemExit(f"{label}: {len(missing)} position(s) lack {key!r} "
                             f"(first indices {missing[:10]}); cannot compare them")
        return [rows[i][key] for i in indices]

    def column(rows, key, label):
        return np.array(required(rows, key, label), dtype=float)

    nll_a, nll_b = column(rows_a, "nll", args.label_a), column(rows_b, "nll", args.label_b)
    conf_a, conf_b = column(rows_a, "top1Prob", args.label_a), column(rows_b, "top1Prob", args.label_b)
    correct_a = np.array([rank == 1 for rank in required(rows_a, "expectedRank", args.label_a)])
    correct_b = np.array([rank == 1 for rank in required(rows_b, "expectedRank", args.label_b)])
    entropy_a, entropy_b = column(rows_a, "entropyNats", args.label_a), column(rows_b, "entropyNats", args.label_b)
    legal = np.array([rows_a[i]["legalCount"] for i in indices])

    n = len(indices)
    diff = nll_b - nll_a
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAP_RESAMPLES)])
    t_stat = diff.mean() / (diff.std(ddof=1) / math.sqrt(n))
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((correct_b & ~correct_a).sum())

    print(f"# {args.label_b} vs {args.label_a} — set {args.set}, {n} positions, paired by index")
    print(f"models: {args.label_a} = {id_a}, {args.label_b} = {id_b}\n")
    print("## Paired tests\n")
    print("| measure | " + args.label_a + " | " + args.label_b + " | B − A | 95% interval / test |")
    print("|---|---:|---:|---:|---|")
    print(f"| mean NLL | {nll_a.mean():.4f} | {nll_b.mean():.4f} | {diff.mean():+.4f} | "
          f"bootstrap [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}], paired t = {t_stat:.2f} |")
    print(f"| top-1 correct | {int(correct_a.sum())} | {int(correct_b.sum())} | {int(correct_b.sum()) - int(correct_a.sum()):+d} | "
          f"McNemar: only {args.label_a} {only_a}, only {args.label_b} {only_b}, exact p = {binomial_two_sided(min(only_a, only_b), only_a + only_b):.4f} |")
    print(f"| positions where B's NLL is lower | | | {int((diff < 0).sum())} of {n} | |\n")

    print("## Confidence\n")
    print("| | NLL | top-1 | mean entropy (nats) | top-1 p > 0.9 | confident errors (wrong, p > 0.8) | mean top-1 p when wrong |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s in (summarize(args.label_a, nll_a, conf_a, correct_a, entropy_a),
              summarize(args.label_b, nll_b, conf_b, correct_b, entropy_b)):
        print(f"| {s['label']} | {s['nll']:.4f} | {s['top1']:.4f} | {s['entropy']:.3f} | "
              f"{s['share_top1_over_0.9']:.4f} | {s['confident_errors']} | {s['mean_top1_prob_when_wrong']:.4f} |")

    print("\n## Calibration (bucketed by top-1 probability)\n")
    cal_a, ece_a = calibration(conf_a, correct_a)
    cal_b, ece_b = calibration(conf_b, correct_b)
    print(f"| top-1 p | {args.label_a} n | {args.label_a} mean p | {args.label_a} accuracy | "
          f"{args.label_b} n | {args.label_b} mean p | {args.label_b} accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    for (lo, hi, na, ca, aa), (_, _, nb, cb, ab) in zip(cal_a, cal_b):
        print(f"| {lo:.1f}–{hi:.1f} | {na} | {fmt(ca)} | {fmt(aa)} | {nb} | {fmt(cb)} | {fmt(ab)} |")
    print(f"\nECE: {args.label_a} {ece_a:.4f}, {args.label_b} {ece_b:.4f}\n")

    print("## By legal-move count\n")
    print("| legal moves | n | NLL A | NLL B | B − A | top-1 A | top-1 B | entropy A | entropy B |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lo, hi in LEGAL_BUCKETS:
        mask = (legal >= lo) & (legal <= hi)
        if not mask.any():
            continue
        print(f"| {lo}–{hi if hi < 1000 else ''} | {int(mask.sum())} | {nll_a[mask].mean():.4f} | {nll_b[mask].mean():.4f} | "
              f"{(nll_b[mask] - nll_a[mask]).mean():+.4f} | {correct_a[mask].mean():.4f} | {correct_b[mask].mean():.4f} | "
              f"{entropy_a[mask].mean():.3f} | {entropy_b[mask].mean():.3f} |")


if __name__ == "__main__":
    main()
