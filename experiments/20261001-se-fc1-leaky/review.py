#!/usr/bin/env python3
"""Leaky-FC1 vs ReLU review at one training step.

Usage: review.py <step> <DrewsChessMachine binary>

For the ReLU baseline (SE experiment scale+bias seed 1) and the leaky-FC1 arm at
the same step:
  - pElo / NLL from `--probe-model <checkpoint> --probe-set wide` (same binary for
    both arms), plus the baseline's recorded values from the dashboard CSV;
  - mean training metrics over the 500 steps up to <step> from each run's log;
  - SE FC1 units per block that kept their initial direction (cosine with the
    fresh net's FC1 row > 0.999995 — TENSOR-STATS.md's "dead almost the whole
    run" test), and the distribution of per-unit cosine with init.
Both arms start from bit-identical FC1 weights (the leaky net is a derived copy).
"""
import csv
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

MODELS = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
LOGS = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

ARMS = {
    "relu": {
        "fresh": "20260929-test_SE_scale+bias-fresh.safetensors",
        "stem": "20260929-test_SE_scale+bias-replay-step",
        "log": "dcm_log_20260929-150727.txt",
    },
    "leaky_fc1": {
        "fresh": "20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors",
        "stem": "20261001-test_SE_scale+bias-fc1leaky-replay-step",
        "log": "dcm_log_20261001-151822.txt",
    },
}
UNMOVED_COSINE = 0.999995
# A unit whose FC1 weight-velocity column norm is below this fraction of its
# block's 90th percentile is "effectively off": under ReLU an always-off unit's
# velocity decays to exactly 0; under leaky ReLU it keeps about the negative
# slope's share (1%) of a live unit's gradient. The reference is the 90th
# percentile, not the median: more than half of each block's units are weak, so
# the median is itself a weak unit and a median reference counts none (the
# correction recorded in README.md, 2026-10-02).
OFF_VELOCITY_FRACTION = 0.05
OFF_VELOCITY_REFERENCE_PERCENTILE = 90
# Velocity-based comparator: ReLU scale+bias seed 2 (same architecture and
# parameters, different random init; its build saves optimizer velocity,
# seed 1's does not). Reached 7,282 steps.
SEED2_RELU_STEM = "20260929-test_SE_scale+bias-seed2-replay-step"


def read_tensors(path, wanted):
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        base = 8 + header_len
        out = {}
        for name in wanted:
            spec = header[name]
            start, end = spec["data_offsets"]
            handle.seek(base + start)
            raw = handle.read(end - start)
            if spec["dtype"] == "F32":
                values = np.frombuffer(raw, dtype=np.float32)
            elif spec["dtype"] == "BF16":
                values = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                raise ValueError(f"{name}: unsupported dtype {spec['dtype']}")
            out[name] = values.reshape(spec["shape"]).astype(np.float64)
        return out


def fc1_names(block):
    return f"blocks.{block}.se_scalebias.fc1.weight"


def velocity_off_counts(path):
    """Per block: (units with exactly-zero FC1 weight velocity, units below
    OFF_VELOCITY_FRACTION of the block's 90th percentile), or None if the file has no
    optimizer velocity."""
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    names = [f"opt.blocks.{b}.se_scalebias.fc1.weight.velocity" for b in range(3)]
    if not all(name in header for name in names):
        return None
    tensors = read_tensors(path, names)
    counts = []
    for name in names:
        # Velocity is stored flat in the graph's [in, out] layout: one column
        # per FC1 unit (checked against TENSOR-STATS.md's dead-unit counts).
        velocity = tensors[name].reshape(-1, 32)
        row_norm = np.linalg.norm(velocity, axis=0)
        reference = np.percentile(row_norm, OFF_VELOCITY_REFERENCE_PERCENTILE)
        counts.append((int((row_norm == 0).sum()), int((row_norm < OFF_VELOCITY_FRACTION * reference).sum())))
    return counts


def probe(binary, path):
    result = subprocess.run([binary, "--probe-model", path, "--probe-set", "wide"],
                            capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            record = json.loads(line)
            if "pElo" in record or "pelo" in record:
                return record
    raise RuntimeError(f"no probe JSON line for {path}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def log_window(log_name, step, width=500):
    pattern = re.compile(r"\[REPLAY\] step=(\d+) loss=([\d.-]+) pLoss=([\d.-]+) vLoss=([\d.-]+) pEnt=([\d.-]+)")
    rows = []
    with open(os.path.join(LOGS, log_name), errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match and step - width < int(match.group(1)) <= step:
                rows.append([float(x) for x in match.groups()[1:]])
    if not rows:
        return None
    return np.mean(np.array(rows), axis=0), len(rows)


def baseline_csv(step):
    with open(os.path.join(REPO, "documentation", "dashboards", "data", "se_sb.csv")) as handle:
        for row in csv.DictReader(handle):
            if int(float(row["cum_step"])) == step:
                return row
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    step = int(sys.argv[1])
    binary = sys.argv[2]
    print(f"=== review at step {step} ===")
    for arm, spec in ARMS.items():
        checkpoint = os.path.join(MODELS, f"{spec['stem']}{step}.safetensors")
        if not os.path.exists(checkpoint):
            print(f"{arm}: no checkpoint {checkpoint}")
            continue
        names = [fc1_names(b) for b in range(3)]
        fresh = read_tensors(os.path.join(MODELS, spec["fresh"]), names)
        now = read_tensors(checkpoint, names)
        unmoved, cos_summary = [], []
        for name in names:
            a, b = fresh[name], now[name]
            cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            unmoved.append(int((cos > UNMOVED_COSINE).sum()))
            cos_summary.append(f"min {cos.min():.4f} p25 {np.percentile(cos, 25):.4f} median {np.median(cos):.4f}")
        record = probe(binary, checkpoint)
        window = log_window(spec["log"], step)
        print(f"\n[{arm}] {os.path.basename(checkpoint)}")
        print(f"  probe (this binary): {json.dumps({k: record[k] for k in record if k in ('pElo', 'nll', 'modelID', 'n')})}")
        if window is not None:
            means, count = window
            print(f"  log mean over last {count} logged steps: loss {means[0]:.4f} pLoss {means[1]:.4f} vLoss {means[2]:.4f} pEnt {means[3]:.3f}")
        print(f"  FC1 units unmoved from init (cos > {UNMOVED_COSINE}) per block: {' / '.join(map(str, unmoved))} of 32")
        off = velocity_off_counts(checkpoint)
        if off is None:
            print("  FC1 velocity: not saved by this run's build")
        else:
            print(f"  FC1 units with zero velocity per block: {' / '.join(str(z) for z, _ in off)}; "
                  f"below {OFF_VELOCITY_FRACTION:.0%} of block p{OFF_VELOCITY_REFERENCE_PERCENTILE}: {' / '.join(str(o) for _, o in off)}")
        for block, summary in enumerate(cos_summary):
            print(f"    block {block} cosine with init: {summary}")
    seed2 = os.path.join(MODELS, f"{SEED2_RELU_STEM}{step}.safetensors")
    if os.path.exists(seed2):
        off = velocity_off_counts(seed2)
        if off is not None:
            print(f"\n[relu seed 2, velocity comparator] zero velocity per block: {' / '.join(str(z) for z, _ in off)}; "
                  f"below {OFF_VELOCITY_FRACTION:.0%} of block p{OFF_VELOCITY_REFERENCE_PERCENTILE}: {' / '.join(str(o) for _, o in off)}")
    row = baseline_csv(step)
    if row is not None:
        print(f"\nbaseline recorded in dashboards/data/se_sb.csv at {step}: pElo {row['pElo']} nll {row['nll']}")


if __name__ == "__main__":
    main()
