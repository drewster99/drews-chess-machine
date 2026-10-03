#!/usr/bin/env python3
"""No-ReZero vs ReZero review at one training step (no SE in any arm).

Usage: review.py <step>

For each run that reached <step>:
  - mean training metrics over the [REPLAY] lines in the 500 steps up to <step>;
  - per-block conv2 weight norm and the branch scale the block output sees
    (effective ReZero alpha x ||conv2||; alpha is 1 without ReZero), from the
    enumerated checkpoint at <step>, identified by its safetensors metadata;
  - pElo tallies, no-ReZero seed 1 vs ReZero seed 1, over every 1k step both reached.
"""
import json
import math
import os
import re
import struct
import sys

import numpy as np

import table

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import dcm_arch  # noqa: E402  each checkpoint's ReZero cap, read from its own metadata

MODELS = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
# (label, session log, out-model stem, trained model_id): a checkpoint is used only
# when its header carries this model_id at the requested training_step.
RUNS = [
    ("ReZero s1", "dcm_log_20260929-150743.txt", "20260929-test_SE_none", "20260929-24-834D"),
    ("no ReZero s1", "dcm_log_20261002-011124.txt", "20261002-bench_v5s3_noSE_noReZero", "20261002-2-5tKN"),
    ("no ReZero s2", "dcm_log_20261002-035513.txt", "20261002-bench_v5s3_noSE_noReZero-seed2", "20261002-4-T79u"),
    ("ReZero s2", "dcm_log_20260930-104117.txt", "20260929-test_SE_none-seed2", "20260930-6-LkS6"),
]
METRICS = ["loss", "pLoss", "vLoss", "pEnt", "gNorm"]


def log_means(log_name, step):
    pattern = re.compile(r"\[REPLAY\] step=(\d+) (.*)")
    sums = {m: [] for m in METRICS}
    for line in open(os.path.join(table.LOGS, log_name), errors="replace"):
        match = pattern.search(line)
        if not match or not step - 500 < int(match.group(1)) <= step:
            continue
        for metric in METRICS:
            value = re.search(rf"\b{metric}=([-0-9.]+)", match.group(2))
            if value:
                sums[metric].append(float(value.group(1)))
    if not sums["loss"]:
        return None
    return {m: sum(v) / len(v) for m, v in sums.items() if v}, len(sums["loss"])


def read_tensors(path, wanted):
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        metadata = header.pop("__metadata__", {})
        out = {}
        for name in wanted:
            if name not in header:
                continue
            spec = header[name]
            start, end = spec["data_offsets"]
            handle.seek(8 + header_len + start)
            raw = handle.read(end - start)
            if spec["dtype"] == "F32":
                values = np.frombuffer(raw, dtype=np.float32)
            elif spec["dtype"] == "BF16":
                values = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                raise ValueError(f"{name}: unsupported dtype {spec['dtype']}")
            out[name] = values.astype(np.float64)
        return metadata, out


def branch_scale(stem, step, expected_model_id):
    path = os.path.join(MODELS, f"{stem}-replay-step{step}.safetensors")
    if not os.path.exists(path):
        return None
    blocks = dcm_arch.rezero_blocks_of_file(path)
    wanted = [f"blocks.{b.index}.{t}" for b in blocks for t in ("conv2.weight", "rezero_alpha")]
    metadata, tensors = read_tensors(path, wanted)
    for key in ("model_id", "training_step"):
        if key not in metadata:
            raise SystemExit(f"{path}: header has no {key}")
    if metadata["model_id"] != expected_model_id:
        raise SystemExit(f"{path}: header model_id {metadata['model_id']} != expected {expected_model_id}")
    if int(metadata["training_step"]) != step:
        raise SystemExit(f"{path}: header training_step {metadata['training_step']} != {step}")
    rows = []
    for block in blocks:
        norm = float(np.linalg.norm(tensors[f"blocks.{block.index}.conv2.weight"]))
        alpha = tensors.get(f"blocks.{block.index}.rezero_alpha")
        if (alpha is not None) != block.use_rezero:
            raise SystemExit(f"{path}: block {block.index} use_rezero={block.use_rezero} "
                             f"disagrees with the tensors present")
        effective = 1.0 if alpha is None else block.effective(float(alpha[0]))
        rows.append((norm, effective, effective * norm))
    return metadata.get("model_id"), rows


def tally(step):
    a = table.csv_points("se_none")
    label, file_name, model_id = table.PROBE_ARMS[0]
    b = table.probe_points(file_name, model_id, label)
    steps = [s for s in range(1000, step + 1, 1000) if s in a and s in b]
    non_finite = [s for s in steps if b[s][0] is None]
    if non_finite:
        raise SystemExit(f"{label}: non-finite pElo at step(s) {non_finite}; no sign test over them")
    diffs = [b[s][0] - a[s][0] for s in steps]
    ahead = sum(d > 0 for d in diffs)
    behind = sum(d < 0 for d in diffs)
    n = ahead + behind
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(ahead, behind) + 1)) / 2 ** n) if n else float("nan")
    return len(diffs), ahead, behind, p, sum(diffs) / len(diffs)


def main():
    step = int(sys.argv[1])
    print(f"=== review at step {step} ===")
    for label, log_name, stem, model_id in RUNS:
        means = log_means(log_name, step)
        scale = branch_scale(stem, step, model_id)
        if means is None and scale is None:
            continue
        print(f"\n[{label}]")
        if means:
            m, count = means
            print("  log means over " + str(count) + " [REPLAY] lines: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()))
        if scale:
            model_id, rows = scale
            print(f"  {model_id}: ||conv2|| / eff alpha / branch scale per block: " +
                  " | ".join(f"{n:.2f} / {e:.3f} / {s:.2f}" for n, e, s in rows))
    count, ahead, behind, p, mean = tally(step)
    print(f"\nno ReZero s1 vs ReZero s1, pElo 1k-{step // 1000}k: {count} checkpoints, ahead {ahead}, "
          f"behind {behind}, tied {count - ahead - behind}, sign test p {p:.3f}, mean difference {mean:+.1f}")


if __name__ == "__main__":
    main()
