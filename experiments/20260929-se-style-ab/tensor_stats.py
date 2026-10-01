#!/usr/bin/env python3
"""Per-tensor statistics for every checkpoint of the SE style experiment.

Reads every safetensors file belonging to the eight runs — the fresh nets in
`models/` plus every enumerated checkpoint (`-replay-step<N>`) in the DCM
Models folder — and writes one row per (checkpoint, tensor) to
`data/tensor_stats.csv`.

Checkpoints are identified by their `__metadata__` (`model_id`,
`training_step`), never by filename. The `-replay-latest` files are skipped:
each is a byte-for-byte copy of that run's final enumerated stop save.

For a scale+bias SE `fc2` weight/bias (2C outputs: rows 0..C-1 produce γ,
rows C..2C-1 produce β) two extra rows are written with `part` = `gamma` /
`beta`, so the halves can be read separately. Every other row has
`part` = `all`.

Usage: python3 experiments/20260929-se-style-ab/tensor_stats.py
"""

import csv
import json
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
OUT = os.path.join(HERE, "data", "tensor_stats.csv")

# run key -> (arm, seed, filename stem shared by the fresh net and the enumerated checkpoints)
RUNS = {
    "se_sb": ("scale+bias", 1, "20260929-test_SE_scale+bias"),
    "se_att": ("attenuate-only", 1, "20260929-test_SE_attenuate-only"),
    "se_none": ("none", 1, "20260929-test_SE_none"),
    "se_sb2": ("scale+bias", 2, "20260929-test_SE_scale+bias-seed2"),
    "se_att2": ("attenuate-only", 2, "20260929-test_SE_attenuate-only-seed2"),
    "se_none2": ("none", 2, "20260929-test_SE_none-seed2"),
    "se_zb1": ("zero-beta scale+bias", 1, "20260929-test_SE_zerobeta-seed1"),
    "se_zb2": ("zero-beta scale+bias", 2, "20260929-test_SE_zerobeta-seed2"),
}

DTYPES = {"F32": np.float32, "F16": np.float16}

COLUMNS = [
    "run", "arm", "seed", "file", "model_id", "training_step", "tensor", "part", "kind",
    "group", "shape", "count", "mean", "std", "min", "max", "abs_max", "mean_abs", "rms",
    "l2_norm", "zero_frac", "nonfinite",
]


def read_safetensors(path):
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        body = fh.read()
    metadata = header.pop("__metadata__", {})
    tensors = {}
    for name, info in header.items():
        dtype = info["dtype"]
        if dtype == "BF16":
            start, end = info["data_offsets"]
            raw = np.frombuffer(body[start:end], dtype=np.uint16).astype(np.uint32) << 16
            array = raw.view(np.float32)
        elif dtype in DTYPES:
            start, end = info["data_offsets"]
            array = np.frombuffer(body[start:end], dtype=DTYPES[dtype]).astype(np.float32)
        else:
            raise ValueError(f"{path}: tensor {name} has unsupported dtype {dtype}")
        tensors[name] = (info["shape"], array)
    return metadata, tensors


def classify(name):
    """(kind, group): kind = weight / bn_running_stat / optimizer; group = network section."""
    kind = "optimizer" if name.startswith("opt.") else (
        "bn_running_stat" if name.endswith(("running_mean", "running_var")) else "parameter")
    bare = name[4:] if name.startswith("opt.") else name
    if bare.startswith("blocks."):
        group = "block" + bare.split(".")[1]
    else:
        group = bare.split(".")[0]
    return kind, group


def stats(values):
    finite = np.isfinite(values)
    nonfinite = int(values.size - finite.sum())
    v = values[finite].astype(np.float64)
    if v.size == 0:
        raise ValueError("tensor has no finite values")
    return {
        "count": int(values.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "abs_max": float(np.abs(v).max()),
        "mean_abs": float(np.abs(v).mean()),
        "rms": float(np.sqrt(np.mean(v * v))),
        "l2_norm": float(np.sqrt(np.sum(v * v))),
        "zero_frac": float(np.mean(v == 0.0)),
        "nonfinite": nonfinite,
    }


def checkpoint_files(stem):
    fresh = os.path.join(HERE, "models", f"{stem}-fresh.safetensors")
    if not os.path.exists(fresh):
        raise FileNotFoundError(fresh)
    files = [fresh]
    pattern = re.compile(re.escape(stem) + r"-replay-step(\d+)\.safetensors$")
    enumerated = sorted(
        (int(m.group(1)), f) for f in os.listdir(MODELS_DIR) if (m := pattern.match(f)))
    if not enumerated:
        raise FileNotFoundError(f"no enumerated checkpoints for {stem} in {MODELS_DIR}")
    files += [os.path.join(MODELS_DIR, f) for _, f in enumerated]
    return files


def main():
    rows = []
    for run, (arm, seed, stem) in RUNS.items():
        model_ids = set()
        for path in checkpoint_files(stem):
            metadata, tensors = read_safetensors(path)
            step = int(metadata.get("training_step", "0"))
            model_id = metadata["model_id"]
            if step > 0:
                model_ids.add(model_id)
            for name, (shape, values) in tensors.items():
                kind, group = classify(name)
                parts = [("all", values)]
                if ".se_scalebias.fc2." in name:
                    half = values.size // 2
                    parts += [("gamma", values[:half]), ("beta", values[half:])]
                for part, v in parts:
                    rows.append({
                        "run": run, "arm": arm, "seed": seed, "file": os.path.basename(path),
                        "model_id": model_id, "training_step": step, "tensor": name,
                        "part": part, "kind": kind, "group": group,
                        "shape": "x".join(str(d) for d in shape), **stats(v),
                    })
        if len(model_ids) != 1:
            raise ValueError(f"{run}: trained checkpoints span ModelIDs {sorted(model_ids)}")
        print(f"{run}: {len({r['file'] for r in rows if r['run'] == run})} checkpoints", file=sys.stderr)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{r[k]:.9g}" if isinstance(r[k], float) else r[k]) for k in COLUMNS})
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
