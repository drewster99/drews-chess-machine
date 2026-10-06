#!/usr/bin/env python3
"""Compare the tensors of two safetensors files, ignoring their metadata.

Usage: safetensors_tensor_compare.py <reference> <other> [--max-relative R]

Both files must hold the same tensor names, and each tensor the same dtype and
shape; any difference there fails the comparison.

- R = 0 (the default) compares raw tensor bytes: bit identity, so +0 and -0
  differ, and so do two NaNs with different payloads.
- R > 0 decodes F32 / F16 / BF16 tensors and computes, per tensor,
  max|a - b| / max(max|a|, FLT_MIN) with `a` the reference -- the rule and
  denominator floor ResumeEquivalenceTests applies -- so an all-zero reference
  tensor compares against FLT_MIN and never divides by zero. A NaN or +-Inf in
  either tensor fails unless the two tensors' bytes are identical. A tensor of
  any other dtype must be byte-identical.

Prints the worst ratio and its tensor. Exit status: 0 when every tensor passes,
1 when any fails or the names / dtypes / shapes differ, 2 when a file cannot be
read as safetensors.

This is the "no change to training math" comparator of the hyperparameter
recording plan: run the same seeded command twice on one build to measure the
machine's run-to-run floor F, then compare a run of the new build against it
with --max-relative F.
"""
import argparse
import json
import math
import struct
import sys

FLT_MIN = 1.1754943508222875e-38

FLOAT_FORMATS = {"F32": ("f", 4), "F16": ("e", 2), "BF16": (None, 2)}

ELEMENT_SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                 "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}


class UnreadableFile(Exception):
    pass


def read_tensors(path):
    """Map tensor name -> (dtype, shape, raw bytes)."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise UnreadableFile(f"{path}: shorter than the 8-byte header length")
            header_length = struct.unpack("<Q", prefix)[0]
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise UnreadableFile(f"{path}: header is truncated")
            header = json.loads(header_bytes)
            if not isinstance(header, dict):
                raise UnreadableFile(f"{path}: header is not a JSON object")
            data = handle.read()
    except OSError as error:
        raise UnreadableFile(f"{path}: {error}") from error
    except (ValueError, UnicodeDecodeError) as error:
        raise UnreadableFile(f"{path}: header is not valid JSON: {error}") from error
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = entry["data_offsets"]
            dtype = entry["dtype"]
            shape = entry["shape"]
        except (KeyError, TypeError, ValueError) as error:
            raise UnreadableFile(f"{path}: tensor {name} has a malformed entry: {error}") from error
        if not (0 <= start <= end <= len(data)):
            raise UnreadableFile(f"{path}: tensor {name} lies outside the data section")
        if dtype in ELEMENT_SIZES and math.prod(shape) * ELEMENT_SIZES[dtype] != end - start:
            raise UnreadableFile(f"{path}: tensor {name} holds {end - start} bytes, not {dtype} {shape}")
        tensors[name] = (dtype, list(shape), data[start:end])
    return tensors


def decode_floats(dtype, raw):
    code, width = FLOAT_FORMATS[dtype]
    count = len(raw) // width
    if code is not None:
        return struct.unpack(f"<{count}{code}", raw)
    # BF16 is the high half of an F32.
    halves = struct.unpack(f"<{count}H", raw)
    return struct.unpack(f"<{count}f", b"".join(struct.pack("<I", half << 16) for half in halves))


def tensor_ratio(dtype, reference_raw, other_raw):
    """The relative difference of one tensor, or math.inf when it cannot pass."""
    if reference_raw == other_raw:
        return 0.0
    if dtype not in FLOAT_FORMATS:
        return math.inf
    reference = decode_floats(dtype, reference_raw)
    other = decode_floats(dtype, other_raw)
    if not all(math.isfinite(value) for value in reference + other):
        return math.inf
    scale = max(max((abs(value) for value in reference), default=0.0), FLT_MIN)
    worst = max((abs(a - b) for a, b in zip(reference, other)), default=0.0)
    return worst / scale


def compare(reference_path, other_path, max_relative):
    """Return (passed, messages)."""
    reference = read_tensors(reference_path)
    other = read_tensors(other_path)
    messages = []
    structural = False
    for name in sorted(set(reference) | set(other)):
        if name not in other:
            messages.append(f"missing from {other_path}: {name}")
            structural = True
        elif name not in reference:
            messages.append(f"missing from {reference_path}: {name}")
            structural = True
        elif reference[name][0] != other[name][0]:
            messages.append(f"dtype differs for {name}: {reference[name][0]} vs {other[name][0]}")
            structural = True
        elif reference[name][1] != other[name][1]:
            messages.append(f"shape differs for {name}: {reference[name][1]} vs {other[name][1]}")
            structural = True
        elif len(reference[name][2]) != len(other[name][2]):
            messages.append(f"byte length differs for {name}")
            structural = True
    if structural:
        return False, messages
    worst_ratio = 0.0
    worst_name = None
    for name in sorted(reference):
        dtype, _, reference_raw = reference[name]
        other_raw = other[name][2]
        if max_relative == 0:
            ratio = 0.0 if reference_raw == other_raw else math.inf
        else:
            ratio = tensor_ratio(dtype, reference_raw, other_raw)
        if worst_name is None or ratio > worst_ratio:
            worst_ratio = ratio
            worst_name = name
    passed = worst_ratio <= max_relative
    if worst_name is None:
        messages.append("no tensors")
    else:
        messages.append(f"worst ratio {worst_ratio!r} at {worst_name} (max relative {max_relative!r})")
    return passed, messages


def main(arguments):
    parser = argparse.ArgumentParser(description="Compare two safetensors files' tensors.")
    parser.add_argument("reference")
    parser.add_argument("other")
    parser.add_argument("--max-relative", type=float, default=0.0)
    options = parser.parse_args(arguments)
    if not (options.max_relative >= 0 and math.isfinite(options.max_relative)):
        parser.error("--max-relative must be a finite number >= 0")
    try:
        passed, messages = compare(options.reference, options.other, options.max_relative)
    except UnreadableFile as error:
        print(f"unreadable: {error}", file=sys.stderr)
        return 2
    for message in messages:
        print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
