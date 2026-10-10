#!/usr/bin/env python3
"""Summarize a DrewsChessMachine crash dump (a `*.dcmcrash` folder).

A dump is written when training stops on a GPU fault or a non-finite loss,
and on a near miss (a gradient norm ≥ 1,000× its reference), under
`~/Library/Application Support/DrewsChessMachine/CrashDumps/`.

Usage:
    dcm_crash_dump.py <dump folder>

Prints: why and when, the step and its batch, the GPU faults the run saw, the
non-finite tensors in the weights after the failure (when they were read),
the recent gradient norms, and what else is in the folder.
"""
from __future__ import annotations

import json
import os
import sys


def read_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not os.path.isdir(argv[1]):
        print("usage: dcm_crash_dump.py <dump folder>")
        return 2
    folder = argv[1]
    manifest = read_json(os.path.join(folder, "manifest.json"))
    print(f"reason:        {manifest['reason']}")
    print(f"detail:        {manifest['detail']}")
    print(f"written:       {manifest['written_at']}  build {manifest['build']}")
    print(f"path / model:  {manifest['path_kind']} / {manifest['model_id']}")
    print(f"trainer step:  {manifest['trainer_step']} (batch is step {manifest['batch_trainer_step']}'s)")
    print(f"batch hash:    {manifest.get('batch_hash') or 'none (no batch captured)'}")
    print(f"LR / momentum: {manifest['learning_rate']} / {manifest['momentum']}")
    print(f"fault monitor: {manifest['gpu_fault_monitor']}")
    faults = manifest.get("gpu_faults", [])
    print(f"GPU faults:    {len(faults)}")
    for fault in faults:
        print(f"  #{fault['sequence']} {fault['time']} {fault['source']}: {fault['detail'][:160]}")
    hashes = manifest.get("recent_batch_hashes", [])
    if hashes:
        last = hashes[-1]
        print(f"batch hashes:  {len(hashes)} kept, last step {last['trainer_step']} "
              f"chain {'partial' if last['chain'] is None else last['chain'][:16]}")
    norms = manifest.get("gradient_norms")
    if norms and norms["pre_clip_norms"]:
        tail = norms["pre_clip_norms"][-10:]
        print(f"pre-clip norms (last {len(tail)} through step {norms['last_trainer_step']}): "
              + " ".join(f"{value:.4g}" for value in tail))
    for note in manifest.get("notes", []):
        print(f"note:          {note}")
    census_path = os.path.join(folder, "weights-after-census.json")
    error_path = os.path.join(folder, "weights-after-error.txt")
    if os.path.exists(census_path):
        census = read_json(census_path)
        bad = [entry for entry in census if entry["non_finite"] > 0]
        print(f"weights after: {len(census)} tensors, {len(bad)} with non-finite values")
        for entry in bad[:20]:
            print(f"  {entry['name']}: {entry['non_finite']} of {entry['count']} non-finite")
    elif os.path.exists(error_path):
        with open(error_path) as handle:
            print(f"weights after: not read — {handle.read().strip()}")
    else:
        print(f"weights after: {manifest['weights_after']}")
    others = manifest.get("other_processes", [])
    print(f"other DrewsChessMachine processes: {len(others)}")
    for line in others:
        print(f"  {line[:200]}")
    print("files:         " + ", ".join(sorted(os.listdir(folder))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
