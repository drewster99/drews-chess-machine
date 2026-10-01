#!/usr/bin/env python3
"""Scan every .safetensors file under the DCM application-support folder
(Models/, Sessions/*.dcmsession/, KeptSelfPlayModels/, ...) and record its
header metadata (model_id, parent_model_id, training_step, architecture,
creator, ...) plus the dtype/shape of every policy-head tensor. Legacy
.dcmmodel files are listed with their size so the report can account for them.

Reads only the 8-byte length prefix and the JSON header of each safetensors
file; tensor payloads are never touched, so a full scan is cheap.

Usage:
    python3 scan_headers.py [DCM_ROOT] > inventory.jsonl
"""
import json
import os
import struct
import sys

DEFAULT_DCM_ROOT = os.path.expanduser(
    "~/Library/Application Support/DrewsChessMachine")


def read_header(path):
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    metadata = header.pop("__metadata__", {})
    return metadata, header


def main():
    dcm_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DCM_ROOT
    for directory, subdirectories, names in os.walk(dcm_root):
        subdirectories.sort()
        for name in sorted(names):
            path = os.path.join(directory, name)
            relative_path = os.path.relpath(path, dcm_root)
            if name.endswith(".dcmmodel"):
                print(json.dumps({"path": relative_path, "kind": "dcmmodel",
                                  "bytes": os.path.getsize(path)}))
                continue
            if not name.endswith(".safetensors"):
                continue
            record = {"path": relative_path, "kind": "safetensors",
                      "bytes": os.path.getsize(path),
                      "mtime": os.path.getmtime(path)}
            try:
                metadata, tensors = read_header(path)
            except Exception as error:  # report, never silently drop
                record["error"] = f"{type(error).__name__}: {error}"
                print(json.dumps(record))
                continue
            record["meta"] = metadata
            record["policy_tensors"] = {
                key: {"dtype": value["dtype"], "shape": value["shape"]}
                for key, value in tensors.items() if "policy" in key}
            print(json.dumps(record))


if __name__ == "__main__":
    main()
