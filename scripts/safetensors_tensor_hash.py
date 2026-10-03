#!/usr/bin/env python3
"""SHA-256 of a safetensors file's tensor data, ignoring its metadata.

Prints two lines: the hash over every tensor except BN running statistics
("trainables"), and the hash over the running statistics alone. Tensors are
hashed in name order, each as its name, dtype, shape and raw bytes, so the
result depends only on the tensors themselves — not on the ModelID, the
timestamp or anything else in `__metadata__`.
"""
import hashlib
import json
import struct
import sys


def tensor_hashes(path):
    with open(path, 'rb') as handle:
        header_length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_length))
        data_start = 8 + header_length
        trainables = hashlib.sha256()
        running = hashlib.sha256()
        for name in sorted(key for key in header if key != '__metadata__'):
            entry = header[name]
            start, end = entry['data_offsets']
            handle.seek(data_start + start)
            raw = handle.read(end - start)
            if len(raw) != end - start:
                raise SystemExit(f'{path}: tensor {name} is truncated')
            target = running if name.endswith('.running_mean') or name.endswith('.running_var') else trainables
            target.update(name.encode())
            target.update(entry['dtype'].encode())
            target.update(json.dumps(entry['shape']).encode())
            target.update(raw)
    return trainables.hexdigest(), running.hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: safetensors_tensor_hash.py <file.safetensors>')
    trainables, running = tensor_hashes(sys.argv[1])
    print(f'trainables={trainables}')
    print(f'bn_running_stats={running}')
