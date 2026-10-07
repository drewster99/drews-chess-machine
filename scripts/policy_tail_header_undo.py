#!/usr/bin/env python3
"""Undo the PT-D3 header edit (`policy_tail_header_edit.py`) from its backups.

Each backup holds a file's original safetensors header and the file's
original whole-file SHA-256. A file is restored only when it is exactly what
the edit made of it: its current header must equal the original header with
the edit's one inserted key (`edited_header`, the edit script's own function,
so the two scripts cannot disagree about what the edit was). The restore
writes the original header over the same data region, and the result must
hash to the original whole-file SHA-256 — byte for byte the file before the
edit — or nothing replaces the edited file.

The same safety as the edit: hidden `.<name>.<uuid>.tmp` staging beside the
file, F_FULLFSYNC, the file's permissions and access/modification times
kept, the identity (device, inode, size, mtime) re-checked just before the
`rename`, a regular file only. The backups are left in place.

Dry run by default; `--apply` restores. `--only` limits it to the given
file paths. `--path-map OLD=NEW` (repeatable) restores a backup's file at
another location — how the undo is tested on copies.
"""
import argparse
import base64
import fcntl
import glob
import hashlib
import json
import os
import stat
import struct
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_tail_header_edit import CHUNK, F_FULLFSYNC, KEY, Skip, edited_header, read_header, sha256_file  # noqa: E402


def restore(path, original_header, original_sha, apply):
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise Skip("not a regular file")
    current_sha = sha256_file(path)
    if current_sha == original_sha:
        return "already original"
    length, header = read_header(path)
    metadata = json.loads(header).get("__metadata__") or {}
    tail = metadata.get(KEY)
    if tail is None:
        raise Skip(f"has no {KEY}, and is not the original either")
    if header != edited_header(original_header, tail):
        raise Skip("current header is not the original plus the edit's key")
    if not apply:
        return "would restore"

    directory, name = os.path.split(path)
    staging = os.path.join(directory, f".{name}.{uuid.uuid4()}.tmp")
    try:
        with open(path, "rb") as source, open(staging, "xb") as target:
            source.seek(8 + length)
            target.write(struct.pack("<Q", len(original_header)))
            target.write(original_header)
            while True:
                block = source.read(CHUNK)
                if not block:
                    break
                target.write(block)
            target.flush()
            fcntl.fcntl(target.fileno(), F_FULLFSYNC)
        if sha256_file(staging) != original_sha:
            raise Skip("restored bytes do not hash to the original whole-file SHA-256")
        os.chmod(staging, stat.S_IMODE(info.st_mode))
        os.utime(staging, ns=(info.st_atime_ns, info.st_mtime_ns))
        now = os.lstat(path)
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
            raise Skip("file changed while being restored")
        os.rename(staging, path)
    except BaseException:
        if os.path.lexists(staging):
            os.remove(staging)
        raise
    directory_handle = os.open(directory, os.O_RDONLY)
    try:
        fcntl.fcntl(directory_handle, F_FULLFSYNC)
    finally:
        os.close(directory_handle)
    if sha256_file(path) != original_sha:
        raise Skip("file does not hash to the original after the rename")
    return "restored"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--only", nargs="*", help="restore only these file paths (as recorded in the backups)")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="restore the backup of OLD at NEW instead")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    path_map = {}
    for entry in args.path_map:
        old, separator, new = entry.partition("=")
        if not separator:
            sys.exit(f"--path-map needs OLD=NEW, got {entry!r}")
        path_map[old] = new
    only = set(args.only) if args.only else None

    counts = {}
    for backup_file in sorted(glob.glob(os.path.join(args.backup_dir, "*.header.json"))):
        backup = json.load(open(backup_file))
        recorded = backup["path"]
        if only is not None and recorded not in only:
            continue
        target = path_map.get(recorded, recorded)
        try:
            outcome = restore(target, base64.b64decode(backup["header_base64"]), backup["old_sha256"], args.apply)
        except Skip as reason:
            outcome = f"skipped: {reason}"
        counts[outcome.split(":")[0]] = counts.get(outcome.split(":")[0], 0) + 1
        print(f"{outcome:<40} {target}", flush=True)
    print(json.dumps(counts, indent=1))
    if any(key.startswith("skipped") for key in counts):
        sys.exit(1)


if __name__ == "__main__":
    main()
