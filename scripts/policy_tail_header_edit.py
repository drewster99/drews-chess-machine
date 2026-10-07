#!/usr/bin/env python3
"""One-off PT-D3 header edit (owner decisions 2026-10-07, POLICY_TAIL_ARCHITECTURE_PLAN.md).

Adds the flat `trainer_policy_tail_precision` key to the safetensors header of
each file the audit (`policy_tail_audit.py`) resolved, and changes nothing
else: the key is inserted as text right after `"__metadata__":{`, so every
other header byte, the format version and the data region stay as they were,
and `content_sha256` (data region only) still holds.

Why a text insertion and not a re-serialization: the owner decided the edit
adds only the key. Re-serializing would re-escape and re-order the rest of the
header, which no reader cares about but which would make "changes nothing
else" untrue and the edit harder to audit.

Safety, per file (any failure skips that file and is reported; nothing is
half-written):
- the file is the one the audit saw: same whole-file SHA-256 as the audit
  recorded, a regular file (not a symbolic link);
- the new header parses, its tensor entries equal the old ones, and its
  metadata equals the old metadata plus exactly the one key;
- the new file is written beside the original under a hidden
  `.<name>.<uuid>.tmp` staging name (FileSafety's own pattern, so the app's
  orphan sweep recognizes it), its data region verified byte-identical by
  SHA-256, fsynced with F_FULLFSYNC, given the original's permissions and
  access/modification times;
- immediately before the swap the original's identity (device, inode, size,
  mtime) is re-checked against what was read, then `rename` replaces it;
- the original header bytes are saved first to a backup folder, so any file
  can be restored exactly (original header + the unchanged data region).

Dry run by default; `--apply` performs the edit. Writes a JSON manifest of
every file (old/new whole-file SHA-256, tail written, outcome).
"""
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
import stat
import struct
import sys
import uuid

KEY = "trainer_policy_tail_precision"
TAILS = {"fp32_from_pre_bn", "mixed_final_projection"}
F_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", 51)
CHUNK = 8 * 1024 * 1024


class Skip(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise Skip("truncated header length")
        (length,) = struct.unpack("<Q", prefix)
        header = handle.read(length)
        if len(header) != length:
            raise Skip("truncated header")
    return length, header


def lineage_tail(metadata):
    raw = metadata.get("dcm_lineage")
    if raw is None:
        return None
    try:
        record = json.loads(raw)
    except ValueError as error:
        raise Skip(f"dcm_lineage does not parse: {error}")
    configuration = record.get("configuration") or {}
    return configuration.get("policy_tail_precision")


def edited_header(header, tail):
    marker = b'"__metadata__":{'
    if header.count(marker) != 1:
        raise Skip("header has no single compact __metadata__ object to insert into")
    insertion = json.dumps(KEY).encode() + b":" + json.dumps(tail).encode()
    position = header.index(marker) + len(marker)
    rest_is_empty = header[position:position + 1] == b"}"
    return header[:position] + insertion + (b"" if rest_is_empty else b",") + header[position:]


def check_edit(old_header, new_header, tail):
    old = json.loads(old_header)
    new = json.loads(new_header)
    old_meta = old.pop("__metadata__", None)
    new_meta = new.pop("__metadata__", None)
    if old_meta is None or new_meta is None:
        raise Skip("no __metadata__")
    if old != new:
        raise Skip("tensor entries changed")
    expected = dict(old_meta)
    expected[KEY] = tail
    if new_meta != expected:
        raise Skip("metadata is not the old metadata plus the one key")


def backup_name(relative_path):
    """Session champions all share one file name, so the backup is named after
    the store-relative path."""
    return relative_path.replace("/", "__") + ".header.json"


def earlier_edit(path, relative_path, expected_sha, current_sha, tail, backup_dir):
    """A file an earlier, interrupted `--apply` already edited: its backup holds
    the audited header, and the file now is exactly that header plus the key
    over the same data region. Anything else is not ours to explain."""
    if backup_dir is None:
        return None
    for name in (backup_name(relative_path), os.path.basename(path) + ".header.json"):
        candidate = os.path.join(backup_dir, name)
        if not os.path.exists(candidate):
            continue
        backup = json.load(open(candidate))
        if backup.get("path") != path or backup.get("old_sha256") != expected_sha:
            continue
        original = base64.b64decode(backup["header_base64"])
        _, header = read_header(path)
        if header != edited_header(original, tail):
            raise Skip("edited earlier, but the header is not the original plus the key")
        return {"old_sha256": expected_sha, "new_sha256": current_sha, "edited_earlier": True}
    return None


def edit(path, relative_path, expected_sha, tail, backup_dir, apply):
    if tail not in TAILS:
        raise Skip(f"unexpected tail {tail!r}")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise Skip("not a regular file")
    old_sha = sha256_file(path)
    if old_sha != expected_sha:
        earlier = earlier_edit(path, relative_path, expected_sha, old_sha, tail, backup_dir)
        if earlier is not None:
            return earlier
        raise Skip("file changed since the audit (whole-file SHA-256 differs)")
    length, header = read_header(path)
    metadata = json.loads(header).get("__metadata__") or {}
    if KEY in metadata:
        raise Skip(f"already records {KEY}={metadata[KEY]!r}")
    recorded = lineage_tail(metadata)
    if recorded is not None and recorded != tail:
        raise Skip(f"lineage records {recorded!r}, audit proposes {tail!r}")
    new_header = edited_header(header, tail)
    check_edit(header, new_header, tail)
    if not apply:
        return {"old_sha256": old_sha, "new_sha256": None}

    with open(os.path.join(backup_dir, backup_name(relative_path)), "x") as backup:
        json.dump({"path": path, "old_sha256": old_sha, "header_length": length,
                   "header_base64": base64.b64encode(header).decode()}, backup)

    directory, name = os.path.split(path)
    staging = os.path.join(directory, f".{name}.{uuid.uuid4()}.tmp")
    data_digest_old = hashlib.sha256()
    data_digest_new = hashlib.sha256()
    try:
        with open(path, "rb") as source, open(staging, "xb") as target:
            source.seek(8 + length)
            target.write(struct.pack("<Q", len(new_header)))
            target.write(new_header)
            while True:
                block = source.read(CHUNK)
                if not block:
                    break
                data_digest_old.update(block)
                target.write(block)
            target.flush()
            fcntl.fcntl(target.fileno(), F_FULLFSYNC)
        with open(staging, "rb") as written:
            written.seek(8 + len(new_header))
            while True:
                block = written.read(CHUNK)
                if not block:
                    break
                data_digest_new.update(block)
        if data_digest_old.digest() != data_digest_new.digest():
            raise Skip("data region differs after copy")
        os.chmod(staging, stat.S_IMODE(info.st_mode))
        os.utime(staging, ns=(info.st_atime_ns, info.st_mtime_ns))
        now = os.lstat(path)
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
            raise Skip("file changed while being edited")
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
    return {"old_sha256": old_sha, "new_sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit", required=True, help="the audit's JSON companion")
    parser.add_argument("--manifest", required=True, help="where to write the per-file manifest (must not exist)")
    parser.add_argument("--backup-dir", help="folder for original headers (must not exist); required with --apply")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted --apply into its existing backup folder")
    args = parser.parse_args()
    if os.path.exists(args.manifest):
        sys.exit(f"refusing: {args.manifest} exists")
    if args.apply:
        if not args.backup_dir:
            sys.exit("--apply needs --backup-dir")
        os.makedirs(args.backup_dir, exist_ok=args.resume)

    audit = json.load(open(args.audit))
    store = audit["store"]
    rows = []
    for candidate in audit["candidates"]:
        tail = candidate.get("proposed")
        row = {"path": candidate["path"], "status": candidate["status"], "tail": tail}
        if tail is None:
            row["outcome"] = "left alone (no tail resolved)"
            rows.append(row)
            continue
        path = os.path.join(store, candidate["path"])
        try:
            result = edit(path, candidate["path"], candidate["sha256"], tail, args.backup_dir, args.apply)
            earlier = result.pop("edited_earlier", False)
            row.update(result)
            row["outcome"] = ("edited (earlier run)" if earlier else "edited") if args.apply else "would edit"
        except Skip as reason:
            row["outcome"] = f"skipped: {reason}"
        rows.append(row)
        print(f"{row['outcome']:<40} {candidate['path']}", flush=True)

    with open(args.manifest, "x") as manifest:
        json.dump({"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "applied": args.apply, "key": KEY, "files": rows}, manifest, indent=1)
    counts = {}
    for row in rows:
        outcome = row["outcome"].split(":")[0]
        counts[outcome] = counts.get(outcome, 0) + 1
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
