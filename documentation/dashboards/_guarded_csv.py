"""Guarded read-modify-write of the dashboards' source-of-truth files.

`_atomic_write.py` makes each replace crash-safe, but a crash-safe replace can still
durably install the wrong content: a rebuild that silently lost rows (a missing log,
a dropped probe line), or a writer that read the file, worked for minutes, and then
replaced a newer version another writer had just saved. This module adds the two
checks a source of truth needs on top of that:

1. Compare-and-swap. `read_rows` returns a `Snapshot` (the SHA-256 of the exact bytes
   it parsed). `replace_rows` re-reads the file under a lock and refuses with
   `ConcurrentModification` if those bytes changed since, so the slower of two writers
   fails loudly instead of winning silently. Rerunning it redoes its work on the new
   content (every writer here is idempotent).

2. No silent shrink. Before replacing, the old and new rows are matched by a row key
   (`row_key`). A row that disappears, or a cell that goes from a value to blank, is
   printed and refused with `SourceOfTruthShrink` unless the caller allows it: an
   explicit `allow_shrink` (a command-line `--allow-shrink`, used after reading the
   printed diff) or a named set of columns the caller is deliberately re-deriving
   (`allowed_blank_columns`, e.g. internals recomputed from checkpoints that no longer
   exist). The `note` column is free text and exempt. A column the old file's header
   does not have counts as absent, not as blanked.

Locking: an exclusive `flock` on the target's folder (an `O_RDONLY` descriptor of the
directory), held only around re-hash, diff and replace, never across probing or log
parsing. It needs no lock file, survives the `os.replace` inside the folder (the lock
is on the folder, not the file), and is released by the kernel if the process dies.
Locks are not re-entrant: taking a folder's lock while this process already holds it
raises instead of deadlocking.

This module has no import-time side effects.
"""
import contextlib
import csv
import fcntl
import hashlib
import io
import os
import sys

from _atomic_write import atomic_write_open

ABSENT = "absent"


class ConcurrentModification(RuntimeError):
    """The file changed between this writer's read and its replace."""


class SourceOfTruthShrink(RuntimeError):
    """A replace would drop rows or blank values the file holds."""


class Snapshot:
    """What a reader saw: the SHA-256 of the bytes it parsed, or ABSENT for no file."""

    def __init__(self, digest):
        self.digest = digest

    def __repr__(self):
        return f"Snapshot({self.digest[:12]})"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path):
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except FileNotFoundError:
        return None


def _parse(data):
    if data is None:
        return [], None
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return list(reader), reader.fieldnames


def read_rows(path):
    """(rows, header fieldnames or None, Snapshot) from one read of the file's bytes."""
    data = _read_bytes(path)
    rows, fieldnames = _parse(data)
    return rows, fieldnames, Snapshot(ABSENT if data is None else _digest(data))


def read_text(path):
    """(text, Snapshot) from one read of an existing file's bytes."""
    data = _read_bytes(path)
    if data is None:
        raise FileNotFoundError(path)
    return data.decode("utf-8"), Snapshot(_digest(data))


def snapshot_of(path):
    """A Snapshot of the file's current bytes (for writers that parse it themselves)."""
    data = _read_bytes(path)
    return Snapshot(ABSENT if data is None else _digest(data))


_held_folders = set()


@contextlib.contextmanager
def folder_lock(directory):
    directory = os.path.realpath(directory)
    if directory in _held_folders:
        raise RuntimeError(f"folder lock on {directory} is already held by this process (locks do not nest)")
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _held_folders.add(directory)
        try:
            yield
        finally:
            _held_folders.discard(directory)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _blank(value):
    return value is None or str(value).strip() == ""


def shrink_report(old_rows, old_fieldnames, new_rows, row_key, allowed_blank_columns):
    """(dropped keys, [(key, column, old value)] blanked) between two row sets."""
    old_by_key = {}
    for row in old_rows:
        key = row_key(row)
        if key in old_by_key:
            raise ValueError(f"the existing file has row key {key!r} twice; resolve that before rewriting it")
        old_by_key[key] = row
    new_by_key = {}
    for row in new_rows:
        key = row_key(row)
        if key in new_by_key:
            raise ValueError(f"the new rows have row key {key!r} twice")
        new_by_key[key] = row
    dropped = sorted(set(old_by_key) - set(new_by_key), key=str)
    blanked = []
    columns = [c for c in (old_fieldnames or []) if c != "note" and c not in allowed_blank_columns]
    for key, old in old_by_key.items():
        new = new_by_key.get(key)
        if new is None:
            continue
        for column in columns:
            if not _blank(old.get(column)) and _blank(new.get(column)):
                blanked.append((key, column, old[column]))
    return dropped, blanked


def _print_report(path, dropped, blanked):
    print(f"{path}: the rewrite would drop {len(dropped)} row(s) and blank {len(blanked)} cell(s)", file=sys.stderr)
    for key in dropped[:20]:
        print(f"  dropped row {key}", file=sys.stderr)
    for key, column, value in blanked[:20]:
        print(f"  blanked {column} at {key} (was {value})", file=sys.stderr)
    if len(dropped) > 20 or len(blanked) > 20:
        print("  …", file=sys.stderr)


def replace_rows(path, rows, fieldnames, snapshot, row_key, *, allowed_blank_columns=frozenset(),
                 allow_shrink=False):
    """Replace `path` with `rows` (in `fieldnames` order) if it still holds the bytes
    `snapshot` saw and the replace drops or blanks nothing it may not."""
    for row in rows:
        extra = set(row) - set(fieldnames)
        if extra:
            raise ValueError(f"{path}: row {row_key(row)!r} has columns outside the schema: {sorted(extra)}")
    with folder_lock(os.path.dirname(os.path.realpath(path))):
        data = _read_bytes(path)
        current = ABSENT if data is None else _digest(data)
        if current != snapshot.digest:
            raise ConcurrentModification(f"{path} changed since it was read; rerun to redo the update on the new content")
        old_rows, old_fieldnames = _parse(data)
        dropped, blanked = shrink_report(old_rows, old_fieldnames, rows, row_key, allowed_blank_columns)
        if dropped or blanked:
            _print_report(path, dropped, blanked)
            if not allow_shrink:
                raise SourceOfTruthShrink(f"{path}: refusing to drop {len(dropped)} row(s) / blank {len(blanked)} "
                                          f"cell(s); rerun with --allow-shrink after checking the list above")
        with atomic_write_open(path, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})


def replace_text_if_unchanged(path, text, snapshot, *, must_extend=False):
    """Replace a file's whole text if it still holds the bytes `snapshot` saw. With
    `must_extend`, the new text must begin with the old text exactly (an append)."""
    with folder_lock(os.path.dirname(os.path.realpath(path))):
        data = _read_bytes(path)
        current = ABSENT if data is None else _digest(data)
        if current != snapshot.digest:
            raise ConcurrentModification(f"{path} changed since it was read; rerun to redo the update on the new content")
        if must_extend and data is not None and not text.encode("utf-8").startswith(data):
            raise SourceOfTruthShrink(f"{path}: the new text does not extend the existing text")
        with atomic_write_open(path, newline="") as handle:
            handle.write(text)


def replay_row_key(row):
    """Corpus-replay / train-vs-UCI rows: (segment, segment-local step). Unlike cum_step it
    survives a corrected segment base (recompute_cum_steps)."""
    return str(row["segment"]).strip(), int(float(row["meta_step"]))


def selfplay_row_key(row):
    """Self-play rows: the 1000-step bucket, because a rebuild may pick a different
    [STATS] line (a different meta_step) to represent the same bucket."""
    return int(round(float(row["cum_step"]) / 1000))
