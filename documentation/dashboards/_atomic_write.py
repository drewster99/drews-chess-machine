"""Crash-safe replacement of the dashboard trackers' source-of-truth files.

data/<run>.csv (which carries imported probes that exist nowhere else),
selfplay_probe/<run>.csv, and registry.json are rewritten by a cron tick every
few minutes. Rewriting them with a plain `open(path, "w")` truncates the file
first, so a kill, a full disk, or an exception between the truncate and the last
row leaves a short or empty file behind -- silently destroying history that the
next tick then reads back as if it were complete.

`atomic_write_open` is the ONE place that knows how to write such a file
safely, so every writer gets the same guarantees:

  1. The new content goes to a uniquely named hidden temp file in the SAME
     directory as the target (same filesystem, so the final rename is atomic).
     The temp name does not end in the target's extension, so nothing that scans
     for `*.csv` / `*.json` can mistake an in-flight write for a real file.
  2. On a clean exit the temp file is flushed and synced to stable storage, then
     `os.replace`d onto the target, and the directory is synced so the rename
     itself survives a power loss. A reader therefore sees either the complete
     old file or the complete new file, never a mixture or a truncation.
  3. On ANY exception the temp file this call created is removed and the
     exception propagates unchanged; the target is never touched. Nothing is
     swallowed.

The handle is opened with exactly the `newline` / `encoding` the caller passes,
mirroring builtin `open`, so swapping `open(p, "w", newline="")` for
`atomic_write_open(p, newline="")` produces byte-identical output (the csv
module's CRLF row terminator included).

Permissions: a plain truncate-and-write keeps the existing file's mode, whereas
a replace installs a brand-new inode. To keep that behaviour the temp file takes
the target's current permission bits when the target exists; otherwise it is
created with the umask-filtered default mode, as `open` would.

Symlinks: a plain `open` writes through a symlink to its referent. To keep that,
the target is resolved with `os.path.realpath` and the temp file + replace
happen next to the real file, so a symlinked target stays a symlink.

Durability uses `F_FULLFSYNC` where the platform has it (macOS), because there a
plain `fsync` only hands the data to the drive's volatile cache.

A SIGKILL or power loss between creating the temp file and the replace leaves a
hidden `.<name>.tmp-<pid>-<token>` file next to the target. It is inert (no
reader looks at it) and safe to delete; the target itself is intact.

This module has no import-time side effects.
"""
import contextlib
import fcntl
import os
import secrets
import stat


def _sync_to_stable_storage(fd):
    """Flush a file or directory descriptor all the way to stable storage."""
    if hasattr(fcntl, "F_FULLFSYNC"):
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


def _sync_directory(directory_path):
    directory_fd = os.open(directory_path, os.O_RDONLY)
    try:
        _sync_to_stable_storage(directory_fd)
    finally:
        os.close(directory_fd)


@contextlib.contextmanager
def atomic_write_open(path, newline=None, encoding=None):
    """Open `path` for text writing such that the target is replaced atomically
    and durably when the `with` block exits normally, and left untouched (with
    the temp file removed and the exception re-raised) when it does not.

    `newline` and `encoding` have builtin `open` semantics and must match what
    the replaced `open(path, "w", ...)` call used, to keep output byte-identical.
    """
    target_path = os.path.realpath(path)
    target_directory = os.path.dirname(target_path)
    temp_path = os.path.join(
        target_directory,
        f".{os.path.basename(target_path)}.tmp-{os.getpid()}-{secrets.token_hex(6)}")
    # O_EXCL: the name is unique to this call, so we can never clobber (and later
    # unlink) a file some other writer owns.
    temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        try:
            target_mode = stat.S_IMODE(os.stat(target_path).st_mode)
        except FileNotFoundError:
            target_mode = None
        if target_mode is not None:
            os.fchmod(temp_fd, target_mode)
        # fdopen takes ownership of temp_fd: closing the handle closes the fd.
        handle = os.fdopen(temp_fd, "w", newline=newline, encoding=encoding)
    except BaseException:
        os.close(temp_fd)
        os.unlink(temp_path)
        raise
    try:
        with handle:
            yield handle
            handle.flush()
            _sync_to_stable_storage(handle.fileno())
        os.replace(temp_path, target_path)
    except BaseException:
        # Only reachable before the replace succeeded (nothing after it can
        # raise inside this block), so temp_path is still ours to remove.
        os.unlink(temp_path)
        raise
    _sync_directory(target_directory)
