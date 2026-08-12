#!/usr/bin/env python3
"""Identify safetensors checkpoints by their embedded metadata rather than by filename.

Why this exists: the corpus-replay runner's --enumerate-checkpoints writes
<stem>-replay-step<N>.safetensors using the SEGMENT-LOCAL step number, and every
resumed segment restarts that counter at 1. Across the v5 lineage five segments
therefore competed for the same names — four different files have been called
`v5-cont-replay-step1000.safetensors`, and later runs silently overwrote earlier
ones as they climbed. A filename identifies nothing.

The authoritative identity is the safetensors `__metadata__` header:
`model_id` (minted per segment) plus `training_step` (segment-local). Together
they name a checkpoint uniquely across the whole lineage. This module reads only
that header -- an 8-byte length prefix plus the JSON that follows -- so it never
loads a 33 MB tensor payload just to ask which run a file belongs to.

A monitor that trusted filenames once produced nine fabricated data points by
re-probing month-old files, so treat the metadata as the only source of truth and
mtime as corroboration, never the reverse.

Usage:
    python3 -I ckpt_inventory.py <dir> [<dir> ...] [--sha256] [--out inventory.json]

--sha256 hashes full file contents (~34 MB each), which is what makes the output
usable as a durable integrity manifest; without it the scan is metadata-only and
returns in seconds.
"""
import os, sys, json, glob, struct, hashlib, argparse, datetime

# Header fields worth carrying forward. `replay_next_game_index` and `replay_epoch`
# are what distinguish an honest resume point from one whose corpus position was
# never actually reached (see the v5 run-4 quarantine).
KEEP = ("model_id", "parent_model_id", "training_step", "replay_epoch",
        "replay_next_game_index", "replay_corpus_id", "created_at_unix",
        "built_by_git", "content_sha256", "notes")


def read_metadata(path):
    """Return the safetensors __metadata__ dict, reading only the header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan(dirs, want_sha):
    entries = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            try:
                meta = read_metadata(path)
            except (OSError, ValueError, struct.error) as e:
                # A truncated or non-safetensors file is a finding, not something
                # to skip silently -- record it so the count still reconciles.
                entries.append(dict(path=path, error=str(e)))
                continue
            st = os.stat(path)
            e = dict(path=path, size=st.st_size,
                     mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))
            for k in KEEP:
                if k in meta:
                    e[k] = meta[k]
            if want_sha:
                e["sha256"] = file_sha256(path)
            entries.append(e)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sha256", action="store_true", help="hash full contents (slow)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    entries = scan(a.dirs, a.sha256)

    by_model = {}
    for e in entries:
        by_model.setdefault(e.get("model_id", "<unreadable>"), []).append(e)

    summary = []
    for mid, v in sorted(by_model.items()):
        steps = sorted(int(x["training_step"]) for x in v if "training_step" in x)
        summary.append(dict(model_id=mid, count=len(v),
                            step_min=steps[0] if steps else None,
                            step_max=steps[-1] if steps else None))

    doc = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"),
               dirs=[os.path.abspath(d) for d in a.dirs],
               hashed=a.sha256, total=len(entries),
               by_model_id=summary, checkpoints=entries)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {a.out}: {len(entries)} checkpoints")

    print(f"{'model_id':<20} {'count':>6} {'step_min':>10} {'step_max':>10}")
    for s in summary:
        print(f"{s['model_id']:<20} {s['count']:>6} "
              f"{s['step_min'] if s['step_min'] is not None else '-':>10} "
              f"{s['step_max'] if s['step_max'] is not None else '-':>10}")
    print(f"{'TOTAL':<20} {len(entries):>6}")


if __name__ == "__main__":
    main()
