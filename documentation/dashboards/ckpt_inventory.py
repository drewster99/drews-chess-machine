#!/usr/bin/env python3
"""Identify safetensors checkpoints by their embedded metadata rather than by filename.

Why this exists: before architecture format v11 the corpus-replay runner's
--enumerate-checkpoints wrote <stem>-replay-step<N>.safetensors using the
SEGMENT-LOCAL step number, and every resumed segment restarted that counter at 1.
Across the v5 lineage five segments therefore competed for the same names — four
different files have been called `v5-cont-replay-step1000.safetensors`, and later
runs silently overwrote earlier ones as they climbed. A filename identifies nothing.
(From v11 the name and `training_step` carry the trainer step, but files of both
eras sit side by side.)

The authoritative identity is the safetensors `__metadata__` header:
`model_id` (minted per segment) plus `training_step` — the writing segment's
step on a corpus-replay or train-vs-UCI file before format v11, the trainer step
from v11 (each entry's `step_basis`, `trainer_step` and `segment_step` say which,
from `dcm_lineage.step_reading`). Files at architecture format v7 and later also
carry a lineage record (`dcm_lineage`,
read through scripts/dcm_lineage.py), which names the run and segment outright
and carries the corpus position that format no longer writes as `replay_*`
keys; each entry's `lineage` is that record's summary, or "unrecorded (format
vN)" for an older file. Together
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import dcm_lineage  # noqa: E402  the file's lineage record, read the way the app reads it

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


def lineage_summary(meta, source):
    """The fields of a file's lineage record worth an inventory column: run,
    segment, cumulative totals and the corpus position. "unrecorded (format vN)"
    for a file written before records existed. Raises dcm_lineage.LineageError
    where the app would refuse the file."""
    record = dcm_lineage.lineage_of(meta, source)
    if isinstance(record, dcm_lineage.Unrecorded):
        return f"unrecorded (format v{record.format_version})"
    corpus = record["fed"]["corpus"]
    return dict(lineage_run_id=record["run"]["lineage_run_id"],
                segment_index=record["run"]["segment_index"],
                segment_id=record["run"]["segment_id"],
                start=record["run"]["start"],
                exact_resume=record["run"]["exact_resume"],
                cum_trainer_step=record["steps"]["cum_trainer_step"],
                segment_local_step=record["steps"]["segment_local_step"],
                cum_games=record["fed"]["cum_games"],
                cum_train_step_sec=record["time"]["cum_train_step_sec"],
                # The first corpus fed, plus the whole feed-order list (a
                # schema-3 replay can feed several; schema 2 named one).
                corpus=None if corpus is None else dict(corpus_id=dcm_lineage.corpus_ids(corpus)[0],
                                                        corpus_ids=dcm_lineage.corpus_ids(corpus),
                                                        epoch=corpus["epoch"],
                                                        next_game_index=corpus["next_game_index"]),
                path_kind=record["invocation"]["path_kind"])


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
            try:
                e["lineage"] = lineage_summary(meta, os.path.basename(path))
                reading = dcm_lineage.step_reading(meta, os.path.basename(path))
                e["step_basis"] = reading.basis
                e["trainer_step"] = reading.trainer_step
                e["segment_step"] = reading.segment_step
            except dcm_lineage.LineageError as err:
                # A file the app would refuse is a finding, recorded like a
                # truncated one rather than dropped.
                e["error"] = str(err)
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
        # The trainer step where the file records one, else the step it states.
        steps = sorted(x["trainer_step"] if x.get("trainer_step") is not None else int(x["training_step"])
                       for x in v if x.get("trainer_step") is not None or "training_step" in x)
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
