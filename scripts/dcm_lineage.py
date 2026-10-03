"""A model file's lineage record, read the way the app reads it, and the run
segments it implies.

Since architecture format v7 every DCM `.safetensors` file carries a
`LineageRecord` (Persistence/LineageRecord.swift) as one JSON value under
`__metadata__["dcm_lineage"]`. It states which run and segment wrote the file
and the run's measured totals: trainer steps, games and positions fed, and
measured trainer-step and wall time. Those totals are what the run-tracking
registry used to carry as hand-entered per-segment bases (`cumstep_base`,
`games_base`, `elapsed_base_sec`, `wall_base_sec`); this module derives them
from the files instead.

The rules mirror the app and CLAUDE.md "Run tracking: three axes":

- A file's format version comes from `dcm_format_version`; a file without one
  is the unversioned legacy format (3), as `ArchitectureFormat` reads it.
- A file older than `LINEAGE_REQUIRED_FROM_VERSION` has no record and is
  reported as unrecorded — nothing is reconstructed for it.
- A file at or after that version without a record, with a record that does
  not parse, or with a record of an unknown schema, is refused (the app
  refuses to load it too).
- A total the record holds as null (a run continuing history written before
  lineage existed) stays absent. A base is the record's total minus its own
  segment's contribution — arithmetic on two measured values — and is never
  filled from anywhere else.
- The flat mirror keys (`lineage_run_id`, `cum_trainer_step`, ...) are never
  read; the JSON value is the only source.

Import from anywhere in the repository with

    sys.path.insert(0, os.path.join(<repo root>, "scripts"))
    import dcm_lineage

No import-time side effects; standard library only.
"""
import datetime
import json
import os
import struct

# ArchitectureFormat.lineageRequiredFromVersion: the first format whose files
# must carry a lineage record.
LINEAGE_REQUIRED_FROM_VERSION = 7
# ArchitectureFormat.unversionedLegacyVersion: what a file with no
# dcm_format_version is.
UNVERSIONED_LEGACY_VERSION = 3
# LineageRecord.currentSchema.
SUPPORTED_SCHEMA = 1
# LineageRecord.metadataKey.
METADATA_KEY = "dcm_lineage"
FORMAT_VERSION_KEY = "dcm_format_version"
# A safetensors header larger than this is not a DCM model header (theirs are
# a few kilobytes); refusing it keeps a damaged length prefix from being read
# as a request for gigabytes.
MAX_HEADER_BYTES = 100 * 1024 * 1024

# The registry segment fields this module derives, in the order they are shown.
DERIVED_SEGMENT_FIELDS = ("lineage_run_id", "segment_id", "model_id", "date", "cumstep_base",
                          "games_base", "elapsed_base_sec", "wall_base_sec", "device")


class LineageError(ValueError):
    """A header whose lineage the app would refuse, or files that contradict
    each other about one segment."""


class Unrecorded:
    """A file written before lineage records existed."""

    def __init__(self, format_version):
        self.format_version = format_version

    def __repr__(self):
        return f"Unrecorded(format v{self.format_version})"

    def __eq__(self, other):
        return isinstance(other, Unrecorded) and other.format_version == self.format_version


def read_metadata(path):
    """The `__metadata__` dict of a .safetensors file, reading only the header."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise LineageError(f"{path}: shorter than a safetensors header length")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > MAX_HEADER_BYTES:
            raise LineageError(f"{path}: safetensors header length {header_length} is not a model header")
        raw = handle.read(header_length)
        if len(raw) != header_length:
            raise LineageError(f"{path}: header truncated ({len(raw)} of {header_length} bytes)")
    header = json.loads(raw)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise LineageError(f"{path}: no __metadata__ object in the safetensors header")
    return metadata


def format_version_of(metadata, source):
    """The file's format version, read the way `ArchitectureFormat` reads it."""
    value = metadata.get(FORMAT_VERSION_KEY)
    if value is None:
        return UNVERSIONED_LEGACY_VERSION
    try:
        version = int(value)
    except (TypeError, ValueError):
        raise LineageError(f"{source}: unparseable {FORMAT_VERSION_KEY} {value!r}") from None
    if version <= 0:
        raise LineageError(f"{source}: unparseable {FORMAT_VERSION_KEY} {value!r}")
    return version


# Keys of the record this module reads. Every key of the record is required by
# the app's decoder; these are checked so a malformed record fails here with a
# message rather than as a KeyError deep in the derivation.
_REQUIRED = {
    "": ("schema", "run", "parent", "steps", "fed", "time", "device", "invocation", "segments"),
    "run": ("lineage_run_id", "segment_index", "segment_id", "segment_started_unix", "start",
            "exact_resume", "continues_unrecorded_history", "recorded_unix"),
    "steps": ("cum_trainer_step", "segment_start_trainer_step", "segment_local_step"),
    "fed": ("cum_games", "cum_positions", "segment_games", "segment_positions", "corpus"),
    "time": ("cum_train_step_sec", "cum_wall_sec", "segment_train_step_sec", "segment_wall_sec"),
    "device": ("hw_model", "chip", "is_vm", "os_version", "gpu_name"),
    "invocation": ("argv", "path_kind"),
}
_REQUIRED_SEGMENT_SUMMARY = ("segment_index", "segment_id", "start", "started_unix", "recorded_unix",
                             "start_trainer_step", "end_trainer_step", "segment_local_step",
                             "segment_games", "segment_positions", "segment_train_step_sec",
                             "segment_wall_sec", "exact_resume", "build", "device")


def lineage_of(metadata, source):
    """The file's lineage record (a dict), or `Unrecorded` for a file written
    before records existed. Raises `LineageError` where the app would refuse."""
    version = format_version_of(metadata, source)
    if version < LINEAGE_REQUIRED_FROM_VERSION:
        if METADATA_KEY in metadata:
            raise LineageError(f"{source}: a format v{version} file carries {METADATA_KEY}, "
                               f"which no writer of that format produced")
        return Unrecorded(version)
    text = metadata.get(METADATA_KEY)
    if text is None:
        raise LineageError(f"{source}: format v{version} file has no {METADATA_KEY}")
    try:
        record = json.loads(text)
    except (TypeError, ValueError) as error:
        raise LineageError(f"{source}: {METADATA_KEY} is not JSON ({error})") from None
    if not isinstance(record, dict):
        raise LineageError(f"{source}: {METADATA_KEY} is not a JSON object")
    for section, keys in _REQUIRED.items():
        holder = record if section == "" else record.get(section)
        if not isinstance(holder, dict):
            raise LineageError(f"{source}: {METADATA_KEY}.{section} is missing or not an object")
        for key in keys:
            if key not in holder:
                where = f"{section}.{key}" if section else key
                raise LineageError(f"{source}: {METADATA_KEY} has no {where}")
    if record["schema"] != SUPPORTED_SCHEMA:
        raise LineageError(f"{source}: {METADATA_KEY} schema {record['schema']} is not the supported "
                           f"schema {SUPPORTED_SCHEMA}")
    if not isinstance(record["segments"], list):
        raise LineageError(f"{source}: {METADATA_KEY}.segments is not a list")
    for position, summary in enumerate(record["segments"]):
        for key in _REQUIRED_SEGMENT_SUMMARY:
            if key not in summary:
                raise LineageError(f"{source}: {METADATA_KEY}.segments[{position}] has no {key}")
    return record


def device_label(device):
    """A registry `device` string from a record's device: the chip without its
    'Apple ' prefix, with ' (VM)' on a virtual machine — the form the
    hand-entered registry uses ("M4 Pro", "M5 (VM)"). None when the record does
    not name the chip, or does not say whether it ran in a VM."""
    chip = device.get("chip")
    is_vm = device.get("is_vm")
    if not chip or is_vm is None:
        return None
    label = chip[len("Apple "):] if chip.startswith("Apple ") else chip
    return f"{label} (VM)" if is_vm else label


def _difference(total, part):
    return None if total is None else total - part


# Two files of one segment state its time bases as (cumulative − own segment)
# sums of floating-point seconds; the same base reached from different files
# differs by rounding, so time fields agree within this relative tolerance.
TIME_RELATIVE_TOLERANCE = 1e-9


def _same_view(a, b):
    """Whether two files' views of one segment agree: exactly for every field
    except the float time bases, which agree within the rounding tolerance."""
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        other = b[key]
        if isinstance(value, float) or isinstance(other, float):
            if abs(value - other) > TIME_RELATIVE_TOLERANCE * max(1.0, abs(value), abs(other)):
                return False
        elif value != other:
            return False
    return True


def _local_date(unix_seconds):
    return datetime.datetime.fromtimestamp(unix_seconds).strftime("%Y%m%d")


class DerivedSegment:
    """One segment of a lineage run, as the registry describes a segment.

    `fields` holds the derived registry values that are known; `unrecorded`
    names the derivable fields whose source total the record holds as null.
    `files` lists the checkpoint files of this segment (empty for a segment
    known only from a later record's history)."""

    def __init__(self, segment_index, fields, unrecorded, files, source):
        self.segment_index = segment_index
        self.fields = fields
        self.unrecorded = unrecorded
        self.files = files
        self.source = source

    def __repr__(self):
        return f"DerivedSegment({self.segment_index}, {self.fields}, unrecorded={self.unrecorded})"


class DerivedRun:
    """One lineage run: its segments (oldest first), the code paths that wrote
    its files, its newest file, and whether it continued history written
    before lineage existed (its totals are then null throughout)."""

    def __init__(self, lineage_run_id, segments, path_kinds, latest_file, continues_unrecorded_history):
        self.lineage_run_id = lineage_run_id
        self.segments = segments
        self.path_kinds = path_kinds
        self.latest_file = latest_file
        self.continues_unrecorded_history = continues_unrecorded_history


class FileLineage:
    """One file's header facts: its model ID, step and lineage record."""

    def __init__(self, path, metadata, record):
        self.path = path
        self.name = os.path.basename(path)
        self.metadata = metadata
        self.record = record


def scan_files(paths):
    """Read the lineage of each .safetensors path, header-only.

    Returns (recorded, unrecorded, errors): `recorded` is a list of
    FileLineage, `unrecorded` maps path -> format version, and `errors` maps
    path -> message for files that could not be read or that the app would
    refuse. Nothing is skipped silently: every path lands in one of the
    three."""
    recorded, unrecorded, errors = [], {}, {}
    for path in paths:
        try:
            metadata = read_metadata(path)
            lineage = lineage_of(metadata, os.path.basename(path))
        except (OSError, ValueError, struct.error) as error:
            errors[path] = str(error)
            continue
        if isinstance(lineage, Unrecorded):
            unrecorded[path] = lineage.format_version
        else:
            recorded.append(FileLineage(path, metadata, lineage))
    return recorded, unrecorded, errors


def scan(directory):
    """`scan_files` over every .safetensors file directly in `directory`, keyed
    by file name."""
    paths = [os.path.join(directory, name) for name in sorted(os.listdir(directory))
             if name.endswith(".safetensors") and os.path.isfile(os.path.join(directory, name))]
    recorded, unrecorded, errors = scan_files(paths)
    return (recorded, {os.path.basename(p): v for p, v in unrecorded.items()},
            {os.path.basename(p): v for p, v in errors.items()})


def _segment_fields(record, run_origin):
    """The registry fields one record states about its own segment."""
    run, steps, fed, time = record["run"], record["steps"], record["fed"], record["time"]
    fields, unrecorded = {}, []
    fields["lineage_run_id"] = run["lineage_run_id"]
    fields["segment_id"] = run["segment_id"]
    fields["date"] = _local_date(run["segment_started_unix"])
    start_step = steps["segment_start_trainer_step"]
    if start_step is None or run_origin is None:
        unrecorded.append("cumstep_base")
    else:
        fields["cumstep_base"] = start_step - run_origin
    for name, value in (("games_base", _difference(fed["cum_games"], fed["segment_games"])),
                        ("elapsed_base_sec", _difference(time["cum_train_step_sec"], time["segment_train_step_sec"])),
                        ("wall_base_sec", _difference(time["cum_wall_sec"], time["segment_wall_sec"]))):
        if value is None:
            unrecorded.append(name)
        else:
            fields[name] = value
    device = device_label(record["device"])
    if device is None:
        unrecorded.append("device")
    else:
        fields["device"] = device
    return fields, unrecorded


def _run_origin(record):
    """The trainer step at which the run's segment 0 began: what `cumstep_base`
    is measured from. Taken from the record itself when it is segment 0, else
    from its segment history. None when that step was never recorded."""
    if record["run"]["segment_index"] == 0:
        return record["steps"]["segment_start_trainer_step"]
    for summary in record["segments"]:
        if summary["segment_index"] == 0:
            return summary["start_trainer_step"]
    raise LineageError(f"lineage run {record['run']['lineage_run_id']}: segment "
                       f"{record['run']['segment_index']}'s record has no segment 0 in its history")


def _history_segment(record, summary, run_origin):
    """Registry fields for an earlier segment known only from `record`'s
    history: its bases are the later record's totals minus everything from that
    segment onward."""
    later = [s for s in record["segments"] if s["segment_index"] >= summary["segment_index"]]
    fed, time = record["fed"], record["time"]
    games_after = fed["segment_games"] + sum(s["segment_games"] for s in later)
    train_after = time["segment_train_step_sec"] + sum(s["segment_train_step_sec"] for s in later)
    wall_after = time["segment_wall_sec"] + sum(s["segment_wall_sec"] for s in later)
    fields, unrecorded = {}, []
    fields["lineage_run_id"] = record["run"]["lineage_run_id"]
    fields["segment_id"] = summary["segment_id"]
    fields["date"] = _local_date(summary["started_unix"])
    if summary["start_trainer_step"] is None or run_origin is None:
        unrecorded.append("cumstep_base")
    else:
        fields["cumstep_base"] = summary["start_trainer_step"] - run_origin
    for name, value in (("games_base", _difference(fed["cum_games"], games_after)),
                        ("elapsed_base_sec", _difference(time["cum_train_step_sec"], train_after)),
                        ("wall_base_sec", _difference(time["cum_wall_sec"], wall_after))):
        if value is None:
            unrecorded.append(name)
        else:
            fields[name] = value
    device = device_label(summary["device"])
    if device is None:
        unrecorded.append("device")
    else:
        fields["device"] = device
    # A summary does not carry its files' model ID; it stays unknown here and is
    # filled from a file of that segment when one is scanned.
    unrecorded.append("model_id")
    return fields, unrecorded


def _progress(file):
    """Order of a segment's files: later step first, then later record."""
    record = file.record
    return (record["steps"]["segment_local_step"], record["run"]["recorded_unix"])


def derive_runs(recorded):
    """Group FileLineage entries by lineage run and derive each run's segments.

    Returns {lineage_run_id: DerivedRun}. Raises LineageError when files of one
    segment disagree about what they should share (model ID, segment index, or
    the segment's bases) — a contradiction is reported, never resolved by
    picking one."""
    by_run = {}
    for file in recorded:
        by_run.setdefault(file.record["run"]["lineage_run_id"], []).append(file)
    derived = {}
    for run_id, files in sorted(by_run.items()):
        by_segment = {}
        for file in files:
            by_segment.setdefault(file.record["run"]["segment_id"], []).append(file)
        latest_record_file = max(files, key=lambda f: (f.record["run"]["segment_index"], _progress(f)))
        run_origin = _run_origin(latest_record_file.record)
        segments = {}
        for segment_id, segment_files in by_segment.items():
            indices = {f.record["run"]["segment_index"] for f in segment_files}
            if len(indices) != 1:
                raise LineageError(f"lineage run {run_id} segment {segment_id}: files disagree on "
                                   f"segment_index {sorted(indices)}")
            model_ids = {f.metadata.get("model_id") for f in segment_files}
            if len(model_ids) != 1 or None in model_ids:
                raise LineageError(f"lineage run {run_id} segment {segment_id}: files disagree on "
                                   f"model_id {sorted(str(m) for m in model_ids)}")
            views = []
            for f in segment_files:
                fields, unrecorded = _segment_fields(f.record, run_origin)
                fields["model_id"] = f.metadata["model_id"]
                views.append((fields, tuple(sorted(unrecorded)), f))
            # The most progressed file's view is the one reported; every other
            # file must agree with it.
            views.sort(key=lambda view: _progress(view[2]), reverse=True)
            reference_fields, reference_unrecorded, reference_file = views[0]
            for fields, unrecorded, f in views[1:]:
                if not _same_view(fields, reference_fields) or unrecorded != reference_unrecorded:
                    raise LineageError(
                        f"lineage run {run_id} segment {segment_id}: {f.name} states {fields} "
                        f"(unrecorded {list(unrecorded)}), {reference_file.name} states "
                        f"{reference_fields} (unrecorded {list(reference_unrecorded)})")
            index = indices.pop()
            ordered = sorted(segment_files, key=_progress)
            segments[index] = DerivedSegment(index, reference_fields, list(reference_unrecorded),
                                             [f.name for f in ordered], "file")
        for summary in latest_record_file.record["segments"]:
            index = summary["segment_index"]
            if index in segments:
                continue
            fields, unrecorded = _history_segment(latest_record_file.record, summary, run_origin)
            segments[index] = DerivedSegment(index, fields, unrecorded, [], "history")
        path_kinds = sorted({f.record["invocation"]["path_kind"] for f in files})
        continues = any(f.record["run"]["continues_unrecorded_history"] for f in files)
        derived[run_id] = DerivedRun(run_id, [segments[i] for i in sorted(segments)], path_kinds,
                                     latest_record_file.name, continues)
    return derived


def checkpoint_facts(path):
    """The per-row facts a tracker takes from one checkpoint's record: the
    measured cumulative games fed and trainer-step seconds, and the record's
    own step clock. Returns None for a file without a record. Values the record
    holds as null are returned as None, never filled."""
    metadata = read_metadata(path)
    lineage = lineage_of(metadata, os.path.basename(path))
    if isinstance(lineage, Unrecorded):
        return None
    return dict(lineage_run_id=lineage["run"]["lineage_run_id"],
                segment_id=lineage["run"]["segment_id"],
                segment_index=lineage["run"]["segment_index"],
                segment_local_step=lineage["steps"]["segment_local_step"],
                cum_trainer_step=lineage["steps"]["cum_trainer_step"],
                run_origin=_run_origin(lineage),
                cum_games=lineage["fed"]["cum_games"],
                segment_games=lineage["fed"]["segment_games"],
                cum_train_step_sec=lineage["time"]["cum_train_step_sec"],
                cum_wall_sec=lineage["time"]["cum_wall_sec"])


def segment_table(derived):
    """A JSON-serializable view of `derive_runs`' result."""
    out = {}
    for run_id, run in derived.items():
        out[run_id] = dict(
            path_kinds=run.path_kinds, latest_file=run.latest_file,
            continues_unrecorded_history=run.continues_unrecorded_history,
            segments=[dict(segment_index=s.segment_index, source=s.source,
                           **{k: s.fields[k] for k in DERIVED_SEGMENT_FIELDS if k in s.fields},
                           unrecorded=s.unrecorded, files=s.files)
                      for s in run.segments])
    return out


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description="Derive each lineage run's segments from the .safetensors headers in a folder "
                    "(read-only; prints JSON).")
    parser.add_argument("directory")
    args = parser.parse_args()
    recorded, unrecorded, errors = scan(os.path.expanduser(args.directory))
    try:
        runs = segment_table(derive_runs(recorded))
    except LineageError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    json.dump(dict(runs=runs, unrecorded_files=len(unrecorded), errors=errors), sys.stdout, indent=1)
    print()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
