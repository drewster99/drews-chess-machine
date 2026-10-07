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
  not parse, or with a record of a schema outside `OLDEST_SUPPORTED_SCHEMA` ...
  `SUPPORTED_SCHEMA` (2 ... 3), is refused (the app refuses to load it too).
  A schema-2 record (every file written before hyperparameter recording P4)
  is read as written: it has no `configuration` / `run_seeds` / `ancestry`,
  and its corpus position names one corpus as `corpus_id` / `corpus_path`
  where schema 3 has `corpus_identity` (read both with `corpus_ids`).
- A record's `cum_*` totals are *this run's* totals: a branch restarts them,
  a derive continues its source's. The totals behind the weights themselves
  are `weights_totals`, which adds the runs a schema-3 `ancestry` names.
- A total the record holds as null (a run continuing history written before
  lineage existed) stays absent. A base is the record's total minus its own
  segment's contribution — arithmetic on two measured values — and is never
  filled from anywhere else.
- The flat mirror keys (`lineage_run_id`, `cum_trainer_step`, ...) are never
  read; the JSON value is the only source.

Import from anywhere in the repository with

    sys.path.insert(0, os.path.join(<repo root>, "scripts"))
    import dcm_lineage

No import-time side effects; needs only the standard library and `dcm_arch`
(beside this file in scripts/), which reads the safetensors header.
"""
import datetime
import json
import os
import struct

import dcm_arch

# ArchitectureFormat.lineageRequiredFromVersion: the first format whose files
# must carry a lineage record.
LINEAGE_REQUIRED_FROM_VERSION = 7
# ArchitectureFormat.unversionedLegacyVersion: what a file with no
# dcm_format_version is.
UNVERSIONED_LEGACY_VERSION = 3
# LineageRecord.currentSchema.
SUPPORTED_SCHEMA = 3
# LineageRecord.oldestDecodableSchema: the oldest schema the app still reads.
OLDEST_SUPPORTED_SCHEMA = 2
# LineageRecord.metadataKey.
METADATA_KEY = "dcm_lineage"
FORMAT_VERSION_KEY = "dcm_format_version"
# The header-size bound of the one header reader (`dcm_arch.read_header`),
# re-exported for callers that build a damaged header against it.
MAX_HEADER_BYTES = dcm_arch.MAX_HEADER_BYTES

# The registry segment fields this module derives, in the order they are shown.
DERIVED_SEGMENT_FIELDS = ("lineage_run_id", "segment_id", "model_id", "date", "cumstep_base",
                          "games_base", "elapsed_base_sec", "wall_base_sec", "device")

# Code paths (`invocation.path_kind`) whose segments write files under more than one
# model ID. A GUI session folder holds trainer.safetensors (the trainer generation's
# ID) and champion.safetensors (the champion's ID), and the trainer's ID changes at
# every promotion within one segment. Every other path writes one model ID per
# segment — corpus replay and train-vs-UCI mint it once per process — so there two
# IDs in one segment are a contradiction and are refused. A GUI segment reports its
# IDs as `model_ids` instead of `model_id`; that is a description of the segment, not
# a registry field, so it is not in DERIVED_SEGMENT_FIELDS (the registry tools read
# only replay and train-vs-UCI files).
MULTIPLE_MODEL_ID_PATH_KINDS = frozenset({"gui"})


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
    """The `__metadata__` dict of a .safetensors file, reading only the header:
    `dcm_arch.read_metadata`, whose refusals surface here as LineageError."""
    try:
        return dcm_arch.read_metadata(path)
    except dcm_arch.ArchitectureError as error:
        raise LineageError(str(error)) from error


def format_version_of(metadata, source):
    """The file's format version, read the way `ArchitectureFormat` reads it."""
    value = metadata.get(FORMAT_VERSION_KEY)
    if value is None:
        return UNVERSIONED_LEGACY_VERSION
    try:
        return dcm_arch.parsed_format_version(value)
    except dcm_arch.ArchitectureError:
        raise LineageError(f"{source}: unparseable {FORMAT_VERSION_KEY} {value!r}") from None


# Keys of the record this module reads. Every key of the record is required by
# the app's decoder; these are checked so a malformed record fails here with a
# message rather than as a KeyError deep in the derivation.
_REQUIRED = {
    "": ("schema", "run", "parent", "steps", "fed", "time", "device", "invocation", "segments"),
    "run": ("lineage_run_id", "segment_index", "segment_id", "segment_started_unix", "start",
            "exact_resume", "not_exact_items", "continues_unrecorded_history", "recorded_unix"),
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
# What schema 3 added: required in a schema-3 record, refused in a schema-2 one
# (the app's decoder does both).
_SCHEMA_3_TOP = ("configuration", "run_seeds", "ancestry")
_SCHEMA_3_BUILD = ("git_diff_sha256", "xcode_build", "sdk_build", "configuration")
_SCHEMA_3_SEGMENT_SUMMARY = ("configuration", "parameters", "corpus_identity", "segment_start_corpus",
                             "path_kind", "argv", "run_seeds")


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
    return validated_record(record, source)


def validated_record(record, source, field=METADATA_KEY):
    """`record` (a decoded lineage record) when it has every field the app
    requires at the supported schema; raises `LineageError` otherwise. Shared
    by model files (`lineage_of`, field `dcm_lineage`) and session.json (field
    `lineage`); `field` names where the record was read, for the message."""
    if not isinstance(record, dict):
        raise LineageError(f"{source}: {field} is not a JSON object")
    for section, keys in _REQUIRED.items():
        holder = record if section == "" else record.get(section)
        if not isinstance(holder, dict):
            raise LineageError(f"{source}: {field}.{section} is missing or not an object")
        for key in keys:
            if key not in holder:
                where = f"{section}.{key}" if section else key
                raise LineageError(f"{source}: {field} has no {where}")
    schema = record["schema"]
    if not isinstance(schema, int) or isinstance(schema, bool) \
            or not OLDEST_SUPPORTED_SCHEMA <= schema <= SUPPORTED_SCHEMA:
        raise LineageError(f"{source}: {field} schema {schema!r} is not a supported schema "
                           f"({OLDEST_SUPPORTED_SCHEMA}...{SUPPORTED_SCHEMA})")
    if not isinstance(record["segments"], list):
        raise LineageError(f"{source}: {field}.segments is not a list")
    build = record.get("build")
    if not isinstance(build, dict):
        raise LineageError(f"{source}: {field}.build is missing or not an object")
    _check_schema_keys(record, _SCHEMA_3_TOP, schema, source, field)
    _check_schema_keys(build, _SCHEMA_3_BUILD, schema, source, f"{field}.build")
    for position, summary in enumerate(record["segments"]):
        for key in _REQUIRED_SEGMENT_SUMMARY:
            if key not in summary:
                raise LineageError(f"{source}: {field}.segments[{position}] has no {key}")
        _check_schema_keys(summary, _SCHEMA_3_SEGMENT_SUMMARY, schema, source, f"{field}.segments[{position}]")
    return record


def _check_schema_keys(holder, keys, schema, source, where):
    """Each of schema 3's `keys` is present in `holder` at schema 3 and
    absent at schema 2, as the app's decoder requires."""
    for key in keys:
        if schema >= 3 and key not in holder:
            raise LineageError(f"{source}: {where} has no {key} (schema {schema})")
        if schema < 3 and key in holder:
            raise LineageError(f"{source}: {where} carries {key}, which schema {schema} never wrote")


def corpus_ids(corpus):
    """The corpus IDs a record's corpus position (`fed.corpus`, a dict) fed,
    in feed order: every corpus of a schema-3 `corpus_identity.listed`, the
    first only for `first_only` (a position carried from schema 2) and for a
    schema-2 position's `corpus_id`."""
    identity = corpus.get("corpus_identity")
    if identity is None:
        return [corpus["corpus_id"]]
    if "listed" in identity:
        return [entry["corpus_id"] for entry in identity["listed"]]
    return [identity["first_only"]["corpus_id"]]


# The totals `weights_totals` sums, as named in `steps` / `fed` / `time` and in
# an ancestor's `totals_at_departure`.
WEIGHTS_TOTAL_KEYS = (("steps", "cum_trainer_step"), ("fed", "cum_games"), ("fed", "cum_positions"),
                      ("time", "cum_train_step_sec"), ("time", "cum_wall_sec"))


def weights_totals(record):
    """The totals behind the weights a record describes (plan B4, O-15):
    the record's own `cum_*` totals (this run's, which already include what
    any derive carried from its source) plus, for every ancestor run the
    next run left by a *branch*, that ancestor's `totals_at_departure`.

    A derive continues its source's totals, so adding an ancestor left by a
    derive would count it twice; only branch boundaries add.

    Returns a dict keyed by total name, or None — never a partial sum — when
    the history is not all recorded: a schema-2 record (no ancestry), a
    `history_before_oldest_run` of `unrecorded`, or a null total the sum
    needs."""
    ancestry = record.get("ancestry")
    if ancestry is None or ancestry["history_before_oldest_run"] != "none":
        return None
    totals = {}
    for section, key in WEIGHTS_TOTAL_KEYS:
        value = record[section][key]
        if value is None:
            return None
        totals[key] = value
    # `runs` is oldest first; each entry is the run its successor left. The
    # successor of the last entry is this record's own run.
    for ancestor in ancestry["runs"]:
        if ancestor["left_by"] != "branch":
            continue
        departure = ancestor["totals_at_departure"]
        for _, key in WEIGHTS_TOTAL_KEYS:
            if departure[key] is None:
                return None
            totals[key] += departure[key]
    return totals


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

    `fields` holds the derived registry values that are known (a GUI segment
    holds `model_ids`, every model ID its files carry, in place of `model_id`;
    see MULTIPLE_MODEL_ID_PATH_KINDS); `unrecorded` names the derivable fields
    whose source total the record holds as null.
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


# A session folder's name ends with this (CheckpointPaths.makeSessionDirectoryName);
# its staging folder adds a further extension and is never read.
SESSION_FOLDER_SUFFIX = ".dcmsession"
# SessionCheckpointLayout.trainerFilename: a session folder's trainer file, whose
# record is the run's own progress at the save.
SESSION_TRAINER_FILENAME = "trainer.safetensors"


def display_name(path):
    """A model file's name as reports show it: the file name, prefixed with its
    session folder when it sits in one. Every session folder holds a
    `trainer.safetensors` and a `champion.safetensors`, so the bare name would not
    say which save a file came from."""
    folder = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    return f"{folder}/{name}" if folder.endswith(SESSION_FOLDER_SUFFIX) else name


def model_paths(directory):
    """Every model file a run saved into `directory`, sorted by path: the
    `.safetensors` files directly in it (step-enumerated checkpoints, rolling
    files) and those directly inside each `.dcmsession` folder in it (GUI and
    train-vs-UCI session saves). Other folders — staging folders included — are
    not read."""
    paths = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if name.endswith(".safetensors") and os.path.isfile(full):
            paths.append(full)
        elif name.endswith(SESSION_FOLDER_SUFFIX) and os.path.isdir(full):
            paths += [os.path.join(full, n) for n in sorted(os.listdir(full))
                      if n.endswith(".safetensors") and os.path.isfile(os.path.join(full, n))]
    return sorted(paths)


class FileLineage:
    """One file's header facts: its model ID, step and lineage record."""

    def __init__(self, path, metadata, record):
        self.path = path
        self.name = display_name(path)
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
            lineage = lineage_of(metadata, display_name(path))
        except (OSError, ValueError, struct.error) as error:
            errors[path] = str(error)
            continue
        if isinstance(lineage, Unrecorded):
            unrecorded[path] = lineage.format_version
        else:
            recorded.append(FileLineage(path, metadata, lineage))
    return recorded, unrecorded, errors


def scan(directory):
    """`scan_files` over `model_paths(directory)`, keyed by `display_name`."""
    recorded, unrecorded, errors = scan_files(model_paths(directory))
    return (recorded, {display_name(p): v for p, v in unrecorded.items()},
            {display_name(p): v for p, v in errors.items()})


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
    segment disagree about what they should share (segment index, path kind,
    the segment's bases, and — except on a path in MULTIPLE_MODEL_ID_PATH_KINDS,
    whose segment reports its `model_ids` — model ID), or a file has no model ID:
    a contradiction is reported, never resolved by picking one."""
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
            path_kinds = {f.record["invocation"]["path_kind"] for f in segment_files}
            if len(path_kinds) != 1:
                raise LineageError(f"lineage run {run_id} segment {segment_id}: files disagree on "
                                   f"path_kind {sorted(path_kinds)}")
            model_ids = {f.metadata.get("model_id") for f in segment_files}
            if None in model_ids:
                raise LineageError(f"lineage run {run_id} segment {segment_id}: a file has no model_id")
            single_model = not (path_kinds & MULTIPLE_MODEL_ID_PATH_KINDS)
            if single_model and len(model_ids) != 1:
                raise LineageError(f"lineage run {run_id} segment {segment_id}: files disagree on "
                                   f"model_id {sorted(model_ids)}")
            views = []
            for f in segment_files:
                fields, unrecorded = _segment_fields(f.record, run_origin)
                if single_model:
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
            fields = dict(reference_fields)
            if not single_model:
                fields["model_ids"] = sorted(model_ids)
            segments[index] = DerivedSegment(index, fields, list(reference_unrecorded),
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
    lineage = lineage_of(metadata, display_name(path))
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
                           **{k: s.fields[k] for k in DERIVED_SEGMENT_FIELDS + ("model_ids",) if k in s.fields},
                           unrecorded=s.unrecorded, files=s.files)
                      for s in run.segments])
    return out


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description="Derive each lineage run's segments from the .safetensors headers in a folder "
                    "and in the .dcmsession folders directly inside it (read-only; prints JSON).")
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
