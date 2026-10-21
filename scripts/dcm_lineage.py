"""A model file's lineage record and step reading, as the app reads them, and the
run segments they imply.

Since architecture format v7 every DCM `.safetensors` file carries a
`LineageRecord` (Persistence/LineageRecord.swift) as one JSON value under
`__metadata__["dcm_lineage"]`. It states which run and segment wrote the file
and the run's measured totals: trainer steps, games and positions fed, and
measured trainer-step and wall time. Those totals are what the run-tracking
registry used to carry as hand-entered per-segment bases (`cumstep_base`,
`games_base`, `elapsed_base_sec`, `wall_base_sec`); this module derives them
from the files instead.

Every rule about reading a file is the app's: whether a file has a record (by
format version), whether the record decodes at its schema (2 ... 5), what a
header's `training_step` means (`ModelFileStepReading`), and how the puzzle
test-set results read. This module asks the app (`dcm_arch.describe_files`,
`--describe-model`, header-only and batched) and implements none of them; a file
the app refuses is a LineageError carrying the app's reason. What stays here is
analysis of the records the app has checked:

- A record's `cum_*` totals are *this run's* totals: a branch restarts them,
  a derive continues its source's. The totals behind the weights themselves
  are `weights_totals`, which adds the runs a schema-3 `ancestry` names.
- A total the record holds as null (a run continuing history written before
  lineage existed) stays absent. A base is the record's total minus its own
  segment's contribution — arithmetic on two measured values — and is never
  filled from anywhere else.
- A record is given as stored, so each schema keeps its own layout: a schema-2
  record has no `configuration` / `run_seeds` / `ancestry`, and its corpus
  position names one corpus as `corpus_id` / `corpus_path` where schema 3 has
  `corpus_identity` (read both with `corpus_ids`); schema 4 adds
  `model_naming` (read it with `model_naming`).

Import from anywhere in the repository with

    sys.path.insert(0, os.path.join(<repo root>, "scripts"))
    import dcm_lineage

No import-time side effects; needs only the standard library and `dcm_arch`
(beside this file in scripts/).
"""
import datetime
import json
import os

import dcm_arch

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

# TrainingStepBasis raw values: the `basis` of a step reading.
BASIS_TRAINER_STEP = "trainer_step"
BASIS_LEGACY_SEGMENT_STEP = "legacy_segment_step"
BASIS_LEGACY_GUI_TRAINER_STEP = "legacy_gui_trainer_step"
BASIS_LEGACY_UNKNOWN_WRITER = "legacy_unknown_writer"
# Every TrainingStepBasis raw value: what a stated basis (a probe record's
# `step_basis`) must be one of.
STEP_BASES = frozenset({BASIS_TRAINER_STEP, BASIS_LEGACY_SEGMENT_STEP, BASIS_LEGACY_GUI_TRAINER_STEP,
                        BASIS_LEGACY_UNKNOWN_WRITER})


class LineageError(ValueError):
    """A file the app refuses (its reason), or files that contradict each other
    about one segment."""


class Unrecorded:
    """A file written before lineage records existed."""

    def __init__(self, format_version):
        self.format_version = format_version

    def __repr__(self):
        return f"Unrecorded(format v{self.format_version})"

    def __eq__(self, other):
        return isinstance(other, Unrecorded) and other.format_version == self.format_version


def read_metadata(path):
    """The `__metadata__` dict of a .safetensors file as stored (string values),
    reading only the header: `dcm_arch.read_metadata`, whose refusals surface here
    as LineageError. Raw header values; the app's reading of them is `describe`."""
    try:
        return dcm_arch.read_metadata(path)
    except dcm_arch.ArchitectureError as error:
        raise LineageError(str(error)) from error


def describe(path):
    """The app's description of the model file at `path` (`dcm_arch.describe`);
    LineageError when the app refuses the file."""
    try:
        return dcm_arch.describe(path)
    except dcm_arch.ArchitectureError as error:
        raise LineageError(str(error)) from None


def describe_all(paths):
    """The app's descriptions of many files in as few app calls as it takes
    (`dcm_arch.describe_files`): {path: description | dcm_arch.ArchitectureError}.
    Each is then cached, so a later `describe` of one of them starts no app.
    LineageError when the app cannot be run at all."""
    try:
        return dcm_arch.describe_files(paths)
    except dcm_arch.ArchitectureError as error:
        raise LineageError(str(error)) from None


def lineage_in(description):
    """A description's lineage: the record (a dict, as stored) or `Unrecorded`."""
    lineage = description["lineage"]
    if lineage["status"] == "unrecorded":
        return Unrecorded(lineage["format_version"])
    return lineage["record"]


def lineage_of(path):
    """The file's lineage record (a dict), or `Unrecorded` for a file written
    before records existed. LineageError where the app refuses the file."""
    return lineage_in(describe(path))


class StepReading:
    """What one header says about its step, as the app reads it
    (`ModelFileStepReading`). `stated_training_step` is the header value as
    written (None when it states none), `trainer_step` the trainer step its
    weights were taken at (None when never recorded — never reconstructed),
    `segment_step` the steps its writing segment had trained (None when
    unknown), `basis` one of the BASIS_* values, and `legacy_note` the one-line
    flag for a pre-v11 file that states a step (None otherwise)."""

    def __init__(self, basis, stated_training_step, trainer_step, segment_step, legacy_note):
        self.basis = basis
        self.stated_training_step = stated_training_step
        self.trainer_step = trainer_step
        self.segment_step = segment_step
        self.legacy_note = legacy_note

    @property
    def trainer_step_or_stated_step(self):
        """The trainer step where one is known, else the stated step: what
        catalogs order and display by."""
        return self.trainer_step if self.trainer_step is not None else self.stated_training_step

    def __repr__(self):
        return (f"StepReading({self.basis}, stated={self.stated_training_step}, "
                f"trainer={self.trainer_step}, segment={self.segment_step})")


def step_reading_in(description):
    """A description's step reading."""
    reading = description["step_reading"]
    if reading["basis"] not in STEP_BASES:
        raise LineageError(f"{description['source']}: step basis {reading['basis']!r} is not one this module knows")
    return StepReading(reading["basis"], reading["stated_training_step"], reading["trainer_step"],
                       reading["segment_step"], reading["legacy_note"])


def step_reading(path):
    """The file's `StepReading`, as the app reads it."""
    return step_reading_in(describe(path))


def test_set_results(path):
    """The file's puzzle test-set results, as the app reads them
    (`ModelTestSetResultsField.reading`). Returns one of:

    - ("not_recorded", None): no `dcm_test_set_results` key (written before
      test-set results);
    - ("unreadable", reason): the value doesn't read; reported, never fatal
      (the results are informational);
    - ("failed", reason): the writer's evaluation failed and it said why;
    - ("evaluated", record): the record as stored, with `sets` (each: id, title,
      description, fingerprint_sha256, positions, top1_correct, top5_correct,
      avg_correct_probability, avg_correct_rank, nll, pelo or pelo_bound,
      themes [{id, title, correct, total}]).

    Results are comparable between files only for sets with equal
    `fingerprint_sha256`."""
    results = describe(path)["test_set_results"]
    status = results["status"]
    if status == "evaluated":
        return ("evaluated", results["record"])
    if status in ("failed", "unreadable"):
        return (status, results["reason"])
    return ("not_recorded", None)


def model_naming(record):
    """The record's model naming as written (`{"name": str|None, "preset_start":
    {"recorded": bool, "value": {"preset": str, "edited": bool}|None}}`), or
    None when the record does not state it: a schema before 4, or a schema-4
    record whose `model_naming` is `{"recorded": false}`."""
    if record["schema"] < 4:
        return None
    wrapper = record["model_naming"]
    if not wrapper.get("recorded"):
        return None
    return wrapper["value"]


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
    """One file's facts as the app describes it: its model ID, step reading and
    lineage record."""

    def __init__(self, path, description, record):
        self.path = path
        self.name = display_name(path)
        self.description = description
        self.model_id = description["model_id"]
        self.record = record

    @property
    def step_reading(self):
        return step_reading_in(self.description)


def scan_files(paths):
    """Read the lineage of each .safetensors path through the app (header-only,
    batched).

    Returns (recorded, unrecorded, errors): `recorded` is a list of
    FileLineage, `unrecorded` maps path -> format version, and `errors` maps
    path -> message for files that could not be read or that the app refuses.
    Nothing is skipped silently: every path lands in one of the three."""
    recorded, unrecorded, errors = [], {}, {}
    described = describe_all(paths)
    for path in paths:
        description = described[path]
        if isinstance(description, dcm_arch.ArchitectureError):
            errors[path] = str(description)
            continue
        lineage = lineage_in(description)
        if isinstance(lineage, Unrecorded):
            unrecorded[path] = lineage.format_version
        else:
            recorded.append(FileLineage(path, description, lineage))
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
    whose segment reports its `model_ids` — model ID): a contradiction is
    reported, never resolved by picking one. (Every described file has a model
    ID: the app refuses one without.)"""
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
            model_ids = {f.model_id for f in segment_files}
            single_model = not (path_kinds & MULTIPLE_MODEL_ID_PATH_KINDS)
            if single_model and len(model_ids) != 1:
                raise LineageError(f"lineage run {run_id} segment {segment_id}: files disagree on "
                                   f"model_id {sorted(model_ids)}")
            views = []
            for f in segment_files:
                fields, unrecorded = _segment_fields(f.record, run_origin)
                if single_model:
                    fields["model_id"] = f.model_id
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
    measured cumulative games fed and trainer-step seconds, the record's own
    step clock, and the file's step reading (`trainer_step`, `segment_step`,
    `step_basis`). Returns None for a file without a record. Values the record
    holds as null are returned as None, never filled."""
    description = describe(path)
    lineage = lineage_in(description)
    if isinstance(lineage, Unrecorded):
        return None
    reading = step_reading_in(description)
    return dict(lineage_run_id=lineage["run"]["lineage_run_id"],
                segment_id=lineage["run"]["segment_id"],
                segment_index=lineage["run"]["segment_index"],
                segment_local_step=lineage["steps"]["segment_local_step"],
                cum_trainer_step=lineage["steps"]["cum_trainer_step"],
                run_origin=_run_origin(lineage),
                cum_games=lineage["fed"]["cum_games"],
                segment_games=lineage["fed"]["segment_games"],
                cum_train_step_sec=lineage["time"]["cum_train_step_sec"],
                cum_wall_sec=lineage["time"]["cum_wall_sec"],
                trainer_step=reading.trainer_step,
                segment_step=reading.segment_step,
                step_basis=reading.basis)


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
