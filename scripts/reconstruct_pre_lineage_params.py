#!/usr/bin/env python3
"""reconstruct_pre_lineage_params.py — a read-only report of which corpus-replay
session log trained each pre-lineage model file, and that log's hyperparameter lines.

Why this exists. Model files written before architecture format v7 carry no
`dcm_lineage` record, so the parameters behind them are in no model file
(HPARAM_RECORDING_PLAN.md, "Gap 2 — the pre-lineage archive"). Rewriting those files
would be migration, which is ruled out. What survives is the session log of each
corpus-replay process: its `[REPLAY-HPARAMS]` banner (and, from build 2026-09-29 on,
its `[REPLAY-CYCLE]` line) states the configuration the run trained under. This
script pairs each file with the one log that wrote it and reports those lines,
labelled "reconstructed from logs". It never writes into a model file, and its
output is not measured data: nothing here may be fed to lineage tooling or the
dashboards as if it were.

The join. A pre-lineage replay log never names its own run's model ID — the
minted ID went only into the files and `results.json` — so a file cannot be joined
to a log by its own ID alone. A file is matched to a log only when every one of
these agrees, and is reported `unmatched`, with the reason, otherwise:

- file name: a `[REPLAY] saved trainer model (…) step=N … -> <name>` line (the
  rolling file) or a `[REPLAY] enumerated checkpoint -> <name>` line (the
  step-enumerated copy of the save just above it) names the file;
- time: the file's `created_at_unix` is stamped just before the bytes are encoded
  and written, and the log line just after, so the line's time lies between
  `created_at_unix` and `created_at_unix + SAVE_TO_LOG_LINE_MAX_SECONDS`;
- step: the file's `training_step` equals that save's `step=`;
- one log: the matching save lines all lie in a single log (several logs is
  ambiguous, none is a missing log);
- parent model ID: the log's `[REPLAY] start-model: … modelID=<id>` equals the
  file's `parent_model_id` (a log with no start-model line is a fresh run, whose
  files carry no parent);
- own model ID: one replay process mints one model ID, so every file matched to
  one log must carry the same `model_id`, and one `model_id` must not be matched
  to two logs. A contradiction unmatches every file involved.

Session-log lines carry only a local wall-clock time (`HH:mm:ss.SSS`); the date
comes from the log's file name (`dcm_log_YYYYMMDD-HHMMSS[-N].txt`) and advances at
each midnight crossing. Turning that into Unix time needs the time zone the
logging Mac was in, which the log does not record, so `--log-timezone` is
required rather than assumed. A save line in the repeated hour at the end of
daylight saving time has two readings; a match that depends on which one is
right is reported unmatched.

Optionally (`--experiments`), an experiment README that names a matched log by
file name and names exactly one `parameters*.json` in its own folder near that
mention attaches that file as `parameters_file`. It is the file the README says
was passed; the log lines remain the record of what the run resolved.

Usage:
  python3 scripts/reconstruct_pre_lineage_params.py \
      --models ~/Library/Application\\ Support/DrewsChessMachine/Models \
      --logs ~/Library/Logs/DrewsChessMachine \
      --log-timezone America/Chicago \
      [--experiments experiments] \
      --out /path/to/report.json

`--out` must not exist; the report is never written over a file.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import struct
import sys
import zoneinfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcm_lineage  # noqa: E402
from dcm_session_logs import session_log_sort_key  # noqa: E402

PROVENANCE = "reconstructed from logs"

# `ModelCheckpointMetadata.creator` of every corpus-replay save. The log lines this
# report joins on are written only by corpus replay, so a file from any other path
# can never be matched; checking the creator first gives such a file a clear reason
# instead of a misleading "no log names this file".
REPLAY_CREATOR = "replay"

# How long after `created_at_unix` the save's log line may be stamped. The stamp is
# taken just before the file is encoded; the rolling-save line follows the encode
# and the synchronous write, and the enumerated-checkpoint line follows the rolling
# write, (in some builds) a layer-health pass, and a second write. Those take
# seconds even for the largest models; two minutes leaves room for a slow disk while
# staying far below the gap between two saves of one file name (every 1000 steps
# for the rolling file, never for an enumerated one). The window only narrows the
# candidates — the name and the step must agree as well.
SAVE_TO_LOG_LINE_MAX_SECONDS = 120.0
# `created_at_unix` is whole seconds, truncated, so it is never after the log line
# in principle; this allows for a small wall-clock adjustment between the two reads.
CLOCK_SLACK_SECONDS = 2.0

# A backward jump of more than this between consecutive line times is a midnight
# crossing (the convention of scripts/analyze_session_log.py). Lines are queued
# asynchronously, so small backward steps of milliseconds are normal and must not
# advance the date.
MIDNIGHT_BACKWARD_JUMP_SECONDS = 12 * 3600

LINE_RE = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d{3})  (.*)")
SAVE_RE = re.compile(r"\[REPLAY\] saved trainer model \(([^)]*)\) step=(\d+)\b.* -> (.+)")
ENUMERATED_RE = re.compile(
    r"\[REPLAY\] enumerated checkpoint -> (.+?)(?: \(replaced this run's own earlier save of step \d+\))?")
START_MODEL_RE = re.compile(r"\[REPLAY\] start-model: (.+) modelID=(\S+) encoding=\S+")
HPARAMS_TAG = "[REPLAY-HPARAMS] "
CYCLE_TAG = "[REPLAY-CYCLE] "
RESUME_WARNING_TAG = "[REPLAY-RESUME] WARNING "
LOG_NAME_IN_TEXT_RE = re.compile(r"dcm_log_\d{8}-\d{6}(?:-\d+)?\.txt")
PARAMETERS_IN_TEXT_RE = re.compile(r"([\w./~-]*?)(parameters[\w.-]*?\.json)")
HEADING_RE = re.compile(r"#{1,6} ")


class RefusedRun(Exception):
    """A usage error that stops the report before anything is written."""


class SaveEvent:
    """One save the log records under one file name: the rolling file's
    saved-trainer-model line, or the enumerated copy written from the same bytes."""

    def __init__(self, log, name, step, line, unix_readings):
        self.log = log
        self.name = name
        self.step = step
        self.line = line
        # One Unix time normally; two for a local time in the repeated
        # end-of-daylight-saving hour.
        self.unix_readings = unix_readings


class LogFacts:
    """The lines of one session log this report uses, in log order."""

    def __init__(self, path):
        self.path = path
        self.hparams_lines = []
        self.cycle_lines = []
        self.start_model_lines = []
        self.resume_warning_lines = []
        self.events = []


def unix_readings(local_datetime, zone):
    """Every Unix time a naive local wall-clock time can mean in `zone`, sorted:
    one, or two in the repeated hour (and for a time skipped by the spring-forward
    jump, which `zoneinfo` maps to both neighbouring offsets)."""
    readings = {local_datetime.replace(tzinfo=zone, fold=fold).timestamp() for fold in (0, 1)}
    return sorted(readings)


def scan_log(path, zone):
    """Read one session log once, streaming, and collect its replay lines with the
    Unix time of each save line."""
    date_text, time_text, _ = session_log_sort_key(path)
    day = datetime.datetime.strptime(date_text, "%Y%m%d").date()
    # The name is stamped when the logger starts, before any line, so it is the
    # first "previous time" a midnight crossing is measured against.
    previous = int(time_text[:2]) * 3600 + int(time_text[2:4]) * 60 + int(time_text[4:6])
    facts = LogFacts(path)
    last_save_step = None
    with open(path, "r", errors="replace") as handle:
        for raw in handle:
            match = LINE_RE.match(raw.rstrip("\n"))
            if not match:
                continue
            hours, minutes, seconds, millis = (int(match.group(i)) for i in range(1, 5))
            second_of_day = hours * 3600 + minutes * 60 + seconds + millis / 1000
            if second_of_day + MIDNIGHT_BACKWARD_JUMP_SECONDS < previous:
                day += datetime.timedelta(days=1)
            previous = second_of_day
            message = match.group(5)
            if "[REPLAY" not in message:
                continue
            if message.startswith(HPARAMS_TAG):
                facts.hparams_lines.append(message)
            elif message.startswith(CYCLE_TAG):
                facts.cycle_lines.append(message)
            elif message.startswith(RESUME_WARNING_TAG):
                facts.resume_warning_lines.append(message)
            elif START_MODEL_RE.fullmatch(message):
                facts.start_model_lines.append(message)
            else:
                save = SAVE_RE.fullmatch(message)
                enumerated = None if save else ENUMERATED_RE.fullmatch(message)
                if not save and not enumerated:
                    continue
                local = datetime.datetime.combine(day, datetime.time(hours, minutes, seconds, millis * 1000))
                if save:
                    last_save_step = int(save.group(2))
                    name, step = save.group(3), last_save_step
                else:
                    # The enumerated copy is written from the bytes of the save
                    # logged just before it, so it carries that save's step. One
                    # with no save before it cannot be given a step.
                    name, step = enumerated.group(1), last_save_step
                facts.events.append(SaveEvent(path, name, step, message, unix_readings(local, zone)))
    return facts


def log_paths(folder):
    """Every session log directly in `folder`, oldest launch first. A name that
    matches the glob but not the naming scheme raises (dcm_session_logs)."""
    return sorted(glob.glob(os.path.join(folder, "dcm_log_*.txt")), key=session_log_sort_key)


def model_file_paths(arguments):
    """The .safetensors files named by `--models`: each file as given, and for a
    folder the files `dcm_lineage.model_paths` reads (directly in it and in each
    .dcmsession folder in it). Duplicates (a file named twice) are reported once."""
    paths, seen = [], set()
    for argument in arguments:
        path = os.path.expanduser(argument)
        if os.path.isdir(path):
            found = dcm_lineage.model_paths(path)
        elif os.path.isfile(path):
            if not path.endswith(".safetensors"):
                raise RefusedRun(f"{path}: not a .safetensors file")
            found = [path]
        else:
            raise RefusedRun(f"{path}: no such file or folder")
        for one in found:
            real = os.path.realpath(one)
            if real not in seen:
                seen.add(real)
                paths.append(os.path.abspath(one))
    return paths


def event_in_window(event, created_at):
    """'yes', 'no', or 'ambiguous' (the line's two daylight-saving readings
    disagree about whether it lies in the save window of `created_at`)."""
    fits = {-CLOCK_SLACK_SECONDS <= reading - created_at <= SAVE_TO_LOG_LINE_MAX_SECONDS
            for reading in event.unix_readings}
    if fits == {True}:
        return "yes"
    if fits == {False}:
        return "no"
    return "ambiguous"


def file_facts(path):
    """(metadata, None) for a pre-lineage file, or (metadata-or-None, entry) when
    the file is settled without a log: it has lineage, or it cannot be read."""
    entry = dict(model_id=None, file=path, source_log=None, hparams_line=None, cycle_line=None)
    try:
        metadata = dcm_lineage.read_metadata(path)
    except (OSError, ValueError, struct.error) as error:
        return None, dict(entry, status="unmatched", reason=f"header not readable: {error}")
    entry["model_id"] = metadata.get("model_id")
    if dcm_lineage.METADATA_KEY in metadata:
        return metadata, dict(entry, status="has_lineage",
                              reason=f"carries {dcm_lineage.METADATA_KEY}; its own record holds its parameters")
    try:
        lineage = dcm_lineage.lineage_of(metadata, path)
    except dcm_lineage.LineageError as error:
        return metadata, dict(entry, status="unmatched", reason=f"header refused: {error}")
    if not isinstance(lineage, dcm_lineage.Unrecorded):
        raise AssertionError(f"{path}: a record without {dcm_lineage.METADATA_KEY} cannot exist")
    return metadata, None


def _integer(metadata, key):
    """A metadata integer stored as text, or None when absent or not an integer."""
    value = metadata.get(key)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def match_file(path, metadata, events_by_name, logs):
    """The report entry for one pre-lineage file (before the cross-file model-ID
    checks in `apply_model_id_checks`)."""
    model_id = metadata.get("model_id")
    entry = dict(model_id=model_id, file=path, source_log=None, hparams_line=None, cycle_line=None)

    def unmatched(reason, **extra):
        return dict(entry, status="unmatched", reason=reason, **extra)

    if not model_id:
        return unmatched("no model_id in the header")
    creator = metadata.get("creator")
    if creator != REPLAY_CREATOR:
        return unmatched(f"creator is {creator!r}, not {REPLAY_CREATOR!r}: only corpus-replay "
                         f"saves are named in the log lines this report joins on")
    created_at = _integer(metadata, "created_at_unix")
    if created_at is None:
        return unmatched(f"created_at_unix is missing or not an integer ({metadata.get('created_at_unix')!r})")
    training_step = _integer(metadata, "training_step")
    if training_step is None:
        return unmatched(f"training_step is missing or not an integer ({metadata.get('training_step')!r})")
    name = os.path.basename(path)
    named = events_by_name.get(name, [])
    if not named:
        return unmatched(f"no session log in the log folder has a save line naming {name}")
    in_window, ambiguous = [], []
    for event in named:
        verdict = event_in_window(event, created_at)
        if verdict == "yes":
            in_window.append(event)
        elif verdict == "ambiguous":
            ambiguous.append(event)
    if ambiguous:
        return unmatched("a save line naming this file has a local time in the repeated daylight-saving "
                         "hour, and its two readings disagree about the save window",
                         candidate_lines=[f"{os.path.basename(e.log)}: {e.line}" for e in ambiguous])
    if not in_window:
        return unmatched(f"{len(named)} save line(s) name {name}, none stamped within "
                         f"{SAVE_TO_LOG_LINE_MAX_SECONDS:g} s after created_at_unix {created_at}",
                         candidate_logs=sorted({os.path.basename(e.log) for e in named}))
    candidate_logs = sorted({e.log for e in in_window}, key=session_log_sort_key)
    if len(candidate_logs) > 1:
        return unmatched(f"ambiguous: {len(candidate_logs)} logs have a save line naming {name} "
                         f"within the save window",
                         candidate_logs=[os.path.basename(p) for p in candidate_logs])
    log = logs[candidate_logs[0]]
    log_name = os.path.basename(log.path)
    steps = sorted({e.step for e in in_window}, key=lambda s: -1 if s is None else s)
    agreeing = [e for e in in_window if e.step == training_step]
    if not agreeing:
        return unmatched(f"training_step {training_step} differs from the step of every save line naming "
                         f"{name} in {log_name} within the window ({steps})",
                         candidate_logs=[log_name])
    if len(log.hparams_lines) != 1:
        return unmatched(f"{log_name} has {len(log.hparams_lines)} [REPLAY-HPARAMS] lines; exactly one "
                         f"is needed (none: a build before the banner)", candidate_logs=[log_name])
    if len(log.cycle_lines) > 1:
        return unmatched(f"{log_name} has {len(log.cycle_lines)} [REPLAY-CYCLE] lines; one process logs "
                         f"at most one", candidate_logs=[log_name])
    if len(log.start_model_lines) > 1:
        return unmatched(f"{log_name} has {len(log.start_model_lines)} start-model lines; one process "
                         f"logs at most one", candidate_logs=[log_name])
    parent = metadata.get("parent_model_id") or None
    start_line = log.start_model_lines[0] if log.start_model_lines else None
    log_parent = START_MODEL_RE.fullmatch(start_line).group(2) if start_line else None
    if parent != log_parent:
        return unmatched(f"parent_model_id {parent!r} differs from {log_name}'s start-model modelID "
                         f"{log_parent!r}", candidate_logs=[log_name])
    reconstructed = dict(
        entry, status="reconstructed", provenance=PROVENANCE,
        source_log=log.path,
        hparams_line=log.hparams_lines[0],
        cycle_line=log.cycle_lines[0] if log.cycle_lines else None,
        start_model_line=start_line,
        resume_warning_lines=list(log.resume_warning_lines),
        save_lines=[e.line for e in agreeing],
        agreement=dict(file_name=name, created_at_unix=created_at, training_step=training_step,
                       parent_model_id=parent,
                       save_line_unix=[e.unix_readings[0] for e in agreeing]))
    if not log.cycle_lines:
        reconstructed["cycle_line_note"] = (f"{log_name} has no [REPLAY-CYCLE] line (builds before "
                                            f"2026-09-29 did not log one); nothing is inferred")
    return reconstructed


def apply_model_id_checks(entries):
    """One replay process mints one model ID: unmatch every reconstructed entry of a
    log whose files carry different model IDs, and of a model ID matched to more
    than one log. Applied together, from the entries as `match_file` left them."""
    reconstructed = [e for e in entries if e["status"] == "reconstructed"]
    ids_by_log, logs_by_id = {}, {}
    for entry in reconstructed:
        ids_by_log.setdefault(entry["source_log"], set()).add(entry["model_id"])
        logs_by_id.setdefault(entry["model_id"], set()).add(entry["source_log"])
    for entry in reconstructed:
        log_ids = ids_by_log[entry["source_log"]]
        id_logs = logs_by_id[entry["model_id"]]
        reasons = []
        if len(log_ids) > 1:
            reasons.append(f"files matched to {os.path.basename(entry['source_log'])} carry different "
                           f"model IDs {sorted(log_ids)}; one replay process mints one")
        if len(id_logs) > 1:
            reasons.append(f"model ID {entry['model_id']} is matched to several logs "
                           f"{sorted(os.path.basename(p) for p in id_logs)}")
        if reasons:
            candidate = entry["source_log"]
            for key in ("provenance", "start_model_line", "resume_warning_lines", "save_lines",
                        "agreement", "cycle_line_note"):
                entry.pop(key, None)
            entry.update(status="unmatched", reason="; ".join(reasons), source_log=None,
                         hparams_line=None, cycle_line=None,
                         candidate_logs=[os.path.basename(candidate)])


def readme_sections(text):
    """The README split at markdown headings: a list of section texts."""
    sections, current = [], []
    for line in text.splitlines():
        if HEADING_RE.match(line) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def own_parameters_files(text, folder):
    """The parameters*.json files of `folder` that `text` names: a bare name, or a
    path whose last folder is this experiment's. A path into another experiment
    (e.g. "a copy of `other-exp/parameters.json`") names that folder's file, not
    this one's."""
    names = set()
    folder_name = os.path.basename(os.path.normpath(folder))
    for match in PARAMETERS_IN_TEXT_RE.finditer(text):
        prefix, name = match.group(1), match.group(2)
        if prefix:
            # A prefix without a trailing "/" is the start of a longer word
            # ("my_parameters.json"), not a folder.
            if not prefix.endswith("/"):
                continue
            if os.path.basename(os.path.normpath(prefix)) not in (folder_name, "."):
                continue
        if os.path.isfile(os.path.join(folder, name)):
            names.add(name)
    return names


def experiment_parameters(experiments_folder):
    """{log file name: [(readme path, {parameters file paths named with it})]}.

    A README that names one parameters file in its own folder ties it to every log
    it names. A README naming several ties to a log only the ones named in the
    sections (between markdown headings) that also name that log."""
    by_log = {}
    for readme in sorted(glob.glob(os.path.join(experiments_folder, "*", "README.md"))):
        folder = os.path.dirname(readme)
        with open(readme, "r", errors="replace") as handle:
            text = handle.read()
        logs = set(LOG_NAME_IN_TEXT_RE.findall(text))
        if not logs:
            continue
        whole = own_parameters_files(text, folder)
        for log_name in logs:
            if len(whole) <= 1:
                named = whole
            else:
                named = set()
                for section in readme_sections(text):
                    if log_name in section:
                        named |= own_parameters_files(section, folder)
            by_log.setdefault(log_name, []).append(
                (readme, {os.path.join(folder, name) for name in named}))
    return by_log


def attach_parameters_files(entries, by_log):
    """Set `parameters_file` on each reconstructed entry whose log exactly one
    README names together with exactly one parameters file; otherwise leave it out
    and say why in `parameters_file_note`."""
    for entry in entries:
        if entry["status"] != "reconstructed":
            continue
        log_name = os.path.basename(entry["source_log"])
        readmes = by_log.get(log_name, [])
        if not readmes:
            entry["parameters_file_note"] = "no experiment README names this log"
            continue
        if len(readmes) > 1:
            entry["parameters_file_note"] = ("several experiment READMEs name this log: "
                                             + ", ".join(r for r, _ in readmes))
            continue
        readme, files = readmes[0]
        if len(files) == 1:
            entry["parameters_file"] = next(iter(files))
            entry["parameters_file_readme"] = readme
        elif not files:
            entry["parameters_file_note"] = f"{readme} names this log but no parameters*.json of its folder"
        else:
            entry["parameters_file_note"] = (f"{readme} names several parameters files with this log: "
                                             + ", ".join(sorted(files)))


def build_report(model_arguments, log_folder, zone_name, experiments_folder=None):
    """The report as a dict. Reads headers, logs and READMEs only."""
    zone = zoneinfo.ZoneInfo(zone_name)
    paths = model_file_paths(model_arguments)
    logs = {}
    events_by_name = {}
    for log_path in log_paths(log_folder):
        facts = scan_log(log_path, zone)
        logs[log_path] = facts
        for event in facts.events:
            events_by_name.setdefault(event.name, []).append(event)
    entries = []
    for path in paths:
        metadata, settled = file_facts(path)
        entries.append(settled if settled is not None else match_file(path, metadata, events_by_name, logs))
    apply_model_id_checks(entries)
    if experiments_folder is not None:
        attach_parameters_files(entries, experiment_parameters(experiments_folder))
    counts = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return dict(
        report="pre-lineage parameter reconstruction",
        provenance=PROVENANCE,
        note=("Values are copied from session-log lines, not measured; never write them into a model "
              "file or feed them to lineage or the dashboards as recorded data."),
        log_folder=os.path.abspath(log_folder),
        log_timezone=zone_name,
        logs_scanned=len(logs),
        save_to_log_line_max_seconds=SAVE_TO_LOG_LINE_MAX_SECONDS,
        clock_slack_seconds=CLOCK_SLACK_SECONDS,
        experiments_folder=os.path.abspath(experiments_folder) if experiments_folder else None,
        counts=counts,
        entries=entries)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only: pair each pre-lineage model file with the corpus-replay session log "
                    "that wrote it and report that log's hyperparameter lines, as JSON.")
    parser.add_argument("--models", nargs="+", required=True,
                        help=".safetensors files or folders (a folder's files and its .dcmsession folders)")
    parser.add_argument("--logs", required=True, help="the session-log folder (dcm_log_*.txt)")
    parser.add_argument("--log-timezone", required=True,
                        help="IANA time zone the logging Mac was in, e.g. America/Chicago")
    parser.add_argument("--experiments", help="experiments folder whose READMEs name logs and parameters files")
    parser.add_argument("--out", required=True, help="report path; refused if it exists")
    args = parser.parse_args(argv)
    out = os.path.expanduser(args.out)
    try:
        if os.path.lexists(out):
            raise RefusedRun(f"{out} exists; the report is never written over a file")
        if not os.path.isdir(os.path.expanduser(args.logs)):
            raise RefusedRun(f"{args.logs}: not a folder")
        if args.experiments is not None and not os.path.isdir(os.path.expanduser(args.experiments)):
            raise RefusedRun(f"{args.experiments}: not a folder")
        try:
            zoneinfo.ZoneInfo(args.log_timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
            raise RefusedRun(f"--log-timezone {args.log_timezone!r}: {error}") from None
        report = build_report(args.models, os.path.expanduser(args.logs), args.log_timezone,
                              os.path.expanduser(args.experiments) if args.experiments else None)
        text = json.dumps(report, indent=1) + "\n"
        # Exclusive create: a file that appeared at `out` while the logs were being
        # read is refused here too, never replaced.
        try:
            with open(out, "x") as handle:
                handle.write(text)
        except FileExistsError:
            raise RefusedRun(f"{out} exists; the report is never written over a file") from None
    except (RefusedRun, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    summary = ", ".join(f"{k}={v}" for k, v in sorted(report["counts"].items())) or "no files"
    print(f"wrote {out}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
