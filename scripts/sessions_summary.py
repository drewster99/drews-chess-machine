#!/usr/bin/env python3
"""sessions_summary.py — list .dcmsession saves with what their session.json records.

Reads every `.dcmsession` folder directly in a Sessions folder (default
~/Library/Application Support/DrewsChessMachine/Sessions/) and prints one line
per save: when it was saved, its trigger, session ID, trainer -> champion,
trainer steps, training time, self-play games, arenas and promotions, whether
it holds the replay buffer, its size, and — for session.json format v2 and
later — its lineage record (path kind, segment, cumulative trainer step,
exact or not). Read-only.

The trigger comes from the folder name `<YYYYMMDD-HHMMSS>-<sessionID>-<trigger>`
(CheckpointPaths.makeSessionDirectoryName), matched against the session ID
session.json states; a folder renamed away from that shape shows `(renamed)`.
GUI saves are `manual` / `promote` / `periodic` / `sigusr2`; `--train-vs-uci`
saves are `vsuci-periodic` / `vsuci-final` / `vsuci-abort` /
`vsuci-health-stop` (a training-health alarm stopped the run; lineage path
kind `vsuci`).

A folder whose session.json is missing or unreadable, or whose lineage record
the app would refuse, is listed with the reason and makes the exit status 1.

Usage:
  python3 scripts/sessions_summary.py                  # last 20 saves
  python3 scripts/sessions_summary.py --tail 50
  python3 scripts/sessions_summary.py --sessions-dir <folder>
  python3 scripts/sessions_summary.py --json           # the rows as JSON
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcm_lineage  # noqa: E402

SESSIONS_DIR = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Sessions")
SESSION_FOLDER_SUFFIX = dcm_lineage.SESSION_FOLDER_SUFFIX
# SessionCheckpointLayout.stateFilename / replayBufferFilename.
STATE_FILENAME = "session.json"
REPLAY_BUFFER_FILENAME = "replay_buffer.bin"
# SessionCheckpointState.currentFormatVersion / lineageRequiredFromFormatVersion.
CURRENT_FORMAT_VERSION = 2
LINEAGE_REQUIRED_FROM_FORMAT_VERSION = 2
# CheckpointPaths.filenameTimestampFormat ("yyyyMMdd-HHmmss").
FOLDER_TIMESTAMP = re.compile(r"\d{8}-\d{6}")
RENAMED = "(renamed)"


class SessionSummaryError(ValueError):
    """A session folder this summary cannot describe as the app would read it."""


def trigger_of(folder_name, session_id):
    """The trigger the folder name states, or `RENAMED` when the name is not
    `<timestamp>-<session_id>-<trigger>.dcmsession`."""
    if not folder_name.endswith(SESSION_FOLDER_SUFFIX):
        return RENAMED
    stem = folder_name[:-len(SESSION_FOLDER_SUFFIX)]
    prefix_length = len("YYYYMMDD-HHMMSS")
    if not FOLDER_TIMESTAMP.fullmatch(stem[:prefix_length]):
        return RENAMED
    rest = stem[prefix_length:]
    expected = f"-{session_id}-"
    if not rest.startswith(expected) or len(rest) == len(expected):
        return RENAMED
    return rest[len(expected):]


def _required(state, key, source):
    if key not in state:
        raise SessionSummaryError(f"{source}: {STATE_FILENAME} has no {key}")
    return state[key]


def lineage_summary(state, source):
    """The lineage facts of `state`, or None for a format the record predates.
    Raises where the app would refuse the session."""
    version = _required(state, "formatVersion", source)
    if not isinstance(version, int) or not 1 <= version <= CURRENT_FORMAT_VERSION:
        raise SessionSummaryError(f"{source}: unsupported {STATE_FILENAME} formatVersion {version!r}")
    record = state.get("lineage")
    if record is None:
        if version >= LINEAGE_REQUIRED_FROM_FORMAT_VERSION:
            raise SessionSummaryError(f"{source}: format v{version} {STATE_FILENAME} has no lineage")
        return None
    record = dcm_lineage.validated_record(record, source, field="lineage")
    return dict(
        path_kind=record["invocation"]["path_kind"],
        lineage_run_id=record["run"]["lineage_run_id"],
        segment_index=record["run"]["segment_index"],
        exact_resume=record["run"]["exact_resume"],
        not_exact_items=record["run"]["not_exact_items"],
        cum_trainer_step=record["steps"]["cum_trainer_step"],
    )


def folder_size_bytes(folder):
    """Bytes of the regular files in `folder` and below (symbolic links are
    not followed)."""
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            if os.path.isfile(path) and not os.path.islink(path):
                total += os.stat(path).st_size
    return total


def summarize(folder):
    """One row describing the session folder at `folder`. A folder that cannot
    be described carries `error` and nothing it could not read."""
    name = os.path.basename(folder)
    row = {"name": name, "path": folder}
    try:
        row["size_bytes"] = folder_size_bytes(folder)
        state_path = os.path.join(folder, STATE_FILENAME)
        if not os.path.isfile(state_path):
            raise SessionSummaryError(f"{name}: no {STATE_FILENAME}")
        with open(state_path, encoding="utf-8") as handle:
            try:
                state = json.load(handle)
            except ValueError as error:
                raise SessionSummaryError(f"{name}: {STATE_FILENAME} is not JSON ({error})") from None
        if not isinstance(state, dict):
            raise SessionSummaryError(f"{name}: {STATE_FILENAME} is not a JSON object")
        session_id = _required(state, "sessionID", name)
        arenas = _required(state, "arenaHistory", name)
        row.update(
            session_id=session_id,
            trigger=trigger_of(name, session_id),
            format_version=_required(state, "formatVersion", name),
            saved_at_unix=_required(state, "savedAtUnix", name),
            trainer_id=_required(state, "trainerID", name),
            champion_id=_required(state, "championID", name),
            training_steps=_required(state, "trainingSteps", name),
            elapsed_training_sec=_required(state, "elapsedTrainingSec", name),
            self_play_games=_required(state, "selfPlayGames", name),
            arenas=len(arenas),
            promotions=sum(1 for arena in arenas if _required(arena, "promoted", f"{name} arenaHistory")),
            # Optional in session.json: absent in sessions written before the field.
            has_replay_buffer=state.get("hasReplayBuffer"),
            replay_buffer_file=os.path.isfile(os.path.join(folder, REPLAY_BUFFER_FILENAME)),
            lineage=lineage_summary(state, name),
        )
    except (OSError, SessionSummaryError, dcm_lineage.LineageError) as error:
        row["error"] = str(error)
    return row


def collect(sessions_dir):
    """A row per `.dcmsession` folder directly in `sessions_dir`, by name."""
    rows = []
    for name in sorted(os.listdir(sessions_dir)):
        full = os.path.join(sessions_dir, name)
        if name.endswith(SESSION_FOLDER_SUFFIX) and os.path.isdir(full):
            rows.append(summarize(full))
    return rows


def binary_size(size_bytes):
    """`size_bytes` in base-2 units, as `du -h` shows it."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / 1024 ** 3:.1f} GB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"


def buffer_label(row):
    """What session.json says about the replay buffer, checked against the
    folder: `yes` / `no` when they agree, `MISSING` when session.json says it
    was saved but the file is gone, `UNLISTED` when the file is there but
    session.json says none was saved. A session written before the field
    shows only whether the file is there."""
    declared, present = row["has_replay_buffer"], row["replay_buffer_file"]
    if declared is None:
        return "file" if present else "none"
    if declared and not present:
        return "MISSING"
    if present and not declared:
        return "UNLISTED"
    return "yes" if declared else "no"


# (header, alignment) of every column after the folder name, in order.
COLUMNS = (("size", ">"), ("saved (local)", "<"), ("trigger", "<"), ("trainer -> champion", "<"),
           ("steps", ">"), ("train h", ">"), ("games", ">"), ("arenas", ">"), ("promotions", ">"),
           ("buffer", "<"), ("lineage", "<"))


def lineage_label(row):
    """The lineage column: path kind, segment, cumulative trainer step and
    how the segment began, or `none (v<format>)` before lineage records."""
    lineage = row["lineage"]
    if lineage is None:
        return f'none (v{row["format_version"]})'
    if lineage["exact_resume"]:
        start = "exact"
    elif lineage["not_exact_items"]:
        start = "not-exact:" + ",".join(lineage["not_exact_items"])
    else:
        start = "fresh"
    cum = lineage["cum_trainer_step"]
    return (f'{lineage["path_kind"]} seg={lineage["segment_index"]} '
            f'cum={cum if cum is not None else "null"} {start}')


def cells(row):
    """The cells of `row` after its name, matching `COLUMNS`; a row that could
    not be read has its size (when known) and its error."""
    size = binary_size(row["size_bytes"]) if "size_bytes" in row else "?"
    if "error" in row:
        return [size, "ERROR: " + row["error"]]
    saved = datetime.datetime.fromtimestamp(row["saved_at_unix"]).strftime("%Y-%m-%d %H:%M")
    return [size, saved, row["trigger"], f'{row["trainer_id"]} -> {row["champion_id"]}',
            str(row["training_steps"]), f'{row["elapsed_training_sec"] / 3600.0:.1f}',
            str(row["self_play_games"]), str(row["arenas"]), str(row["promotions"]),
            buffer_label(row), lineage_label(row)]


def render_table(rows):
    """The printed lines: a header, then one line per row, every column
    aligned to its widest cell. An error row's message runs on from its size."""
    table = [["folder"] + [title for title, _ in COLUMNS]]
    table += [[row["name"]] + cells(row) for row in rows]
    # Every line's last cell runs on unpadded, so it never sets a width.
    widths = [max(len(line[i]) for line in table if i < len(line) - 1) for i in range(len(COLUMNS))]
    alignments = ["<"] + [alignment for _, alignment in COLUMNS]
    lines = []
    for line in table:
        padded = [f"{cell:{alignments[i]}{widths[i]}}" for i, cell in enumerate(line[:-1])]
        lines.append("  ".join(padded + [line[-1]]))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tail", type=int, default=20, help="show the last N saves by name (default 20)")
    parser.add_argument("--sessions-dir", default=SESSIONS_DIR, help="the Sessions folder to read")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    if args.tail < 1:
        parser.error("--tail must be at least 1")
    sessions_dir = os.path.expanduser(args.sessions_dir)
    if not os.path.isdir(sessions_dir):
        print(f"not a folder: {sessions_dir}", file=sys.stderr)
        return 2
    rows = collect(sessions_dir)
    if not rows:
        print(f"no .dcmsession folders in {sessions_dir}", file=sys.stderr)
        return 1
    shown = rows[-args.tail:]
    if args.json:
        print(json.dumps(shown, indent=2))
    else:
        for line in render_table(shown):
            print(line)
    return 1 if any("error" in row for row in shown) else 0


if __name__ == "__main__":
    sys.exit(main())
