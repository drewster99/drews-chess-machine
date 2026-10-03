"""The app's session-log files, in launch order.

`SessionLogger` names each launch's log `dcm_log_YYYYMMDD-HHMMSS.txt`, and a second
launch in the same second `dcm_log_YYYYMMDD-HHMMSS-2.txt`, `-3`, … . Sorting those
names as strings puts `-2` before the unsuffixed first launch ('-' sorts before '.')
and `-10` before `-2`, so every script that wants "the latest log" or "logs in order"
sorts with `session_log_sort_key` instead. A file that matches the glob but not the
naming scheme is an error, never silently ordered somewhere.

The stamp is local time, so two launches in the repeated hour at the end of daylight
saving time can still sort in the wrong order; the name alone cannot tell them apart.
"""
import datetime
import glob
import os
import re
from pathlib import Path

LOG_DIRECTORY = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
LOG_GLOB = os.path.join(LOG_DIRECTORY, "dcm_log_*.txt")
NAME_RE = re.compile(r"dcm_log_(\d{8})-(\d{6})(?:-(\d+))?\.txt")


def session_log_sort_key(path):
    """(date, time, launch number within that second) for a session-log path."""
    match = NAME_RE.fullmatch(os.path.basename(str(path)))
    if not match:
        raise ValueError(f"{path}: not a session-log name (dcm_log_YYYYMMDD-HHMMSS[-N].txt)")
    return match.group(1), match.group(2), int(match.group(3)) if match.group(3) else 1


def session_logs():
    """Every session log in the log folder, oldest launch first."""
    return sorted((Path(p) for p in glob.glob(LOG_GLOB)), key=session_log_sort_key)


def latest_session_log():
    """The most recent launch's log, or None when there is none."""
    logs = session_logs()
    return logs[-1] if logs else None


def session_base_date(path):
    """The calendar date a log's launch started on, from its name."""
    date_text = session_log_sort_key(path)[0]
    return datetime.datetime.strptime(date_text, "%Y%m%d").date()
