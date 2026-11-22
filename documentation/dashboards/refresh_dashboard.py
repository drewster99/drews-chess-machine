#!/usr/bin/env python3
"""Refresh the DCM Master Dashboard: the one command that brings every live run's
rows up to date, re-renders the page and updates the repo copy of the artifact.

    DCM_BIN=<app binary> /usr/bin/python3 documentation/dashboards/refresh_dashboard.py [--all] [--runs a,b]

Steps, in order:

1. Corpus replay (`registry.json`): for each target run, `replay.probe_backfill`
   (probe every saved checkpoint whose row has no pElo) and then
   `replay.log_backfill` (per-1000-step log rows, and the writer's logged test-set
   result for saves whose file is gone). Targets are the runs whose rolling
   out-model a live `--replay-corpus` process is writing (its latest segment's log
   only); `--all` takes every run and every segment, `--runs` names them.
2. `replay.py derive-registry` (read-only): a live run that started a new segment
   (a resume) shows as NEW SEGMENT; add it to the run in `registry.json` (its log
   and label) and refresh again. Nothing is written to the registry here.
3. GUI self-play (`selfplay_registry.json`): the probe marks of every run marked
   `probe_from_log` are appended from its logs (`selfplay_probe_append.py`,
   append-only), then every self-play CSV is rebuilt (`selfplay.py`).
4. `master.py` renders `dcm_master.html` (repo root), `make_hosted.py` strips it to
   the artifact page body, and that body is copied to
   `documentation/artifacts/dcm-master-dashboard.html`: the file to publish to
   https://claude.ai/artifact/LfMS23HF3QEVo1r2N4GYaU.

It holds the cron tick's lock (`/tmp/dcm_cron_tick.lock`, owner record
"<pid> <start time>", the protocol of `cron_tick.sh`) while it writes CSVs, so a
tick and a refresh never rewrite the same CSV at once; it waits for a tick that
holds the lock. A checkpoint that cannot be probed does not stop the refresh: every
step still runs, the failures are printed at the end and the exit status is 1.
Probing needs the app binary (`DCM_BIN`); describing files needs `DCM_BINARY` or a
DerivedData build (set both to the same frozen build).
"""
import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import replay  # noqa: E402

LOCK = "/tmp/dcm_cron_tick.lock"
LOCK_OWNER = os.path.join(LOCK, "owner")
# How long to wait for a tick holding the lock before giving up (a tick that
# probes a batch of checkpoints can take minutes).
LOCK_WAIT_SECONDS = 1800
LOCK_POLL_SECONDS = 10
RENDERED = os.path.join(REPO, "dcm_master.html")
HOSTED = os.path.join(HERE, "dcm_master_hosted.html")
ARTIFACT_COPY = os.path.join(REPO, "documentation", "artifacts", "dcm-master-dashboard.html")


def process_identity(pid):
    """'<pid> <start time>' byte for byte as cron_tick.sh writes it: `ps -o lstart=` pads
    the time with trailing spaces and the shell's `$(…)` drops only the trailing
    newlines, so only those are dropped here. Any other difference makes the tick judge
    a live refresh's lock stale and break it."""
    started = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                             check=True).stdout.rstrip("\n")
    return f"{pid} {started}"


@contextlib.contextmanager
def tick_lock():
    """Hold the cron tick's lock directory for the duration. A stale lock is left to
    cron_tick.sh to judge and break (it alone records why); this waits, then fails."""
    me = process_identity(os.getpid())
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise SystemExit(f"{LOCK} is still held after {LOCK_WAIT_SECONDS}s "
                                 f"(owner: {open(LOCK_OWNER).read().strip() if os.path.exists(LOCK_OWNER) else 'none recorded'})")
            time.sleep(LOCK_POLL_SECONDS)
    try:
        with open(LOCK_OWNER, "w") as handle:
            handle.write(me + "\n")
        yield
    finally:
        try:
            with open(LOCK_OWNER) as handle:
                mine = handle.read().rstrip("\n") == me
        except OSError:
            mine = False
        if mine:
            os.remove(LOCK_OWNER)
            os.rmdir(LOCK)


def live_replay_runs():
    """Registry runs whose rolling out-model a live --replay-corpus process writes.
    A failing `ps` raises: "could not look" is not "nothing is training"."""
    args = subprocess.run(["ps", "-Ao", "args="], capture_output=True, text=True, check=True).stdout
    lines = [line for line in args.splitlines() if "--replay-corpus" in line and "--out-model" in line]
    return [run for run, cfg in replay.REG["runs"].items()
            if cfg.get("out_model") and any(cfg["out_model"] in line for line in lines)]


def refresh_replay(runs, every_segment, failures):
    for run in runs:
        print(f"== {run}")
        try:
            replay.probe_backfill(run, verbose=False)
        except (replay.ProbeFailure, replay.BackfillIncomplete) as error:
            failures.append(f"{run} probe-backfill: {error}")
        segments = None if every_segment else [len(replay.REG["runs"][run]["segments"]) - 1]
        replay.log_backfill(run, segments=segments)


def check_registry():
    """derive-registry, read-only; its report is printed as is."""
    status = subprocess.run([sys.executable, os.path.join(HERE, "replay.py"), "derive-registry"],
                            capture_output=True, text=True)
    for line in status.stdout.splitlines():
        if not line.startswith(("  + ", "  excluded ", "  carried copy")):
            print(line)
    return status.returncode


def refresh_selfplay(failures):
    import json
    reg = json.load(open(os.path.join(HERE, "selfplay_registry.json")))
    for run, cfg in reg["runs"].items():
        if not cfg.get("probe_from_log"):
            continue
        for segment in range(len(cfg["logs"])):
            done = subprocess.run([sys.executable, os.path.join(HERE, "selfplay_probe_append.py"), run,
                                   "--segment", str(segment)], capture_output=True, text=True)
            if done.returncode != 0:
                failures.append(f"{run} probe marks, segment {segment}: {done.stderr.strip() or done.stdout.strip()}")
    done = subprocess.run([sys.executable, os.path.join(HERE, "selfplay.py")], capture_output=True, text=True)
    print(done.stdout.rstrip())
    if done.returncode != 0:
        failures.append(f"selfplay.py: {done.stderr.strip()[-2000:]}")


def render():
    for script in ("master.py", "make_hosted.py"):
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
    shutil.copyfile(HOSTED, ARTIFACT_COPY)
    print(f"copied {HOSTED} -> {ARTIFACT_COPY} (publish this file)")


def main():
    ap = argparse.ArgumentParser(description="Refresh the DCM Master Dashboard and its artifact copy.")
    ap.add_argument("--all", action="store_true", help="every replay run, every segment (not only the live runs)")
    ap.add_argument("--runs", help="comma-separated replay runs to refresh (every segment)")
    ap.add_argument("--no-render", action="store_true", help="update the CSVs only")
    a = ap.parse_args()
    failures = []
    if a.runs:
        runs = a.runs.split(",")
        unknown = [r for r in runs if r not in replay.REG["runs"]]
        if unknown:
            raise SystemExit(f"unknown replay runs: {unknown}")
    elif a.all:
        runs = list(replay.REG["runs"])
    else:
        runs = live_replay_runs()
        print(f"live replay runs: {runs or 'none'}")
    with tick_lock():
        refresh_replay(runs, a.all or bool(a.runs), failures)
        refresh_selfplay(failures)
    registry_status = check_registry()
    if not a.no_render:
        render()
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    if registry_status != 0:
        print("derive-registry needs attention (see above)", file=sys.stderr)
    return 1 if failures or registry_status != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
