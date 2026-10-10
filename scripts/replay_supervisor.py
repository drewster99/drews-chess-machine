#!/usr/bin/env python3
"""Run one corpus-replay line to a target trainer step, resuming it exactly
after every GPU-fault stop.

Why this exists: from the first build after commit b1e7ef35, any GPU fault in
a `--replay-corpus` process -- including a command buffer macOS discarded
because *another* process hung the GPU -- stops training with no save after
it, writes results.json (`termination_reason: gpu_fault`) and exits 36. The
recovery is an exact resume of the last checkpoint written before the fault.
This supervisor does that unattended, so a shared-GPU reset costs at most the
steps since the last 1,000-step checkpoint plus a short back-off.

It does not take the app's word alone for "the last save is fault-free". The
app's barrier polls macOS's log before each save, but a message that reaches
the log store a moment after that poll would be missed, and a process killed
outright (or one whose system-log monitor was unavailable) reports nothing.
So the supervisor reads macOS's log itself (`log show`, this child's pid,
`kIOGPUCommandBufferCallbackError…`) and keeps only checkpoints whose save was
on disk before the earliest fault, minus FAULT_MARGIN_SEC. When no fault time
can be read at all, it distrusts the attempt's newest save.

What it never does: delete or overwrite a file; signal the trainer; pass
`--overwrite-out-model` or `--accept-inexact` on a restart (only the line's
first launch arguments, given after `--`, are ever repeated, and only when the
restart is that same start); restart after a refusal (exit 2), a training-
health stop (35), a manual stop, a failure (33) or a signal death without a
GPU fault in macOS's log, or a finished run whose pid macOS logged faults for;
launch, move or create a file while any other running trainer names this
`--out-model`; run a binary whose commit does not contain b1e7ef35.

Checkpoints it stops trusting are moved (rename, same volume) into
  ~/Library/Application Support/DrewsChessMachine/ReplaySupervisor/quarantine/<time>-<label>-a<k>/
with a manifest, because the app's TrainerOutputFileGuard refuses an exact
resume from step N while the stem holds a step file above N, and a rolling
`-latest` file ahead of the start model.

Usage (detached; arguments after `--` are the line's first start, passed
verbatim to the first attempt and to any rerun of that same start):

  nohup python3 scripts/replay_supervisor.py \\
      --bin "<FrozenBuilds/DCM-N-hash.app/Contents/MacOS/DrewsChessMachine>" \\
      --corpus 20260624-192615-w3aA5b --parameters <parameters.json> [--epochs 12] \\
      --out-model "<Models/<base>-replay-latest.safetensors>" --target-trainer-step 100000 \\
      --run-dir <experiment folder> --label <name> \\
      -- --start-model "<Models/<base>-replay-step60000.safetensors>" --resume-exact \\
      >/dev/null 2>&1 &!

  A fresh line: `-- --seed 20261008 [--preset p]`; a branch: `-- --start-model <file> --seed …`.
  `--attach-pid` supervises an already running new-build trainer instead of
  launching attempt 0 (its exit status is then unknown and is read from its
  results.json / stdout / macOS's log). `--dry-run` prints the checks and the
  first command and exits.

Files, in --run-dir: supervisor-<label>.log, and per attempt k
train-<label>-a<k>.stdout and results-<label>-a<k>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import dcm_lineage  # noqa: E402

# `--replay-corpus` exit statuses (CorpusReplayRunner.runAndExit).
EXIT_DONE = 0
EXIT_REFUSED = 2
EXIT_FAILED = 33
EXIT_HEALTH_STOP = 35
EXIT_GPU_FAULT = 36

# results.json `termination_reason` values (CliTrainingRecorder.TerminationReason).
REASON_STEP_LIMIT = "step_limit_reached"
REASON_GPU_FAULT = "gpu_fault"

# What macOS logs in a process whose command buffer was hung, discarded or
# otherwise failed (the GPUFaultMonitor's allowlist).
GPU_FAULT_TEXT = "kIOGPUCommandBufferCallbackError"
GPU_HANG_TEXT = "kIOGPUCommandBufferCallbackErrorHang"
STOPPED_BY_GPU_FAULT_LINE = "[REPLAY] stopped by a GPU fault"

RUN_TAG = "replay"
EXTENSION = ".safetensors"
ROLLING_MARKER = f"-{RUN_TAG}-latest"

# Build 2481 (the newest before this was written) was built from an
# uncommitted tree before b1e7ef35; the first build to freeze is later.
DEFAULT_MIN_BUILD = 2482
# The commit that made a GPU fault stop a CLI run with status 36. A binary's
# commit must descend from it: the build counter is per machine and says
# nothing about which source a build compiled.
GPU_FAULT_STOP_COMMIT = "b1e7ef35"

# A save counts as written before a fault only when its file was complete at
# least this long before macOS's earliest fault message. Both clocks are the
# same wall clock; the margin covers the order of a GPU reset and its log
# messages (milliseconds apart on 2026-10-09) with room to spare. Distrusting
# a good save costs at most 1,000 steps; trusting a bad one costs the run.
FAULT_MARGIN_SEC = 10.0

SUPERVISOR_HOME = Path.home() / "Library/Application Support/DrewsChessMachine/ReplaySupervisor"

# Launch options the supervisor sets itself; they may not appear after `--`.
OWN_OPTIONS = frozenset({"--replay-corpus", "--parameters", "--out-model", "--training-step-limit",
                         "--enumerate-checkpoints", "--output", "--epochs", "--overwrite-out-model",
                         "--overwrite-output"})


class SupervisorError(Exception):
    """Something the supervisor will not decide on its own."""


# ---------- logging ----------

class Journal:
    def __init__(self, path: Path, notify_cmd: Optional[str]):
        self.path = path
        self.notify_cmd = notify_cmd

    def log(self, message: str) -> None:
        line = f"{dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z')} {message}"
        with open(self.path, "a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def notify(self, message: str) -> None:
        """Log, and hand the message to --notify-cmd (one argument, no shell)."""
        self.log(f"NOTIFY {message}")
        if not self.notify_cmd:
            return
        try:
            subprocess.run([self.notify_cmd, message], timeout=60, check=False,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log(f"notify command failed: {error}")


# ---------- checkpoints ----------

@dataclass(frozen=True)
class Checkpoint:
    path: Path
    model_id: str
    trainer_step: int
    run_id: str
    segment_id: str
    mtime: float

    def describe(self) -> str:
        return f"{self.path.name} (model {self.model_id}, trainer step {self.trainer_step})"


def read_checkpoint(path: Path) -> Checkpoint:
    """A trainer file's identity from its header (CLAUDE.md: checkpoints are
    identified by `__metadata__`, never by name)."""
    source = path.name
    try:
        metadata = dcm_lineage.read_metadata(str(path))
        reading = dcm_lineage.step_reading(metadata, source)
        lineage = dcm_lineage.lineage_of(metadata, source)
    except dcm_lineage.LineageError as error:
        raise SupervisorError(f"{path}: {error}") from None
    if reading.basis != dcm_lineage.BASIS_TRAINER_STEP or reading.trainer_step is None:
        raise SupervisorError(f"{path}: no format-v11 trainer step ({reading!r}); only v11+ trainer files are continued")
    if isinstance(lineage, dcm_lineage.Unrecorded):
        raise SupervisorError(f"{path}: no lineage record")
    model_id = metadata.get("model_id")
    if not model_id:
        raise SupervisorError(f"{path}: no model_id")
    return Checkpoint(path=path, model_id=model_id, trainer_step=reading.trainer_step,
                      run_id=lineage["run"]["lineage_run_id"], segment_id=lineage["run"]["segment_id"],
                      mtime=path.stat().st_mtime)


def rolling_stem(out_model: Path) -> str:
    if not out_model.name.endswith(EXTENSION):
        raise SupervisorError(f"--out-model {out_model} does not end in {EXTENSION}")
    return out_model.name[: -len(EXTENSION)]


def enumerated_name(stem: str, step: int) -> str:
    """EnumeratedCheckpointNaming.fileName: `<base>-replay-step<N>` from a
    `<base>-replay-latest` stem, `<stem>-step<N>` from any other."""
    if ROLLING_MARKER in stem:
        return stem.replace(ROLLING_MARKER, f"-{RUN_TAG}-step{step}") + EXTENSION
    return f"{stem}-step{step}{EXTENSION}"


def step_of_enumerated_name(stem: str, name: str) -> Optional[int]:
    """EnumeratedCheckpointNaming.trainerStep(ofFileName:): the step, confirmed
    by rebuilding the name."""
    marker = stem.find(ROLLING_MARKER)
    prefix = (stem[:marker] + f"-{RUN_TAG}-step") if marker >= 0 else f"{stem}-step"
    if not name.startswith(prefix):
        return None
    digits = re.match(r"[0-9]+", name[len(prefix):])
    if digits is None:
        return None
    step = int(digits.group(0))
    return step if enumerated_name(stem, step) == name else None


def stem_files(out_model: Path, above: int, through: int) -> List[Tuple[int, Path]]:
    """The stem's step files with above < step <= through: exactly the range
    TrainerOutputFileGuard.reachableEnumeratedCheckpoints refuses."""
    stem = rolling_stem(out_model)
    found = []
    if not out_model.parent.is_dir():
        return found
    for name in os.listdir(out_model.parent):
        step = step_of_enumerated_name(stem, name)
        if step is not None and above < step <= through:
            found.append((step, out_model.parent / name))
    return sorted(found)


# ---------- faults ----------

@dataclass(frozen=True)
class Fault:
    time: float
    origin: str
    text: str


def _log_time(epoch: float) -> str:
    """Local time, for the journal only."""
    return dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _log_query_time(epoch: float, round_up: bool) -> str:
    """A `log show --start/--end` argument in UTC with an explicit offset: a
    local wall time without one is ambiguous in the autumn DST hour. Whole
    seconds, rounded outward so the query covers the interval; entries are
    filtered by their exact time afterwards."""
    whole = int(epoch) + (1 if round_up and epoch != int(epoch) else 0)
    return dt.datetime.fromtimestamp(whole, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S+0000")


def log_show(predicate: str, start: float, end: float, journal: Journal) -> Optional[List[dict]]:
    """macOS's log entries matching `predicate` with start <= time <= end, or
    None when the query did not complete (never read as "no entries")."""
    command = ["/usr/bin/log", "show", "--style", "ndjson", "--start", _log_query_time(start, round_up=False),
               "--end", _log_query_time(end, round_up=True), "--predicate", predicate]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as error:
        journal.log(f"log show failed: {error}")
        return None
    if completed.returncode != 0:
        journal.log(f"log show exited {completed.returncode}: {completed.stderr.strip()[:300]}")
        return None
    entries, finished = [], False
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            journal.log(f"log show printed a line that is not JSON: {line[:200]}")
            return None
        if "finished" in entry:
            finished = entry.get("finished") == 1
        elif "eventMessage" in entry:
            try:
                entry["_epoch"] = _log_entry_time(entry)
            except (KeyError, TypeError, ValueError) as error:
                journal.log(f"log show entry without a readable timestamp ({error}): {line[:200]}")
                return None
            if start <= entry["_epoch"] <= end:
                entries.append(entry)
    if not finished:
        journal.log("log show did not report a finished query")
        return None
    return entries


def _log_entry_time(entry: dict) -> float:
    """`2026-10-10 03:14:35.747605-0500`: carries its offset, so unambiguous."""
    return dt.datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S.%f%z").timestamp()


def macos_faults(pid: int, start: float, end: float, journal: Journal) -> Optional[List[Fault]]:
    entries = log_show(f'processID == {pid} AND eventMessage CONTAINS "{GPU_FAULT_TEXT}"', start, end, journal)
    if entries is None:
        return None
    return [Fault(e["_epoch"], "macOS log", e["eventMessage"]) for e in entries]


def results_faults(results: Optional[dict]) -> List[Fault]:
    """`gpu_faults.faults` (GPUFaultReport: ISO 8601 with milliseconds, `Z`).
    A time it cannot read stops the supervisor: deciding which saves to keep
    needs the earliest fault, and guessing it is how a bad save gets kept."""
    faults = []
    for fault in ((results or {}).get("gpu_faults") or {}).get("faults") or []:
        try:
            stamp = dt.datetime.fromisoformat(fault["time"].replace("Z", "+00:00")).timestamp()
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise SupervisorError(f"results.json gpu_faults entry with an unreadable time ({error}): {fault!r}") from None
        faults.append(Fault(stamp, f"results ({fault.get('source')})", fault.get("detail", "")))
    return faults


def hang_origins(fault_time: float, journal: Journal) -> str:
    """Which processes macOS logged a GPU hang for around the fault
    (informational: the hang's owner is usually not this process)."""
    entries = log_show(f'eventMessage CONTAINS "{GPU_HANG_TEXT}"', fault_time - 120, fault_time + 5, journal)
    if entries is None:
        return "unknown (log query failed)"
    owners = sorted({f"{Path(e.get('processImagePath', '?')).name}[{e.get('processID')}]" for e in entries})
    return ", ".join(owners) if owners else "none logged"


# ---------- attempts ----------

@dataclass
class Attempt:
    index: int
    start_args: List[str]
    src: Optional[Checkpoint]       # the checkpoint it starts from (None: fresh)
    exact: bool                     # --resume-exact
    start_step: int                 # trainer step it starts at
    stdout: Path
    results: Path
    previous_stdout: Optional[Path] = None  # set when it resumes a faulted attempt's line exactly
    pid: int = 0
    started: float = 0.0
    ended: float = 0.0
    returncode: Optional[int] = None


@dataclass
class Outcome:
    kind: str                       # "done", "fault" or "stop"
    summary: str
    earliest_fault: Optional[float] = None


@dataclass
class RestartPlan:
    start_args: List[str]
    src: Optional[Checkpoint]
    exact: bool
    start_step: int
    progressed: bool
    quarantine: List[Path] = field(default_factory=list)
    complete_enumerated: Optional[Tuple[Path, Path]] = None   # (rolling file, enumerated path to create)
    notes: List[str] = field(default_factory=list)


class Supervisor:
    def __init__(self, cfg: argparse.Namespace, journal: Journal):
        self.cfg = cfg
        self.journal = journal
        self.out_model = Path(cfg.out_model)
        self.run_dir = Path(cfg.run_dir)

    # --- launch arguments ---

    def command(self, attempt: Attempt) -> List[str]:
        limit = self.cfg.target_trainer_step - attempt.start_step
        command = [self.cfg.bin, "--replay-corpus", self.cfg.corpus, "--parameters", self.cfg.parameters,
                   "--out-model", str(self.out_model), "--training-step-limit", str(limit),
                   "--enumerate-checkpoints", "--output", str(attempt.results)]
        if self.cfg.epochs is not None:
            command += ["--epochs", str(self.cfg.epochs)]
        return command + attempt.start_args

    def attempt_paths(self, index: int) -> Tuple[Path, Path]:
        label = self.cfg.label
        return (self.run_dir / f"train-{label}-a{index}.stdout", self.run_dir / f"results-{label}-a{index}.json")

    def next_free_index(self, at_least: int) -> int:
        index = at_least
        while any(p.exists() for p in self.attempt_paths(index)):
            index += 1
        return index

    def make_attempt(self, index: int, start_args: List[str], previous_stdout: Optional[Path] = None) -> Attempt:
        for option in start_args:
            if option in OWN_OPTIONS:
                raise SupervisorError(f"{option} is set by the supervisor; remove it after --")
        src, exact, start_step = None, "--resume-exact" in start_args, 0
        if exact:
            # Only an exact resume continues the start file's trainer clock
            # (a branch starts at trainer step 0, CorpusReplayRunner's
            # segmentStartTrainerStep), so only then is its header needed --
            # and a branch may start from a pre-v11 file this reader refuses.
            if start_args.count("--start-model") != 1:
                raise SupervisorError("--resume-exact needs exactly one --start-model after --")
            position = start_args.index("--start-model")
            if position + 1 >= len(start_args):
                raise SupervisorError("--start-model without a file after --")
            src = read_checkpoint(Path(start_args[position + 1]).expanduser())
            start_step = src.trainer_step
        if start_step >= self.cfg.target_trainer_step:
            raise SupervisorError(f"start trainer step {start_step} is not below the target {self.cfg.target_trainer_step}")
        stdout, results = self.attempt_paths(index)
        return Attempt(index=index, start_args=start_args, src=src, exact=exact, start_step=start_step,
                       stdout=stdout, results=results, previous_stdout=previous_stdout)

    # --- running ---

    def require_no_live_trainer(self, before: str) -> None:
        """Refuse when any running trainer names this stem's rolling file as
        its --out-model: the supervisor's lock covers supervisors only, and a
        trainer started by hand on the same stem would have its files moved
        or raced."""
        others = live_trainers_writing(self.out_model)
        if others:
            raise SupervisorError(f"not {before}: running trainer(s) name {self.out_model.name} as --out-model: "
                                  + "; ".join(f"pid {pid}: {args[:300]}" for pid, args in others))

    def launch_and_wait(self, attempt: Attempt) -> None:
        self.require_no_live_trainer("launching")
        command = self.command(attempt)
        self.journal.log(f"attempt {attempt.index}: launching from trainer step {attempt.start_step} "
                         f"({attempt.src.describe() if attempt.src else 'no start checkpoint'}), "
                         f"limit {self.cfg.target_trainer_step - attempt.start_step}: {command}")
        with open(attempt.stdout, "xb") as stdout:   # never overwrite an attempt's output
            attempt.started = time.time()
            # Its own session: a signal to the supervisor's terminal or process
            # group never reaches the trainer.
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        attempt.pid = process.pid
        self.journal.log(f"attempt {attempt.index}: pid {process.pid}, stdout {attempt.stdout}")
        attempt.returncode = process.wait()
        attempt.ended = time.time()
        self.journal.log(f"attempt {attempt.index}: pid {attempt.pid} exited with status {attempt.returncode} "
                         f"after {(attempt.ended - attempt.started) / 3600:.2f} h")

    def wait_attached(self, attempt: Attempt, pid: int) -> None:
        started = process_start_time(pid)
        if started is None:
            raise SupervisorError(f"pid {pid} is not running")
        # The plan after it exits assumes this process is this line's trainer,
        # started from the given start arguments with the step limit this
        # supervisor would have passed; check what `ps` can show.
        args = process_args(pid)
        # check_binary vetted --bin; the attached process must be that binary.
        if not args.startswith(self.cfg.bin + " "):
            raise SupervisorError(f"pid {pid} does not run --bin {self.cfg.bin}: {args[:300]}")
        limit = re.search(r"--training-step-limit (\d+)(?:\s|$)", args)
        missing = [token for token in ("--replay-corpus", "--enumerate-checkpoints", "--out-model",
                                       self.out_model.name, "--output", Path(self.cfg.attach_results).name)
                   if token not in args]
        if missing or limit is None or int(limit.group(1)) != self.cfg.target_trainer_step - attempt.start_step:
            raise SupervisorError(f"pid {pid} is not this line's trainer as started by the supervisor's rules "
                                  f"(missing {missing}, --training-step-limit "
                                  f"{limit.group(1) if limit else 'absent'}, expected "
                                  f"{self.cfg.target_trainer_step - attempt.start_step}): {args[:400]}")
        attempt.pid, attempt.started = pid, started
        self.journal.log(f"attempt {attempt.index}: attached to pid {pid} (started {_log_time(started)})")
        while process_start_time(pid) == started:   # a reused pid has another start time
            time.sleep(30)
        attempt.ended = time.time()
        self.journal.log(f"attempt {attempt.index}: pid {pid} exited (status unknown: not this supervisor's child)")

    # --- after an attempt ---

    def classify(self, attempt: Attempt) -> Outcome:
        results = load_json(attempt.results, self.journal)
        reason = (results or {}).get("termination_reason")
        tail = read_tail(attempt.stdout)
        stdout_fault = STOPPED_BY_GPU_FAULT_LINE in tail
        # From the child's start, not before: the pid was another process's
        # until then. Up to 2 s after its exit for messages logged late.
        macos = macos_faults(attempt.pid, attempt.started, attempt.ended + 2, self.journal)
        reported = results_faults(results)
        faults = (macos or []) + reported
        earliest = min((f.time for f in faults), default=None)
        monitor = ((results or {}).get("gpu_faults") or {}).get("monitor")
        status = attempt.returncode
        summary = (f"status={status} termination_reason={reason} stdout_fault_line={stdout_fault} "
                   f"macos_faults={'unreadable' if macos is None else len(macos)} "
                   f"results_faults={len(reported)} monitor={monitor} "
                   f"earliest_fault={_log_time(earliest) if earliest else None}")
        self.journal.log(f"attempt {attempt.index}: {summary}")
        for fault in sorted(faults, key=lambda f: f.time)[:5]:
            self.journal.log(f"  fault {_log_time(fault.time)} {fault.origin}: {fault.text[:200]}")
        if earliest is not None:
            self.journal.log(f"  GPU hang logged around the earliest fault by: {hang_origins(earliest, self.journal)}")

        said_fault = status == EXIT_GPU_FAULT or reason == REASON_GPU_FAULT or stdout_fault
        if said_fault:
            if status not in (EXIT_GPU_FAULT, None) or (results is not None and reason != REASON_GPU_FAULT):
                self.journal.log("  the fault signals disagree (exit status, results, stdout); treated as a GPU fault")
            return Outcome("fault", summary, earliest)
        if status == EXIT_DONE or (status is None and reason is not None):
            if macos:
                return Outcome("stop", f"the run ended normally but macOS logged {len(macos)} GPU fault(s) for "
                                       f"pid {attempt.pid} (monitor={monitor}); its saves after "
                                       f"{_log_time(earliest)} may be suspect. Check before using them. {summary}")
            if macos is None:
                self.journal.log("  macOS's log could not be read, so the run's own fault report stands alone")
            if reason == REASON_STEP_LIMIT:
                return Outcome("done", summary)
            return Outcome("stop", f"ended with termination_reason={reason}. {summary}")
        if status == EXIT_HEALTH_STOP:
            return Outcome("stop", f"training-health stop. {summary}")
        if status == EXIT_REFUSED:
            return Outcome("stop", f"refused: {last_error_line(tail)}. {summary}")
        if macos:
            # Exit 33 or a signal death (a fault at start-up, during the
            # load, or a framework abort) with macOS fault messages for it.
            return Outcome("fault", summary, earliest)
        if macos is None:
            return Outcome("stop", f"failed and macOS's log could not be read to tell whether a GPU fault caused it: "
                                   f"{last_error_line(tail)}. {summary}")
        return Outcome("stop", f"failed without a GPU fault: {last_error_line(tail)}. {summary}")

    def compare_batch_hashes(self, attempt: Attempt) -> Optional[bool]:
        """Whether the resumed attempt trained on the faulted attempt's batches
        at every common trainer step (exact corpus-replay resumes do)."""
        if attempt.previous_stdout is None:
            return None
        completed = subprocess.run([sys.executable, str(REPO / "scripts/compare_batch_hashes.py"),
                                    str(attempt.previous_stdout), str(attempt.stdout)],
                                   capture_output=True, text=True, timeout=300)
        verdict = (completed.stdout.strip().splitlines() or ["(no output)"])[-1]
        self.journal.log(f"attempt {attempt.index}: batch hashes vs attempt {attempt.index - 1}: "
                         f"exit {completed.returncode}: {verdict}")
        if completed.returncode == 1:
            return False
        return True if completed.returncode == 0 else None

    def plan_restart(self, attempt: Attempt, earliest_fault: Optional[float]) -> RestartPlan:
        target = self.cfg.target_trainer_step
        stem = rolling_stem(self.out_model)

        # The attempt's enumerated saves. The launch guard refused the start if
        # any file was in this range, so every one is this attempt's.
        enumerated: Dict[int, Checkpoint] = {}
        for step, path in stem_files(self.out_model, attempt.start_step, target):
            checkpoint = read_checkpoint(path)
            if checkpoint.trainer_step != step:
                raise SupervisorError(f"{path.name} is named for step {step} but holds trainer step "
                                      f"{checkpoint.trainer_step}")
            if checkpoint.mtime < attempt.started:
                raise SupervisorError(f"{path.name} predates attempt {attempt.index}; the launch guard should have "
                                      f"refused that start")
            enumerated[step] = checkpoint
        writers = {(c.model_id, c.run_id) for c in enumerated.values()}
        if len(writers) > 1:
            raise SupervisorError(f"the attempt's step files have more than one writer: {sorted(writers)}")
        if attempt.exact and attempt.src and enumerated and next(iter(writers))[1] != attempt.src.run_id:
            raise SupervisorError("the attempt's step files are not in its start model's lineage run")

        latest = read_checkpoint(self.out_model) if self.out_model.exists() else None
        latest_ours = latest is not None and latest.mtime >= attempt.started
        if latest_ours:
            if writers and (latest.model_id, latest.run_id) not in writers:
                raise SupervisorError(f"{latest.describe()} was written during the attempt by another writer")
            if latest.trainer_step <= attempt.start_step or latest.trainer_step > target:
                raise SupervisorError(f"{latest.describe()} is outside the attempt's range")
            if attempt.exact and attempt.src and latest.run_id != attempt.src.run_id:
                raise SupervisorError(f"{latest.describe()} is not in the start model's lineage run")

        # One save = one encoded payload, written to the rolling file and then
        # to its step file; it was complete when the first of the two was.
        save_time: Dict[int, float] = {step: c.mtime for step, c in enumerated.items()}
        if latest_ours:
            save_time[latest.trainer_step] = min(save_time.get(latest.trainer_step, latest.mtime), latest.mtime)
        steps = sorted(save_time)
        plan_notes = []
        if earliest_fault is None:
            trusted = steps[:-1]
            plan_notes.append("no fault time could be read: the attempt's newest save is not trusted")
        else:
            cutoff = earliest_fault - FAULT_MARGIN_SEC
            trusted = [s for s in steps if save_time[s] < cutoff]
            if trusted != steps[:len(trusted)]:
                raise SupervisorError("the attempt's save times do not increase with step")
            for s in steps[len(trusted):]:
                plan_notes.append(f"save at trainer step {s} ({_log_time(save_time[s])}) is not trusted: not on disk "
                                  f"{FAULT_MARGIN_SEC:.0f} s before the earliest fault ({_log_time(earliest_fault)})")

        if trusted:
            step = trusted[-1]
            completion = None
            if step in enumerated:
                src = enumerated[step]
            else:
                # The rolling save landed but its step-file copy failed (the
                # app tolerates one such failure). Write the copy it would
                # have written: the same bytes, under the step file's name.
                dest = self.out_model.parent / enumerated_name(stem, step)
                completion = (self.out_model, dest)
                src = latest
                plan_notes.append(f"step file for trainer step {step} is missing; it is recreated from {self.out_model.name}")
            plan = RestartPlan(start_args=["--start-model", str(src.path if completion is None else completion[1]),
                                           "--resume-exact"],
                               src=src, exact=True, start_step=step, progressed=True,
                               complete_enumerated=completion, notes=plan_notes)
        else:
            # No save of this attempt is kept: rerun the attempt's own start
            # (its arguments, including any --accept-inexact or --seed).
            plan = RestartPlan(start_args=list(attempt.start_args), src=attempt.src, exact=attempt.exact,
                               start_step=attempt.start_step, progressed=False, notes=plan_notes)
            if not attempt.exact:
                plan.notes.append("the line's first start is rerun: a fresh or branch start draws a new seed unless "
                                  "--seed or the parameters file fixes it")

        plan.quarantine = [c.path for s, c in sorted(enumerated.items()) if s > plan.start_step]
        if latest_ours and latest.trainer_step > plan.start_step:
            plan.quarantine.append(self.out_model)
        if latest is not None and not latest_ours:
            # The rolling file predates the attempt: it must be the start's own state.
            if not (plan.exact and plan.src and latest.model_id == plan.src.model_id
                    and latest.trainer_step == plan.src.trainer_step):
                raise SupervisorError(f"{latest.describe()} predates the attempt and is not the restart's start state")
        return plan

    def apply(self, plan: RestartPlan, attempt: Attempt, earliest_fault: Optional[float]) -> None:
        self.require_no_live_trainer("moving or creating checkpoint files")
        if plan.quarantine:
            quarantine_root = SUPERVISOR_HOME / "quarantine"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            # A rename across volumes fails (EXDEV) part-way through a list;
            # check every source first so nothing is moved unless all can be.
            volume = os.stat(quarantine_root).st_dev
            elsewhere = [str(p) for p in plan.quarantine if os.stat(p).st_dev != volume]
            if elsewhere:
                raise SupervisorError(f"not on the quarantine folder's volume ({quarantine_root}), so they cannot "
                                      f"be moved by rename: {elsewhere}")
            folder = quarantine_root / (
                f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.cfg.label}-a{attempt.index}")
            folder.mkdir(exist_ok=False)
            # The manifest goes first, so a move that fails part-way leaves the
            # plan beside whatever was moved (each move is also journalled).
            manifest = {"label": self.cfg.label, "attempt": attempt.index, "pid": attempt.pid,
                        "earliest_fault": _log_time(earliest_fault) if earliest_fault else None,
                        "fault_margin_sec": FAULT_MARGIN_SEC, "restart_start_step": plan.start_step,
                        "notes": plan.notes,
                        "moves": [{"from": str(p), "to": str(folder / p.name)} for p in plan.quarantine]}
            with open(folder / "manifest.json", "x") as handle:
                json.dump(manifest, handle, indent=2)
            for path in plan.quarantine:
                destination = folder / path.name
                if os.path.lexists(destination):
                    raise SupervisorError(f"{destination} exists")
                os.rename(path, destination)
                self.journal.log(f"  quarantined {path.name} -> {destination}")
        if plan.complete_enumerated:
            rolling, destination = plan.complete_enumerated
            publish_copy(rolling, destination)
            copy = read_checkpoint(destination)
            if (copy.model_id, copy.trainer_step) != (plan.src.model_id, plan.src.trainer_step):
                raise SupervisorError(f"{destination.name} does not hold {plan.src.describe()}")
            plan.src = copy
            self.journal.log(f"  recreated {destination.name} from {rolling.name}")
        # The app's two launch guards, checked here so a mistake stops the
        # supervisor rather than becoming an exit-2 restart loop.
        left = stem_files(self.out_model, plan.start_step, self.cfg.target_trainer_step)
        if left:
            raise SupervisorError(f"step files remain above the restart step: {[p.name for _, p in left]}")
        if self.out_model.exists():
            latest = read_checkpoint(self.out_model)
            if not (plan.exact and plan.src and (latest.model_id, latest.trainer_step)
                    == (plan.src.model_id, plan.src.trainer_step)):
                raise SupervisorError(f"{latest.describe()} would be refused as the rolling file of this restart")

    def verify_finished(self) -> str:
        target = self.cfg.target_trainer_step
        final_path = self.out_model.parent / enumerated_name(rolling_stem(self.out_model), target)
        if not final_path.exists():
            latest = read_checkpoint(self.out_model) if self.out_model.exists() else None
            raise SupervisorError(f"no step file at the target {target}; rolling file: "
                                  f"{latest.describe() if latest else 'none'} (epoch budget or corpus end?)")
        final = read_checkpoint(final_path)
        latest = read_checkpoint(self.out_model)
        if final.trainer_step != target or (latest.model_id, latest.trainer_step) != (final.model_id, target):
            raise SupervisorError(f"final files disagree: {final.describe()} vs {latest.describe()}")
        return final.describe()

    # --- the loop ---

    def run(self, start_args: List[str]) -> int:
        check_binary(self.cfg, self.journal)
        attempt = self.make_attempt(self.next_free_index(0), start_args)
        self.journal.log(f"supervising {self.out_model.name} to trainer step {self.cfg.target_trainer_step}")
        if self.cfg.dry_run:
            self.journal.log(f"dry run: attempt {attempt.index} would run {self.command(attempt)}")
            self.journal.log(f"dry run: step files in the guard's range: "
                             f"{[p.name for _, p in stem_files(self.out_model, attempt.start_step, self.cfg.target_trainer_step)]}")
            if self.out_model.exists():
                self.journal.log(f"dry run: rolling file {read_checkpoint(self.out_model).describe()}")
            self.journal.log(f"dry run: running trainers naming this --out-model (a launch refuses while any "
                             f"run): {[pid for pid, _ in live_trainers_writing(self.out_model)]}")
            return 0
        if self.cfg.attach_pid:
            attempt.stdout, attempt.results = Path(self.cfg.attach_stdout), Path(self.cfg.attach_results)
            self.wait_attached(attempt, self.cfg.attach_pid)
        else:
            require_free_space(self.out_model.parent, self.cfg.min_free_gb)
            self.launch_and_wait(attempt)

        restarts, without_progress = 0, 0
        while True:
            outcome = self.classify(attempt)
            hashes_match = self.compare_batch_hashes(attempt)
            if hashes_match is False:
                self.journal.notify(f"{self.cfg.label}: attempt {attempt.index} did not train on the same batches as "
                                    f"attempt {attempt.index - 1}; the exact resume is not exact. Not restarting again.")
                return 1
            if outcome.kind == "done":
                self.journal.notify(f"{self.cfg.label}: finished at {self.verify_finished()} after {restarts} GPU-fault restart(s)")
                return 0
            if outcome.kind == "stop":
                self.journal.notify(f"{self.cfg.label}: attempt {attempt.index} stopped; not restarting: {outcome.summary}")
                return 1

            plan = self.plan_restart(attempt, outcome.earliest_fault)
            for note in plan.notes:
                self.journal.log(f"  {note}")
            without_progress = 0 if plan.progressed else without_progress + 1
            if restarts >= self.cfg.max_restarts:
                self.journal.notify(f"{self.cfg.label}: GPU fault; {restarts} restarts already, the cap. Last kept "
                                    f"trainer step {plan.start_step}. Not restarting.")
                return 1
            if without_progress >= self.cfg.max_restarts_without_progress:
                self.journal.notify(f"{self.cfg.label}: {without_progress} GPU-fault stops in a row without a kept "
                                    f"checkpoint (trainer step {plan.start_step}). Not restarting.")
                return 1
            self.apply(plan, attempt, outcome.earliest_fault)
            delay = min(self.cfg.backoff_sec * (2 ** without_progress), self.cfg.backoff_max_sec)
            self.journal.log(f"restart {restarts + 1}: from trainer step {plan.start_step} "
                             f"({'kept a checkpoint' if plan.progressed else 'no checkpoint kept'}) after {delay:.0f} s")
            time.sleep(delay)
            require_free_space(self.out_model.parent, self.cfg.min_free_gb)
            resumes_same_line = plan.exact
            attempt = self.make_attempt(self.next_free_index(attempt.index + 1), plan.start_args,
                                        previous_stdout=attempt.stdout if resumes_same_line else None)
            restarts += 1
            self.journal.notify(f"{self.cfg.label}: GPU fault stop; restart {restarts} (attempt {attempt.index}) "
                                f"from trainer step {attempt.start_step}")
            self.launch_and_wait(attempt)


# ---------- helpers ----------

def publish_copy(source: Path, destination: Path) -> None:
    """Copy to a hidden staging sibling, then hard-link it to `destination`
    (fails if anything is there: never overwrites), then drop the staging name."""
    # FileSafety's hidden staging name (`.<name>.<UUID>.tmp`, the UUID as
    # Swift's `uuidString` spells it: upper case), so a copy left by a crash
    # is debris the app's launch sweep recognizes.
    staging = destination.parent / f".{destination.name}.{str(uuid.uuid4()).upper()}.tmp"
    with open(source, "rb") as reader, open(staging, "xb") as writer:
        shutil.copyfileobj(reader, writer, 16 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    try:
        os.link(staging, destination)
    except FileExistsError:
        raise SupervisorError(f"{destination} appeared while it was being recreated; left untouched") from None
    finally:
        os.unlink(staging)   # the supervisor's own staging name, created exclusively above


def process_start_time(pid: int) -> Optional[float]:
    completed = subprocess.run(["/bin/ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                               env={**os.environ, "LC_ALL": "C"})
    text = completed.stdout.strip()
    if completed.returncode != 0 or not text:
        return None
    return dt.datetime.strptime(text, "%a %b %d %H:%M:%S %Y").timestamp()


def process_args(pid: int) -> str:
    completed = subprocess.run(["/bin/ps", "-ww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True)
    if completed.returncode != 0:
        raise SupervisorError(f"ps could not read pid {pid}'s arguments: {completed.stderr.strip()}")
    return completed.stdout.strip()


def live_trainers_writing(out_model: Path) -> List[Tuple[int, str]]:
    """Running processes whose arguments name `out_model` (by file name, so
    any spelling of its folder matches) beside `--out-model` and a training
    mode. Readers such as a probe loop pass no `--out-model` and do not count."""
    completed = subprocess.run(["/bin/ps", "-axww", "-o", "pid=,args="], capture_output=True, text=True)
    if completed.returncode != 0:
        raise SupervisorError(f"ps -ax failed: {completed.stderr.strip()}")
    found = []
    for line in completed.stdout.splitlines():
        pid_text, _, args = line.strip().partition(" ")
        if not pid_text.isdigit() or int(pid_text) == os.getpid():
            continue
        if "--out-model" in args and out_model.name in args and ("--replay-corpus" in args or "--train-vs-uci" in args):
            found.append((int(pid_text), args))
    return found


def load_json(path: Path, journal: Journal) -> Optional[dict]:
    if not path.exists():
        journal.log(f"  {path.name}: not written")
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        journal.log(f"  {path.name}: unreadable: {error}")
        return None


def read_tail(path: Path, size: int = 512 * 1024) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as handle:
        handle.seek(max(0, path.stat().st_size - size))
        return handle.read().decode("utf-8", errors="replace")


def last_error_line(tail: str) -> str:
    lines = [l for l in tail.splitlines() if l.startswith("error:") or "refused" in l or "failed" in l]
    return lines[-1].strip()[:400] if lines else "(no error line in stdout)"


def require_free_space(folder: Path, minimum_gb: float) -> None:
    free = shutil.disk_usage(folder).free
    if free < minimum_gb * 1024 ** 3:
        raise SupervisorError(f"{free / 1024 ** 3:.1f} GB free on {folder}'s volume, below {minimum_gb:.0f} GB")


def check_binary(cfg: argparse.Namespace, journal: Journal) -> None:
    """Refuse a build without the exit-36 behavior, or an unrecorded one."""
    completed = subprocess.run([cfg.bin, "--version"], capture_output=True, text=True, timeout=60)
    line = completed.stdout.strip()
    match = re.match(r"DrewsChessMachine build (\d+) git=([0-9a-f]+)(\*?) branch=\S+ configuration=(\S+)", line)
    if completed.returncode != 0 or match is None:
        raise SupervisorError(f"{cfg.bin} --version: status {completed.returncode}, {line or completed.stderr.strip()}")
    build, git_hash, dirty, configuration = int(match.group(1)), match.group(2), match.group(3) == "*", match.group(4)
    journal.log(f"binary: {line}")
    if build < cfg.min_build:
        raise SupervisorError(f"build {build} predates GPU-fault stops (needs {cfg.min_build}+): a fault would not end it with status 36")
    ancestry = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", GPU_FAULT_STOP_COMMIT, git_hash],
                              capture_output=True, text=True, timeout=60)
    if ancestry.returncode == 1:
        raise SupervisorError(f"build {build}'s commit {git_hash} does not contain {GPU_FAULT_STOP_COMMIT} "
                              f"(GPU-fault stops with status 36)")
    if ancestry.returncode != 0:
        raise SupervisorError(f"cannot tell whether {git_hash} contains {GPU_FAULT_STOP_COMMIT}: "
                              f"git exited {ancestry.returncode}: {ancestry.stderr.strip()[:300]}")
    if dirty and not cfg.allow_dirty_build:
        raise SupervisorError(f"build {build} ({git_hash}*) was built from an uncommitted tree")
    if configuration != "Release":
        journal.log(f"warning: {configuration} build")


def main(argv: List[str]) -> int:
    own, start_args = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bin", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--target-trainer-step", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-restarts", type=int, default=24)
    parser.add_argument("--max-restarts-without-progress", type=int, default=3)
    parser.add_argument("--backoff-sec", type=float, default=120.0)
    parser.add_argument("--backoff-max-sec", type=float, default=1800.0)
    parser.add_argument("--min-build", type=int, default=DEFAULT_MIN_BUILD)
    parser.add_argument("--allow-dirty-build", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--notify-cmd")
    parser.add_argument("--attach-pid", type=int)
    parser.add_argument("--attach-stdout")
    parser.add_argument("--attach-results")
    parser.add_argument("--dry-run", action="store_true")
    cfg = parser.parse_args(own)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cfg.label):
        parser.error("--label: letters, digits, '.', '_' and '-' only")
    if cfg.attach_pid and not (cfg.attach_stdout and cfg.attach_results):
        parser.error("--attach-pid needs --attach-stdout and --attach-results")
    # Absolute paths: they are passed to the app, compared with ps output and
    # written to quarantine manifests. Not resolved through symbolic links:
    # the app is handed the path as the operator spelled it.
    cfg.bin = os.path.abspath(os.path.expanduser(cfg.bin))
    cfg.out_model = os.path.abspath(os.path.expanduser(cfg.out_model))
    cfg.run_dir = os.path.abspath(os.path.expanduser(cfg.run_dir))
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_dir():
        parser.error(f"--run-dir {run_dir} is not a folder")

    journal = Journal(run_dir / f"supervisor-{cfg.label}.log", cfg.notify_cmd)
    SUPERVISOR_HOME.mkdir(parents=True, exist_ok=True)
    lock_path = SUPERVISOR_HOME / f"{Path(cfg.out_model).name}.lock"
    lock = open(lock_path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        journal.log(f"another supervisor holds {lock_path}; exiting")
        return 1
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def on_term(signum, _frame):
        journal.log(f"supervisor received signal {signum}; exiting. A running trainer continues, unsupervised.")
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_term)
    try:
        return Supervisor(cfg, journal).run(start_args)
    except SupervisorError as error:
        journal.notify(f"{cfg.label}: supervisor stopped: {error}")
        return 1
    except Exception as error:   # an unexpected failure is reported, never swallowed
        journal.log(traceback.format_exc())
        journal.notify(f"{cfg.label}: supervisor stopped on an unexpected {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
