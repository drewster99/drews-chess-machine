#!/usr/bin/env python3
"""Turn one `--probe-model` run into a probes.jsonl record, identified by the checkpoint itself.

Usage (probe_loop.sh pipes the probe's stdout in):

    probe_record.py <checkpoint.safetensors> <step from the file name> <probes.jsonl> <probe build>

The record's identity comes from the checkpoint's safetensors `__metadata__`
(`model_id`, `training_step`), never from the file name: the step in the name must
equal the header's `training_step`, the probe's `modelID` must equal the header's
`model_id`, and every record already in <probes.jsonl> must carry that same
`model_id` (one probes file = one run; a resumed segment mints a new model_id and
needs its own file). On success it prints one JSON line, `"step"` first, for the
caller to append, carrying `probe_build` (scripts/dcm_probe_build.py) so measurements
from different builds are never mistaken for comparable ones. Nothing is written here.

Exit status (the constants below; probe_loop.sh relies on them):
  success            record printed
  usage error        wrong arguments
  EXIT_PROBE_FAILED  the probe failed (an error event, or not exactly one summary line) -- retryable
  EXIT_IDENTITY      identity mismatch (header vs file name vs probe) -- not retryable
  EXIT_OTHER_RUN     <probes.jsonl> already holds a different model_id -- not retryable

A summary without `pElo` is the probe reporting a non-finite value (it omits the
key then); that is a valid measurement, recorded as `"pElo": null`, not a failure.

`load_probe_points` is the reader the experiment table scripts share.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import dcm_arch  # noqa: E402
from dcm_probe_build import UNRECORDED  # noqa: E402

EXIT_PROBE_FAILED = 3
EXIT_IDENTITY = 4
EXIT_OTHER_RUN = 5


def record_model_id(record):
    """The model ID a probes.jsonl record names (new records carry both keys, older ones only `modelID`)."""
    if "model_id" in record:
        return record["model_id"]
    if "modelID" in record:
        return record["modelID"]
    raise KeyError(f"probe record for step {record.get('step')} names no model ID")


def existing_model_ids(probes_path):
    ids = set()
    if not os.path.exists(probes_path):
        return ids
    with open(probes_path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                # An append still in flight (or cut short); the next pass sees it whole.
                continue
            ids.add(record_model_id(json.loads(line)))
    return ids


def build_record(checkpoint_path, expected_step, probe_stdout, probes_path, probe_build):
    """(exit status, record or message) for one probe of one checkpoint."""
    events = []
    for line in probe_stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            events.append(json.loads(line))
    errors = [e for e in events if e.get("event") == "error"]
    if errors:
        return EXIT_PROBE_FAILED, f"probe reported an error: {errors[0].get('error')}"
    summaries = [e for e in events if "set" in e and "event" not in e]
    if len(summaries) != 1:
        return EXIT_PROBE_FAILED, f"expected one probe summary line, got {len(summaries)}"
    summary = summaries[0]

    metadata = dcm_arch.read_metadata(checkpoint_path)
    for key in ("model_id", "training_step"):
        if key not in metadata:
            return EXIT_IDENTITY, f"{checkpoint_path}: header has no {key}"
    model_id = metadata["model_id"]
    training_step = int(metadata["training_step"])
    if training_step != expected_step:
        return EXIT_IDENTITY, (f"{checkpoint_path}: file name says step {expected_step}, "
                               f"header training_step is {training_step}")
    if summary.get("modelID") != model_id:
        return EXIT_IDENTITY, (f"{checkpoint_path}: probe modelID {summary.get('modelID')} != "
                               f"header model_id {model_id}")
    others = existing_model_ids(probes_path) - {model_id}
    if others:
        return EXIT_OTHER_RUN, (f"{probes_path} already holds model_id(s) {sorted(others)}; "
                                f"{model_id} needs its own probes file")

    record = {"step": expected_step, "training_step": training_step, "model_id": model_id}
    if "parent_model_id" in metadata:
        record["parent_model_id"] = metadata["parent_model_id"]
    record["probe_build"] = probe_build
    record.update(summary)
    if "pElo" not in summary:
        record["pElo"] = None
    return 0, record


def load_probe_points(path, expected_model_id):
    """{step: (pElo or None, nll)} from a probes.jsonl that must exist and belong to one run.

    Refuses (raises) a missing file, a record from another model, a repeated step, or a
    record whose `training_step` disagrees with its `step`. A `pElo` of None is a probe
    that measured a non-finite value; callers show it as such rather than as a gap."""
    if expected_model_id is None:
        raise ValueError(f"{path}: no expected model_id given (an arm that has not started has no probes file to read)")
    points = {}
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                print(f"{path}:{number}: unterminated last line (an append in flight); skipped", file=sys.stderr)
                continue
            record = json.loads(line)
            model_id = record_model_id(record)
            if model_id != expected_model_id:
                raise ValueError(f"{path}:{number}: model_id {model_id} != expected {expected_model_id}")
            step = record["step"]
            if "training_step" in record and record["training_step"] != step:
                raise ValueError(f"{path}:{number}: step {step} but training_step {record['training_step']}")
            if step in points:
                raise ValueError(f"{path}:{number}: step {step} appears twice")
            pelo = record.get("pElo")
            if pelo is not None and not math.isfinite(pelo):
                raise ValueError(f"{path}:{number}: pElo {pelo!r} is not a finite number")
            points[step] = (pelo, record["nll"])
    return points


NOT_STARTED = None  # an arm's model_id before its run has started


def probe_builds(path):
    """The probe builds a probes file's records came from (UNRECORDED for records
    written before builds were recorded)."""
    builds = set()
    with open(path) as handle:
        for line in handle:
            if line.strip() and line.endswith("\n"):
                builds.add(json.loads(line).get("probe_build", UNRECORDED))
    return builds


def arm_points(path, model_id, label):
    """`load_probe_points` for a table arm, or None for an arm declared NOT_STARTED.

    An arm is declared not started by giving NOT_STARTED as its model_id. If its probes
    file already has records, the run has started and the table refuses until the arm's
    model_id is filled in, so a column is never read without its identity check."""
    if model_id is NOT_STARTED:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            raise ValueError(f"{label}: {path} has probe records but the arm is declared not started; "
                             f"set its model_id")
        return None
    return load_probe_points(path, model_id)


def pelo_cell(points, step):
    """Table cell for a pElo: blank where the run never reached the step, 'non-finite'
    where the probe measured a non-finite value."""
    if step not in points:
        return ""
    pelo = points[step][0]
    return "non-finite" if pelo is None else f"{pelo:.1f}"


def main():
    if len(sys.argv) != 5:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    checkpoint_path, step_text, probes_path, probe_build = sys.argv[1:]
    status, result = build_record(checkpoint_path, int(step_text), sys.stdin.read(), probes_path, probe_build)
    if status:
        print(result, file=sys.stderr)
        return status
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
