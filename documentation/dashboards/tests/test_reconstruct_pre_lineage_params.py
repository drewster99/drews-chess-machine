"""Tests for the pre-lineage reconstruction report (scripts/reconstruct_pre_lineage_params.py).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test builds synthetic session logs and header-only .safetensors files in a temporary
folder and passes every folder explicitly; no test reads the real log or model folders.
"""
import contextlib
import datetime
import io
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import reconstruct_pre_lineage_params as recon  # noqa: E402

UTC = datetime.timezone.utc
RUN_ID = "20261001-3-AbCd"
PARENT_ID = "20260930-7-Prnt"
HPARAMS = ("[REPLAY-HPARAMS] lr=0.01 batch=4096 wd=0.0001 momentum=0.9 gradClip=30 entropyBonus=0 "
           "drawPenalty=0 policyW=1 valueW=1 illegalW=0 pLabelSmooth=0 vLabelSmooth=0 dropout=0 "
           "lrWarmup=100 bufCap=500000 replayRatio=1 minPrefill=250000 complementCE=off sqrtBatchLR=off "
           "batchStats=100 klProbe=0")
CYCLE = "[REPLAY-CYCLE] lr=cosine(1.0..0.001 period=8000) mom=off origin=fresh"


def unix(year, month, day, hour, minute, second):
    return int(datetime.datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp())


def write_header(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header)


def write_model(folder, name, created_at, training_step, model_id=RUN_ID, parent=PARENT_ID,
                creator="replay", extra=None):
    metadata = {"dcm_format_version": "6", "model_id": model_id, "created_at_unix": str(created_at),
                "creator": creator, "training_step": str(training_step), "parent_model_id": parent,
                "notes": f"corpus replay autosave @ step {training_step}"}
    metadata.update(extra or {})
    path = os.path.join(folder, name)
    write_header(path, metadata)
    return path


def write_log(folder, name, lines):
    """A session log: `lines` are (HH:MM:SS.mmm, message)."""
    path = os.path.join(folder, name)
    with open(path, "w") as handle:
        for stamp, message in lines:
            handle.write(f"{stamp}  {message}\n")
    return path


def replay_log_lines(saves, start_model=True, cycle=True):
    """The startup banner of a replay run, then each (stamp, step, rolling name,
    enumerated name or None) save."""
    lines = [("12:00:00.100", "[APP] launched build=2275 git=de0f22b branch=main")]
    if start_model:
        lines.append(("12:00:00.200", f"[REPLAY] start-model: seed.safetensors modelID={PARENT_ID} encoding=basic30"))
    lines.append(("12:00:00.300", HPARAMS))
    if cycle:
        lines.append(("12:00:01.000", CYCLE))
    for stamp, step, rolling, enumerated in saves:
        lines.append((stamp, f"[REPLAY] step={step} loss=1.0000 pLoss=0.5000 vLoss=0.5000 trainerStep={step}"))
        lines.append((stamp, f"[REPLAY] saved trainer model (autosave) step={step} trainerStep={step} "
                             f"nextGame=10 shard=0 epoch=0 -> {rolling}"))
        if enumerated:
            lines.append((stamp, f"[REPLAY] enumerated checkpoint -> {enumerated}"))
    return lines


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = self.temporary.name
        self.models = os.path.join(root, "models")
        self.logs = os.path.join(root, "logs")
        self.experiments = os.path.join(root, "experiments")
        for folder in (self.models, self.logs, self.experiments):
            os.mkdir(folder)
        self.out = os.path.join(root, "report.json")

    def tearDown(self):
        self.temporary.cleanup()

    def run_report(self, *extra):
        argv = ["--models", self.models, "--logs", self.logs, "--log-timezone", "UTC",
                "--out", self.out, *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = recon.main(argv)
        self.assertEqual(status, 0)
        with open(self.out) as handle:
            report = json.load(handle)
        return {os.path.basename(e["file"]): e for e in report["entries"]}, report


class CleanMatchTests(ReportTestCase):
    def test_rolling_and_enumerated_files_match_their_log(self):
        log = write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines([
            ("12:10:01.500", 1000, "run-replay-latest.safetensors", "run-replay-step1000.safetensors"),
            ("12:20:03.000", 2000, "run-replay-latest.safetensors", "run-replay-step2000.safetensors"),
        ]))
        write_model(self.models, "run-replay-step1000.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000)
        write_model(self.models, "run-replay-step2000.safetensors", unix(2026, 10, 1, 12, 20, 1), 2000)
        # The rolling file holds the last save's bytes.
        write_model(self.models, "run-replay-latest.safetensors", unix(2026, 10, 1, 12, 20, 1), 2000)
        entries, report = self.run_report()
        self.assertEqual(report["counts"], {"reconstructed": 3})
        for name, step in (("run-replay-step1000.safetensors", 1000),
                           ("run-replay-step2000.safetensors", 2000),
                           ("run-replay-latest.safetensors", 2000)):
            entry = entries[name]
            self.assertEqual(entry["status"], "reconstructed", entry)
            self.assertEqual(entry["provenance"], "reconstructed from logs")
            self.assertEqual(entry["model_id"], RUN_ID)
            self.assertEqual(entry["source_log"], log)
            self.assertEqual(entry["hparams_line"], HPARAMS)
            self.assertEqual(entry["cycle_line"], CYCLE)
            self.assertEqual(entry["agreement"]["training_step"], step)
            self.assertEqual(entry["agreement"]["parent_model_id"], PARENT_ID)
            self.assertNotIn("parameters_file", entry)
        self.assertEqual(report["provenance"], "reconstructed from logs")

    def test_fresh_run_without_start_model_or_cycle_line(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "fresh-replay-latest.safetensors", None)], start_model=False, cycle=False))
        write_model(self.models, "fresh-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000, parent="")
        entries, _ = self.run_report()
        entry = entries["fresh-replay-latest.safetensors"]
        self.assertEqual(entry["status"], "reconstructed", entry)
        self.assertIsNone(entry["cycle_line"])
        self.assertIn("no [REPLAY-CYCLE] line", entry["cycle_line_note"])
        self.assertIsNone(entry["start_model_line"])

    def test_save_after_midnight_takes_the_next_date(self):
        lines = [("23:59:50.000", HPARAMS),
                 ("23:59:59.900", "[REPLAY] step=999 loss=1.0 trainerStep=999"),
                 ("00:00:05.000", "[REPLAY] saved trainer model (autosave) step=1000 trainerStep=1000 "
                                  "nextGame=1 shard=0 epoch=0 -> late.safetensors")]
        write_log(self.logs, "dcm_log_20261001-235949.txt", lines)
        write_model(self.models, "late.safetensors", unix(2026, 10, 2, 0, 0, 4), 1000, parent="")
        entries, _ = self.run_report()
        self.assertEqual(entries["late.safetensors"]["status"], "reconstructed", entries["late.safetensors"])

    def test_experiment_readme_attaches_the_one_parameters_file_named_with_the_log(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "run-replay-latest.safetensors", None)]))
        write_model(self.models, "run-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000)
        experiment = os.path.join(self.experiments, "20261001-exp")
        os.mkdir(experiment)
        for name in ("parameters.json", "parameters-continue.json"):
            with open(os.path.join(experiment, name), "w") as handle:
                handle.write("{}\n")
        with open(os.path.join(experiment, "README.md"), "w") as handle:
            handle.write("# Experiment\n\nThe same `parameters.json` as `other-exp/parameters-continue.json`.\n\n"
                         "## Continuation\n\n`parameters-continue.json`, log `dcm_log_20261001-150000.txt`.\n\n"
                         "## Launch record\n\nlog `dcm_log_20261001-120000.txt`\n"
                         "  --parameters experiments/20261001-exp/parameters.json\n")
        entries, _ = self.run_report("--experiments", self.experiments)
        entry = entries["run-replay-latest.safetensors"]
        self.assertEqual(entry["parameters_file"], os.path.join(experiment, "parameters.json"))
        self.assertEqual(entry["parameters_file_readme"], os.path.join(experiment, "README.md"))


class UnmatchedTests(ReportTestCase):
    def test_two_logs_naming_the_file_in_the_window_is_ambiguous(self):
        for name in ("dcm_log_20261001-120000.txt", "dcm_log_20261001-120000-2.txt"):
            write_log(self.logs, name, replay_log_lines(
                [("12:10:01.500", 1000, "shared-replay-latest.safetensors", None)]))
        write_model(self.models, "shared-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000)
        entries, _ = self.run_report()
        entry = entries["shared-replay-latest.safetensors"]
        self.assertEqual(entry["status"], "unmatched")
        self.assertIn("ambiguous", entry["reason"])
        self.assertEqual(entry["candidate_logs"], ["dcm_log_20261001-120000.txt", "dcm_log_20261001-120000-2.txt"])
        self.assertIsNone(entry["source_log"])
        self.assertIsNone(entry["hparams_line"])
        self.assertNotIn("provenance", entry)

    def test_missing_log_is_unmatched(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "other-replay-latest.safetensors", None)]))
        write_model(self.models, "lost-replay-step1000.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000)
        entries, _ = self.run_report()
        entry = entries["lost-replay-step1000.safetensors"]
        self.assertEqual(entry["status"], "unmatched")
        self.assertIn("no session log", entry["reason"])
        self.assertIsNone(entry["source_log"])

    def test_save_line_outside_the_window_is_unmatched(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "run-replay-latest.safetensors", None)]))
        # Created an hour before the only save line naming it.
        write_model(self.models, "run-replay-latest.safetensors", unix(2026, 10, 1, 11, 10, 0), 1000)
        entries, _ = self.run_report()
        entry = entries["run-replay-latest.safetensors"]
        self.assertEqual(entry["status"], "unmatched")
        self.assertIn("none stamped within", entry["reason"])

    def test_training_step_disagreeing_with_the_save_line_is_unmatched(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "run-replay-latest.safetensors", None)]))
        write_model(self.models, "run-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 999)
        entries, _ = self.run_report()
        entry = entries["run-replay-latest.safetensors"]
        self.assertEqual(entry["status"], "unmatched")
        self.assertIn("training_step 999", entry["reason"])

    def test_parent_disagreeing_with_the_start_model_line_is_unmatched(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "run-replay-latest.safetensors", None)]))
        write_model(self.models, "run-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000,
                    parent="20260101-1-Else")
        entries, _ = self.run_report()
        entry = entries["run-replay-latest.safetensors"]
        self.assertEqual(entry["status"], "unmatched")
        self.assertIn("parent_model_id", entry["reason"])

    def test_two_model_ids_matched_to_one_log_unmatch_both(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines([
            ("12:10:01.500", 1000, "run-replay-latest.safetensors", "run-replay-step1000.safetensors"),
        ]))
        write_model(self.models, "run-replay-step1000.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000)
        write_model(self.models, "run-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000,
                    model_id="20261001-9-Othr")
        entries, report = self.run_report()
        self.assertEqual(report["counts"], {"unmatched": 2})
        for entry in entries.values():
            self.assertIn("different model IDs", entry["reason"])
            self.assertNotIn("provenance", entry)

    def test_non_replay_file_is_unmatched(self):
        write_model(self.models, "gui.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000, creator="gui")
        entries, _ = self.run_report()
        self.assertIn("not 'replay'", entries["gui.safetensors"]["reason"])


class HasLineageTests(ReportTestCase):
    def test_file_with_a_lineage_record_is_reported_has_lineage(self):
        write_log(self.logs, "dcm_log_20261001-120000.txt", replay_log_lines(
            [("12:10:01.500", 1000, "new-replay-latest.safetensors", None)]))
        write_model(self.models, "new-replay-latest.safetensors", unix(2026, 10, 1, 12, 10, 0), 1000,
                    extra={"dcm_format_version": "7", "dcm_lineage": "{}"})
        entries, report = self.run_report()
        entry = entries["new-replay-latest.safetensors"]
        self.assertEqual(entry["status"], "has_lineage")
        self.assertEqual(entry["model_id"], RUN_ID)
        self.assertIsNone(entry["source_log"])
        self.assertIsNone(entry["hparams_line"])
        self.assertEqual(report["counts"], {"has_lineage": 1})


class LineParsingTests(unittest.TestCase):
    def test_enumerated_line_name_excludes_the_replaced_note(self):
        match = recon.ENUMERATED_RE.fullmatch(
            "[REPLAY] enumerated checkpoint -> run-replay-step1000.safetensors "
            "(replaced this run's own earlier save of step 1000)")
        self.assertEqual(match.group(1), "run-replay-step1000.safetensors")

    def test_repeated_daylight_saving_hour_has_two_readings_and_is_ambiguous_at_the_edge(self):
        zone = recon.zoneinfo.ZoneInfo("America/Chicago")
        readings = recon.unix_readings(datetime.datetime(2026, 11, 1, 1, 30, 0), zone)
        self.assertEqual(len(readings), 2)
        self.assertEqual(readings[1] - readings[0], 3600)
        event = recon.SaveEvent("log", "f.safetensors", 1, "line", readings)
        # Fits the first reading only: which hour the line meant decides the match.
        self.assertEqual(recon.event_in_window(event, int(readings[0]) - 10), "ambiguous")
        self.assertEqual(recon.event_in_window(event, int(readings[0]) - 7200), "no")


class OutputTests(ReportTestCase):
    def test_existing_out_is_refused_and_left_untouched(self):
        with open(self.out, "w") as handle:
            handle.write("keep me\n")
        argv = ["--models", self.models, "--logs", self.logs, "--log-timezone", "UTC", "--out", self.out]
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            status = recon.main(argv)
        self.assertEqual(status, 2)
        self.assertIn("never written over", stderr.getvalue())
        with open(self.out) as handle:
            self.assertEqual(handle.read(), "keep me\n")

    def test_models_are_required(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                recon.main(["--logs", self.logs, "--log-timezone", "UTC", "--out", self.out])
        self.assertFalse(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
