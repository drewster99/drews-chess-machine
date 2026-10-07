"""Tests for the shared analysis tooling: architecture rules (scripts/dcm_arch.py), session-log
ordering (scripts/dcm_session_logs.py), probe records (experiments/probe_record.py) and the
replay-buffer game length (experiments/table_common.py).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic files in a temporary folder.
"""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, os.path.join(REPO, "documentation", "dashboards"))
import dcm_arch  # noqa: E402
import dcm_session_logs  # noqa: E402
import probe_record  # noqa: E402
import table_common  # noqa: E402
sys.path.insert(0, HERE)
from test_lineage import v11_header  # noqa: E402


def group(**fields):
    base = dict(count=3, channels=128, se_style="scale_and_bias", activation_function="relu", use_rezero=True,
                rezero_alpha_init=0.4472136, se_beta_init="glorot", se_activation="relu")
    base.update(fields)
    return base


def write_header(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header)


class ArchitectureTests(unittest.TestCase):
    def test_legacy_file_resolves_cap_to_init(self):
        blocks = dcm_arch.rezero_blocks({"architecture": json.dumps({"block_groups": [group()]}),
                                         "dcm_format_version": "5"})
        self.assertEqual([b.alpha_cap for b in blocks], [0.4472136] * 3)

    def test_v6_file_uses_its_cap_and_allows_zero_init(self):
        blocks = dcm_arch.rezero_blocks({"architecture": json.dumps(
            {"block_groups": [group(rezero_alpha_init=0.0, rezero_alpha_cap=1.0)]}), "dcm_format_version": "6"})
        self.assertEqual([(b.alpha_init, b.alpha_cap) for b in blocks], [(0.0, 1.0)] * 3)
        self.assertEqual(blocks[0].effective(0.0), 0.0)
        self.assertAlmostEqual(blocks[0].effective(0.5), 0.46211715726000974)

    def test_v6_file_without_cap_is_refused(self):
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.rezero_blocks({"architecture": json.dumps({"block_groups": [group()]}),
                                    "dcm_format_version": "6"})

    def test_non_positive_cap_is_refused(self):
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.rezero_blocks({"architecture": json.dumps({"block_groups": [group(rezero_alpha_init=0.0)]}),
                                    "dcm_format_version": "5"})

    def test_block_without_rezero_has_no_effective_alpha(self):
        blocks = dcm_arch.rezero_blocks({"architecture": json.dumps({"block_groups": [group(use_rezero=False)]}),
                                         "dcm_format_version": "5"})
        with self.assertRaises(dcm_arch.ArchitectureError):
            blocks[0].effective(0.1)


class HeaderReadTests(unittest.TestCase):
    """`dcm_arch.read_metadata` is the one safetensors header reader; a damaged header is an
    ArchitectureError, never an OverflowError / MemoryError / AttributeError from reading it."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "damaged.safetensors")

    def tearDown(self):
        self.folder.cleanup()

    def write_raw(self, data):
        with open(self.path, "wb") as handle:
            handle.write(data)

    def test_damaged_header_length_is_refused(self):
        self.write_raw(struct.pack("<Q", 2 ** 63) + b"{}")
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.read_metadata(self.path)

    def test_header_length_beyond_the_bound_or_zero_is_refused(self):
        for length in (dcm_arch.MAX_HEADER_BYTES + 1, 0):
            self.write_raw(struct.pack("<Q", length) + b"{}")
            with self.assertRaises(dcm_arch.ArchitectureError):
                dcm_arch.read_metadata(self.path)

    def test_header_that_is_not_an_object_or_has_no_metadata_object_is_refused(self):
        for header in (b"[1, 2]", b'{"__metadata__": [1]}', b'{"x": 1}', b"not json", b"\xff\xfe"):
            self.write_raw(struct.pack("<Q", len(header)) + header)
            with self.assertRaises(dcm_arch.ArchitectureError, msg=header):
                dcm_arch.read_metadata(self.path)

    def test_read_header_returns_the_tensor_index_and_data_start(self):
        header = {"__metadata__": {"model_id": "M1"}, "t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        raw = json.dumps(header).encode()
        self.write_raw(struct.pack("<Q", len(raw)) + raw + struct.pack("<f", 1.5))
        read, data_start = dcm_arch.read_header(self.path)
        self.assertEqual(read, header)
        self.assertEqual(data_start, 8 + len(raw))
        self.assertEqual(dcm_arch.read_metadata(self.path), {"model_id": "M1"})

    def test_lineage_reader_is_the_same_reader_with_its_own_error_type(self):
        import dcm_lineage
        self.assertEqual(dcm_lineage.MAX_HEADER_BYTES, dcm_arch.MAX_HEADER_BYTES)
        self.write_raw(struct.pack("<Q", 2 ** 63) + b"{}")
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.read_metadata(self.path)
        write_header(self.path, {"model_id": "M1"})
        self.assertEqual(dcm_lineage.read_metadata(self.path), dcm_arch.read_metadata(self.path))


class SessionLogTests(unittest.TestCase):
    def test_launches_in_one_second_sort_in_launch_order(self):
        names = ["dcm_log_20261002-215333-10.txt", "dcm_log_20261002-215333-2.txt",
                 "dcm_log_20261002-215333.txt", "dcm_log_20261001-000000.txt"]
        ordered = sorted(names, key=dcm_session_logs.session_log_sort_key)
        self.assertEqual(ordered, ["dcm_log_20261001-000000.txt", "dcm_log_20261002-215333.txt",
                                   "dcm_log_20261002-215333-2.txt", "dcm_log_20261002-215333-10.txt"])

    def test_name_outside_the_scheme_is_refused(self):
        with self.assertRaises(ValueError):
            dcm_session_logs.session_log_sort_key("dcm_log_copy.txt")


class ProbeRecordTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.checkpoint = os.path.join(self.folder.name, "run-replay-step2000.safetensors")
        write_header(self.checkpoint, {"model_id": "M1", "training_step": "2000", "parent_model_id": "M0"})
        self.probes = os.path.join(self.folder.name, "probes.jsonl")

    def tearDown(self):
        self.folder.cleanup()

    def summary(self, **fields):
        base = {"modelID": "M1", "pElo": 1200.5, "nll": 2.4, "set": "wide"}
        base.update(fields)
        return json.dumps(base) + "\n"

    def test_good_probe_becomes_a_record_with_identity(self):
        status, record = probe_record.build_record(self.checkpoint, 2000, "[LOG] x\n" + self.summary(), self.probes, "Test.app@sha256:000000000000")
        self.assertEqual(status, 0)
        self.assertEqual(list(record)[:4], ["step", "training_step", "model_id", "parent_model_id"])
        self.assertEqual((record["step"], record["model_id"], record["pElo"]), (2000, "M1", 1200.5))
        self.assertEqual(record["probe_build"], "Test.app@sha256:000000000000")

    def test_error_event_and_missing_summary_are_retryable_failures(self):
        error = json.dumps({"event": "error", "error": "boom"}) + "\n"
        self.assertEqual(probe_record.build_record(self.checkpoint, 2000, error, self.probes, "Test.app@sha256:000000000000")[0], 3)
        self.assertEqual(probe_record.build_record(self.checkpoint, 2000, "", self.probes, "Test.app@sha256:000000000000")[0], 3)

    def test_identity_mismatches_are_refused(self):
        self.assertEqual(probe_record.build_record(self.checkpoint, 3000, self.summary(), self.probes, "Test.app@sha256:000000000000")[0], 4)
        self.assertEqual(probe_record.build_record(self.checkpoint, 2000, self.summary(modelID="X"), self.probes, "Test.app@sha256:000000000000")[0], 4)

    def test_probes_file_from_another_run_is_refused(self):
        with open(self.probes, "w") as handle:
            handle.write(json.dumps({"step": 1000, "modelID": "OTHER", "pElo": 1, "nll": 2}) + "\n")
        self.assertEqual(probe_record.build_record(self.checkpoint, 2000, self.summary(), self.probes, "Test.app@sha256:000000000000")[0], 5)

    def test_non_finite_pelo_is_recorded_as_null(self):
        text = json.dumps({"modelID": "M1", "nll": 2.4, "set": "wide"}) + "\n"
        status, record = probe_record.build_record(self.checkpoint, 2000, text, self.probes, "Test.app@sha256:000000000000")
        self.assertEqual((status, record["pElo"]), (0, None))

    def test_a_v11_resumed_segments_record_carries_the_trainer_step_and_its_basis(self):
        checkpoint = os.path.join(self.folder.name, "run-replay-step2000-v11.safetensors")
        write_header(checkpoint, v11_header(local_step=487, cum=2000, model_id="M1"))
        status, record = probe_record.build_record(checkpoint, 2000, self.summary(), self.probes,
                                                   "Test.app@sha256:000000000000")
        self.assertEqual(status, 0, record)
        self.assertEqual(list(record)[:3], ["step", "training_step", "model_id"])
        self.assertEqual((record["step"], record["trainer_step"], record["segment_step"], record["step_basis"]),
                         (2000, 2000, 487, "trainer_step"))
        keys = list(record)
        self.assertGreater(keys.index("step_basis"), keys.index("model_id"), "the new keys come after the leading ones")

    def test_a_pre_v11_record_names_its_segment_step_basis(self):
        status, record = probe_record.build_record(self.checkpoint, 2000, self.summary(), self.probes,
                                                   "Test.app@sha256:000000000000")
        self.assertEqual(status, 0)
        self.assertEqual(record["step_basis"], "legacy_unknown_writer")
        self.assertEqual(list(record)[:4], ["step", "training_step", "model_id", "parent_model_id"])

    def test_a_record_appends_to_a_probes_file_of_records_without_a_basis(self):
        with open(self.probes, "w") as handle:
            handle.write(json.dumps({"step": 1000, "training_step": 1000, "model_id": "M1", "pElo": 1.0,
                                     "nll": 2.0}) + "\n")
        status, record = probe_record.build_record(self.checkpoint, 2000, self.summary(), self.probes,
                                                   "Test.app@sha256:000000000000")
        self.assertEqual(status, 0, record)
        with open(self.probes, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.assertEqual(sorted(probe_record.load_probe_points(self.probes, "M1")), [1000, 2000])

    def test_a_header_the_tools_refuse_is_an_identity_failure(self):
        checkpoint = os.path.join(self.folder.name, "run-replay-step2000-bad.safetensors")
        write_header(checkpoint, v11_header(local_step=487, cum=2000, model_id="M1", stated=487))
        self.assertEqual(probe_record.build_record(checkpoint, 487, self.summary(), self.probes,
                                                   "Test.app@sha256:000000000000")[0], 4)

    def test_load_probe_points_refusals(self):
        with self.assertRaises(FileNotFoundError):
            probe_record.load_probe_points(self.probes, "M1")
        with open(self.probes, "w") as handle:
            handle.write(json.dumps({"step": 1000, "modelID": "M1", "pElo": 1.0, "nll": 2.0}) + "\n")
            handle.write(json.dumps({"step": 1000, "modelID": "M1", "pElo": 1.0, "nll": 2.0}) + "\n")
        with self.assertRaises(ValueError):
            probe_record.load_probe_points(self.probes, "M1")
        with open(self.probes, "w") as handle:
            handle.write(json.dumps({"step": 1000, "modelID": "M2", "pElo": 1.0, "nll": 2.0}) + "\n")
        with self.assertRaises(ValueError):
            probe_record.load_probe_points(self.probes, "M1")

    def test_arm_declared_not_started_refuses_once_it_has_records(self):
        self.assertIsNone(probe_record.arm_points(self.probes, probe_record.NOT_STARTED, "D"))
        with open(self.probes, "w") as handle:
            handle.write(json.dumps({"step": 1000, "modelID": "M1", "pElo": 1.0, "nll": 2.0}) + "\n")
        with self.assertRaises(ValueError):
            probe_record.arm_points(self.probes, probe_record.NOT_STARTED, "D")


class CsvPointsTests(unittest.TestCase):
    """table_common's dashboard-CSV arm reads a non-finite probe as a measurement, the way
    probe_record reads a probes.jsonl arm, never as a step the run did not reach."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        rows = [
            dict(cum_step="1000", pElo="1500.25", nll="2.4", note="", probe_build="A.app@sha256:1"),
            dict(cum_step="2000", pElo="", nll="2.5", note="probe-backfill; probe: pElo non-finite",
                 probe_build="A.app@sha256:2"),
            dict(cum_step="3000", pElo="", nll="", note="log-backfill (fast-net)", probe_build=""),
            dict(cum_step="4000", pElo="1510", nll="", note="", probe_build="A.app@sha256:1"),
        ]
        with open(os.path.join(self.folder.name, "arm.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cum_step", "pElo", "nll", "note", "probe_build"])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.folder.cleanup()

    def test_csv_points_keeps_non_finite_rows(self):
        with mock.patch.object(table_common, "DATA", self.folder.name):
            points = table_common.csv_points("arm")
            builds = table_common.csv_probe_builds("arm")
        self.assertEqual(points, {1000: (1500.25, 2.4), 2000: (None, 2.5), 4000: (1510.0, None)})
        self.assertEqual(probe_record.pelo_cell(points, 2000), "non-finite")
        self.assertEqual(probe_record.pelo_cell(points, 3000), "", "a row never probed is not a measurement")
        self.assertEqual(builds, {"A.app@sha256:1", "A.app@sha256:2"})

    def test_nll_cell_is_blank_where_there_is_no_value(self):
        points = {1000: (1500.25, 2.4), 4000: (1510.0, None)}
        self.assertEqual([probe_record.nll_cell(points, s) for s in (1000, 3000, 4000)], ["2.4000", "", ""])

    def test_marker_string_has_one_source(self):
        import _schema
        self.assertIs(table_common.NON_FINITE_PELO_NOTE, _schema.NON_FINITE_PELO_NOTE)


class InitReproducibilityScriptTests(unittest.TestCase):
    """scripts/init_reproducibility.sh fails when a minted file cannot be hashed, rather than
    printing the other lines and exiting 0 (which reads as a complete, comparable listing)."""

    def test_init_reproducibility_fails_when_hashing_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = os.path.join(folder, "fake-dcm")
            with open(fake, "w") as handle:
                handle.write('#!/bin/zsh\nwhile [ $# -gt 0 ]; do\n'
                             '  if [ "$1" = "--out-model" ]; then print -r -- "not a safetensors file" > "$2"; fi\n'
                             '  shift\ndone\n')
            os.chmod(fake, 0o755)
            completed = subprocess.run(
                ["/bin/zsh", os.path.join(REPO, "scripts", "init_reproducibility.sh"), fake,
                 os.path.join(folder, "minted")], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class ProbeLoopScriptTests(unittest.TestCase):
    """experiments/probe_loop.sh, run with HOME pointed at a temporary folder (its Models
    folder is built from $HOME, so nothing under the real Application Support is read or
    written) and a stub probe binary inside a temporary .app bundle."""

    STEM = "20261003-probe-loop-test"

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.home = os.path.join(self.folder.name, "home")
        self.models = os.path.join(self.home, "Library", "Application Support", "DrewsChessMachine", "Models")
        os.makedirs(self.models)
        self.probes = os.path.join(self.folder.name, "probes.jsonl")
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            process.kill()
            process.wait()
        self.folder.cleanup()

    def app_executable(self, bundle, body):
        macos = os.path.join(self.folder.name, bundle, "Contents", "MacOS")
        os.makedirs(macos)
        path = os.path.join(macos, "DrewsChessMachine")
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)
        return path

    def run_loop(self, arguments, **environment):
        env = dict(os.environ, HOME=self.home, **environment)
        return subprocess.run(["/bin/zsh", os.path.join(REPO, "experiments", "probe_loop.sh")] + arguments,
                              capture_output=True, text=True, timeout=120, env=env)

    def test_probe_loop_refuses_a_trainer_pid_that_is_not_the_trainer(self):
        trainer = self.app_executable("Trainer.app", "sleep 60\n")
        rolling = os.path.join(self.models, f"{self.STEM}-replay-latest.safetensors")
        self.processes.append(subprocess.Popen([trainer, "--replay-corpus", "corpus", "--out-model", rolling]))
        probe = self.app_executable("Probe.app", "exit 0\n")
        completed = self.run_loop([self.STEM, self.probes], PROBE_BIN=probe, TRAINER_PID=str(os.getpid()),
                                  PROBE_START_WAIT_SEC="30")
        self.assertEqual(completed.returncode, 3, completed.stdout + completed.stderr)
        self.assertIn(f"TRAINER_PID {os.getpid()} is not the trainer", completed.stderr)

    def test_probe_loop_does_not_record_a_probe_that_exited_nonzero(self):
        write_header(os.path.join(self.models, f"{self.STEM}-replay-step2000.safetensors"),
                     {"model_id": "M1", "training_step": "2000"})
        summary = json.dumps({"modelID": "M1", "pElo": 1200.5, "nll": 2.4, "set": "wide"})
        probe = self.app_executable("Probe.app", f"echo '{summary}'\nexit 1\n")
        completed = self.run_loop(["--once", self.STEM, self.probes], PROBE_BIN=probe, PROBE_MAX_ATTEMPTS="1")
        with open(self.probes) as handle:
            self.assertEqual(handle.read(), "", "a probe that exited non-zero is not a measurement")
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("probe exit 1", completed.stderr)

    def test_probe_loop_records_a_probe_that_succeeded(self):
        write_header(os.path.join(self.models, f"{self.STEM}-replay-step2000.safetensors"),
                     {"model_id": "M1", "training_step": "2000"})
        summary = json.dumps({"modelID": "M1", "pElo": 1200.5, "nll": 2.4, "set": "wide"})
        probe = self.app_executable("Probe.app", f"echo '{summary}'\n")
        completed = self.run_loop(["--once", self.STEM, self.probes], PROBE_BIN=probe)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        with open(self.probes) as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual([(r["step"], r["model_id"], r["pElo"]) for r in records], [(2000, "M1", 1200.5)])


    def test_probe_loop_above_step_skips_an_earlier_segments_files_under_the_stem(self):
        # Segment 0 (model M0) wrote steps 1000 and 1513; the resumed segment (M1, v11)
        # continued the series under the same stem from trainer step 1513.
        write_header(os.path.join(self.models, f"{self.STEM}-replay-step1000.safetensors"),
                     {"model_id": "M0", "training_step": "1000"})
        write_header(os.path.join(self.models, f"{self.STEM}-replay-step1513.safetensors"),
                     {"model_id": "M0", "training_step": "1513"})
        write_header(os.path.join(self.models, f"{self.STEM}-replay-step2000.safetensors"),
                     v11_header(local_step=487, cum=2000, model_id="M1"))
        summary = json.dumps({"modelID": "M1", "pElo": 1200.5, "nll": 2.4, "set": "wide"})
        probe = self.app_executable("Probe.app", f"echo '{summary}'\n")
        completed = self.run_loop(["--once", self.STEM, self.probes], PROBE_BIN=probe, PROBE_ABOVE_STEP="1513")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("skipping step 1000", completed.stdout)
        self.assertIn("skipping step 1513", completed.stdout)
        with open(self.probes) as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual([(r["step"], r["segment_step"], r["step_basis"]) for r in records],
                         [(2000, 487, "trainer_step")])

    def test_probe_loop_refuses_probe_segment_with_above_step(self):
        probe = self.app_executable("Probe.app", "exit 0\n")
        completed = self.run_loop(["--once", self.STEM, self.probes], PROBE_BIN=probe, PROBE_SEGMENT="1",
                                  PROBE_ABOVE_STEP="1000")
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("cannot be used together", completed.stderr)


class BufferGameLengthTests(unittest.TestCase):
    def test_games_at_the_boundary_are_interpolated(self):
        feed = [(50, 0, 0), (100, 400_000, 6_000), (150, 800_000, 12_000)]
        axis = [p for _, p, _ in feed]
        self.assertEqual(table_common.games_fed_at(feed, axis, 400_000), 6_000)
        self.assertEqual(table_common.games_fed_at(feed, axis, 600_000), 9_000)
        self.assertIsNone(table_common.games_fed_at(feed, axis, -1))
        self.assertIsNone(table_common.games_fed_at(feed, axis, 900_000))

    def test_buffer_game_length_by_trainer_step_keys_on_the_trainer_step(self):
        # A resumed segment's log (segment 1, started at trainer step 513): its lines'
        # trainer steps are its segment steps plus 513, so its 1000-step marks fall on
        # trainer steps, not segment steps; a line from an older build without
        # trainerStep= is not read.
        lines = ["00:00:00.000 [REPLAY] step=1 loss=1 plies=100000 games=1000 epoch=0\n"]
        for segment_step, plies, games in ((487, 400_000, 6_000), (987, 800_000, 12_000), (1487, 1_200_000, 18_000)):
            lines.append(f"00:00:00.000 [REPLAY] step={segment_step} loss=1 plies={plies} games={games} epoch=0 "
                         f"mom=0.9 trainerStep={segment_step + 513}\n")
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "seg1.txt"), "w") as handle:
                handle.writelines(lines)
            with mock.patch.object(table_common, "LOGS", folder):
                by_trainer = table_common.buffer_plies_per_game_by_trainer_step("seg1.txt")
                by_segment = table_common.buffer_plies_per_game("seg1.txt")
        # Trainer step 1000's buffer boundary (400,000 − 500,000 plies) lies before the
        # first line read, so it has no value; 2000's (700,000) is interpolated between
        # the lines at 400,000 (6,000 games) and 800,000 (12,000): 10,500 games.
        self.assertEqual(list(by_trainer), [2000])
        self.assertAlmostEqual(by_trainer[2000], table_common.BUFFER / (18_000 - 10_500))
        self.assertEqual(by_segment, {}, "no segment step of this log is a multiple of 1000")


if __name__ == "__main__":
    unittest.main()
