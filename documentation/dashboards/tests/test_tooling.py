"""Tests for the shared analysis tooling: architecture rules (scripts/dcm_arch.py), session-log
ordering (scripts/dcm_session_logs.py), probe records (experiments/probe_record.py) and the
replay-buffer game length (experiments/table_common.py).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic files in a temporary folder.
"""
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import dcm_arch  # noqa: E402
import dcm_session_logs  # noqa: E402
import probe_record  # noqa: E402
import table_common  # noqa: E402


def group(**fields):
    base = dict(count=3, channels=128, activation_function="relu", use_rezero=True,
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


class BufferGameLengthTests(unittest.TestCase):
    def test_games_at_the_boundary_are_interpolated(self):
        feed = [(50, 0, 0), (100, 400_000, 6_000), (150, 800_000, 12_000)]
        axis = [p for _, p, _ in feed]
        self.assertEqual(table_common.games_fed_at(feed, axis, 400_000), 6_000)
        self.assertEqual(table_common.games_fed_at(feed, axis, 600_000), 9_000)
        self.assertIsNone(table_common.games_fed_at(feed, axis, -1))
        self.assertIsNone(table_common.games_fed_at(feed, axis, 900_000))


if __name__ == "__main__":
    unittest.main()
