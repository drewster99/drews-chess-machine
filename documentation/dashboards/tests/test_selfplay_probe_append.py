"""Tests for selfplay_probe_append.py: it only ever extends selfplay_probe/<run>.csv, and
refuses (leaving the file byte-identical) a CSV it cannot extend correctly — a torn last
line, a header other than its own columns, or a row with an empty field.

Run: python3 -m unittest discover -s documentation/dashboards/tests
The module's registry folder and log folder are pointed at a temporary folder; no real
CSV, registry or log is touched.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import selfplay_probe_append  # noqa: E402

LOG = "dcm_log_20261003-120000.txt"
MODEL = "20261003-1-AAAA"


class SelfplayProbeAppendTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = self.folder.name
        with open(os.path.join(root, "selfplay_registry.json"), "w") as handle:
            json.dump({"runs": {"r": {"logs": [LOG], "base_modelID": MODEL}}}, handle)
        with open(os.path.join(root, LOG), "w") as handle:
            handle.write(f"12:00:00.000 [TACTICAL-LICHESS] tick set=wide step=2000 n=4435 NLL=3.1234 pElo=1510 "
                         f"model={MODEL}-3\n")
        os.mkdir(os.path.join(root, "selfplay_probe"))
        self.csv = os.path.join(root, "selfplay_probe", "r.csv")
        self.patches = [mock.patch.object(selfplay_probe_append, "HERE", root),
                        mock.patch.object(selfplay_probe_append, "LOGDIR", root),
                        mock.patch.object(sys, "argv", ["selfplay_probe_append.py", "r"])]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.folder.cleanup()

    def write_csv(self, data):
        with open(self.csv, "wb") as handle:
            handle.write(data)

    def read_csv(self):
        with open(self.csv, "rb") as handle:
            return handle.read()

    def assert_refused_unchanged(self, data):
        self.write_csv(data)
        with self.assertRaises(SystemExit) as caught:
            selfplay_probe_append.main()
        self.assertNotEqual(caught.exception.code, 0)
        self.assertEqual(self.read_csv(), data, "a refused CSV is left byte-identical")

    def test_a_torn_last_line_is_refused(self):
        self.assert_refused_unchanged(b"step,pElo,nll,segment\r\n1000,1500")

    def test_a_header_other_than_the_appenders_columns_is_refused(self):
        self.assert_refused_unchanged(b"step,pElo,nll\r\n1000,1500,3.2\r\n")

    def test_a_row_with_an_empty_field_is_refused(self):
        self.assert_refused_unchanged(b"step,pElo,nll,segment\r\n1000,,3.2,0\r\n")

    def test_a_row_with_a_missing_field_is_refused(self):
        self.assert_refused_unchanged(b"step,pElo,nll,segment\r\n1000,1500,3.2\r\n")

    def test_a_good_csv_is_extended_and_keeps_its_bytes(self):
        existing = b"step,pElo,nll,segment\r\n1000,1500,3.2,0\r\n"
        self.write_csv(existing)
        selfplay_probe_append.main()
        self.assertEqual(self.read_csv(), existing + b"2000,1510,3.1234,0\r\n")


if __name__ == "__main__":
    unittest.main()
