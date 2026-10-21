"""Tests for `test_set_results` in scripts/dcm_lineage.py: the app's reading of a
model file's `dcm_test_set_results` (test-set results plan D4), handed back as
(status, record-or-reason).

What reads as evaluated, failed or unreadable is the app's rule
(`ModelTestSetResultsField.reading`; ModelTestSetResultsTests in Swift). These
tests check that each status the app reports reaches the caller intact, on real
model files (`app_files`).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Needs a built DrewsChessMachine binary; without one these tests skip.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)
import app_files  # noqa: E402
import dcm_lineage  # noqa: E402

KEY = "dcm_test_set_results"


def tearDownModule():
    app_files.remove_base()


class TestSetResultsTests(unittest.TestCase):
    """Fails if: a status is mapped to another, the evaluated record or the
    reason is dropped, or an unreadable value raises instead of being reported."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def file(self, name, metadata=None, remove=()):
        return app_files.write_model_file(os.path.join(self.dir, name), metadata or {}, remove=remove,
                                          keep_lineage=True)

    def test_absent_is_not_recorded(self):
        self.assertEqual(dcm_lineage.test_set_results(self.file("absent.safetensors", remove=(KEY,))),
                         ("not_recorded", None))

    def test_a_minted_file_is_evaluated_with_its_sets(self):
        kind, record = dcm_lineage.test_set_results(self.file("minted.safetensors"))
        self.assertEqual(kind, "evaluated")
        self.assertEqual(sorted(s["id"] for s in record["sets"]), ["lichess-200", "lichess-wide"])
        self.assertTrue(all(len(s["fingerprint_sha256"]) == 64 for s in record["sets"]))

    def test_failed_carries_the_writers_reason(self):
        raw = json.dumps({"reason": "r", "schema": 1, "status": "failed"}, sort_keys=True)
        self.assertEqual(dcm_lineage.test_set_results(self.file("failed.safetensors", {KEY: raw})), ("failed", "r"))

    def test_an_unreadable_value_is_reported_not_raised(self):
        kind, reason = dcm_lineage.test_set_results(self.file("bad.safetensors", {KEY: "not json"}))
        self.assertEqual(kind, "unreadable")
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
