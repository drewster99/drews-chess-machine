"""Tests for `test_set_results` in scripts/dcm_lineage.py: the reading of a
model file's `dcm_test_set_results` (test-set results plan D4), mirroring
`ModelTestSetResultsField.reading` in Persistence/ModelTestSetResults.swift.

Run: python3 -m unittest discover -s documentation/dashboards/tests
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_lineage  # noqa: E402

KEY = dcm_lineage.TEST_SET_RESULTS_KEY


def evaluated(pelo=1630.4, pelo_bound=None):
    """The JSON the app writes (sorted keys), one set."""
    return json.dumps({
        "build": 2427, "evaluated_at_unix": 1791414697, "policy_tail_precision": "mixed_final_projection",
        "schema": 1, "status": "evaluated",
        "sets": [{
            "avg_correct_probability": 0.1923, "avg_correct_rank": 3.667, "description": "200 puzzles.",
            "fingerprint_sha256": "a" * 64, "id": "lichess-200", "nll": 2.071, "pelo": pelo,
            "pelo_bound": pelo_bound, "positions": 200,
            "themes": [{"correct": 20, "id": "hangingPiece", "title": "Hanging piece", "total": 25}],
            "title": "Lichess puzzles, 200", "top1_correct": 99, "top5_correct": 170,
        }],
    }, sort_keys=True, separators=(",", ":"))


class TestSetResultsTests(unittest.TestCase):

    def test_absent_is_not_recorded(self):
        self.assertEqual(dcm_lineage.test_set_results({}), ("not_recorded", None))

    def test_evaluated(self):
        kind, record = dcm_lineage.test_set_results({KEY: evaluated()})
        self.assertEqual(kind, "evaluated")
        self.assertEqual(record["sets"][0]["top1_correct"], 99)
        self.assertEqual(record["sets"][0]["themes"][0]["id"], "hangingPiece")

    def test_a_bound_instead_of_an_estimate(self):
        kind, record = dcm_lineage.test_set_results({KEY: evaluated(pelo=None, pelo_bound="all_correct")})
        self.assertEqual(kind, "evaluated")
        self.assertEqual(record["sets"][0]["pelo_bound"], "all_correct")

    def test_failed(self):
        raw = json.dumps({"reason": "r", "schema": 1, "status": "failed"}, sort_keys=True)
        self.assertEqual(dcm_lineage.test_set_results({KEY: raw}), ("failed", "r"))

    def test_unreadable_values_are_reported_not_raised(self):
        for raw in ["not json", "[]", json.dumps({"schema": 2, "status": "failed", "reason": "r"}),
                    json.dumps({"schema": 1, "status": "maybe"}), json.dumps({"schema": 1, "status": "failed"}),
                    evaluated(pelo=None, pelo_bound=None), evaluated(pelo=1.0, pelo_bound="all_wrong"),
                    evaluated(pelo=None, pelo_bound="maybe"),
                    json.dumps({"schema": True, "status": "failed", "reason": "r"}),
                    evaluated().replace('"build":2427,', ''),
                    evaluated().replace('"pelo_bound":null,', ''),
                    evaluated().replace('"top1_correct":99', '"top1_correct":"99"')]:
            kind, reason = dcm_lineage.test_set_results({KEY: raw})
            self.assertEqual(kind, "unreadable", raw)
            self.assertIsInstance(reason, str)

    def test_the_key_and_schema_match_the_app(self):
        swift = open(os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachine", "Persistence",
                                  "ModelTestSetResults.swift")).read()
        self.assertIn(f'static let metadataKey = "{KEY}"', swift)
        self.assertIn(f"static let schema = {dcm_lineage.TEST_SET_RESULTS_SCHEMA}", swift)


if __name__ == "__main__":
    unittest.main()
