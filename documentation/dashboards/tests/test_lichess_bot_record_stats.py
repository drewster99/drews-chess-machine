"""Tests for the Lichess bot record-statistics checker (scripts/lichess_bot_record_stats.py):
a time without a zone is refused with a clear error instead of crashing.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic game records in a temporary folder.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "lichess_bot_record_stats.py")


def game(created_at):
    return {"createdAt": created_at, "ourColor": "white", "moves": [],
            "outcome": {"ourScore": 1}, "opponent": {"ratingBefore": 1500},
            "setup": {"rated": True, "variant": "standard"}, "us": {"ratingDiff": 6}}


class RecordStatsTimeZoneTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)

    def run_script(self, now, created_at="2026-10-06T23:30:00Z"):
        with open(os.path.join(self.folder, "g.json"), "w") as handle:
            json.dump(game(created_at), handle)
        return subprocess.run([sys.executable, SCRIPT, "--games", self.folder, "--tz", "UTC", "--now", now],
                              capture_output=True, text=True)

    def test_a_now_with_a_zone_is_read(self):
        result = self.run_script("2026-10-07T00:00:00Z")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"lastHour\s+games=1 ")

    def test_a_now_without_a_zone_is_refused_with_a_clear_error(self):
        result = self.run_script("2026-10-07T00:00")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--now '2026-10-07T00:00' has no time zone", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_now_that_is_not_a_time_is_refused_with_a_clear_error(self):
        result = self.run_script("yesterday")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("is not an ISO 8601 time", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_record_time_without_a_zone_is_refused_with_a_clear_error(self):
        result = self.run_script("2026-10-07T00:00:00Z", created_at="2026-10-06T23:30:00")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("createdAt '2026-10-06T23:30:00' has no time zone", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
