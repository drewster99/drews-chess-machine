"""Tests for replay.probe(): every way a probe can fail is a ProbeFailure, never an empty
result, and a measurement records the build that made it.

Run: python3 -m unittest discover -s documentation/dashboards/tests
The tracker is pointed at a temporary registry (DCM_DASH_ROOT) and a stub binary inside a
temporary .app bundle; no real model, log or dashboard file is touched.
"""
import importlib
import json
import os
import stat
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

STUB = """#!/usr/bin/env python3
import json, os, sys, time
mode = os.environ["STUB_MODE"]
if mode == "ok":
    print("[PROBE] log line")
    print(json.dumps({"modelID": "M", "pElo": 1234.5, "nll": 2.3, "set": "wide",
                      "policy_logit_abs_max": 17.0, "policy_logit_abs_max_peak": 28.0}))
elif mode == "nopelo":
    print(json.dumps({"modelID": "M", "nll": 2.3, "set": "wide"}))
elif mode == "error":
    print(json.dumps({"event": "error", "error": "cannot load"}))
elif mode == "exit":
    sys.exit(3)
elif mode == "hang":
    time.sleep(30)
"""


class ReplayProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = cls.folder.name
        with open(os.path.join(root, "registry.json"), "w") as handle:
            json.dump({"models_dir": root, "logs_dir": root, "runs": {}}, handle)
        macos = os.path.join(root, "Stub.app", "Contents", "MacOS")
        os.makedirs(macos)
        cls.binary = os.path.join(macos, "stub")
        with open(cls.binary, "w") as handle:
            handle.write(STUB)
        os.chmod(cls.binary, os.stat(cls.binary).st_mode | stat.S_IXUSR)
        os.environ["DCM_DASH_ROOT"] = root
        os.environ["DCM_BIN"] = cls.binary
        import replay
        cls.replay = importlib.reload(replay)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DCM_DASH_ROOT")
        os.environ.pop("DCM_BIN")
        cls.folder.cleanup()

    def run_mode(self, mode):
        os.environ["STUB_MODE"] = mode
        return self.replay.probe("unused.safetensors")

    def test_measurement_carries_the_build(self):
        result = self.run_mode("ok")
        self.assertEqual(result["pElo"], 1234.5)
        self.assertTrue(result["probe_build"].startswith("Stub.app@sha256:"))

    def test_missing_pelo_is_a_non_finite_measurement(self):
        self.assertIsNone(self.run_mode("nopelo")["pElo"])

    def test_error_event_exit_status_and_timeout_are_failures(self):
        for mode in ("error", "exit"):
            with self.assertRaises(self.replay.ProbeFailure):
                self.run_mode(mode)
        saved = self.replay.PROBE_TIMEOUT_SECONDS
        self.replay.PROBE_TIMEOUT_SECONDS = 1
        try:
            with self.assertRaises(self.replay.ProbeFailure):
                self.run_mode("hang")
        finally:
            self.replay.PROBE_TIMEOUT_SECONDS = saved


if __name__ == "__main__":
    unittest.main()
