"""Tests for replay.probe(): every way a probe can fail is a ProbeFailure, never an empty
result, and a measurement records the build that made it; for probe_backfill, which saves
every row it filed before reporting the checkpoints it could not probe or file; and for
tick.py, which renders the dashboard even when probing fails.

Run: python3 -m unittest discover -s documentation/dashboards/tests
The tracker is pointed at a temporary registry (DCM_DASH_ROOT) and a stub binary inside a
temporary .app bundle; no real model, log or dashboard file is touched.
"""
import csv
import importlib
import json
import os
import runpy
import stat
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, DASH)
sys.path.insert(0, HERE)
from test_lineage import write_chain  # noqa: E402

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


def make_stub_binary(root):
    macos = os.path.join(root, "Stub.app", "Contents", "MacOS")
    os.makedirs(macos)
    binary = os.path.join(macos, "stub")
    with open(binary, "w") as handle:
        handle.write(STUB)
    os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
    return binary


class ProbeBackfillTests(unittest.TestCase):
    """probe_backfill over a run whose segments are identified by lineage `segment_id`:
    the checkpoints are header-only lineage files, the probe is the stub, and the
    architecture-derived cells are patched (the files carry no tensors)."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = self.folder.name
        write_chain(root)
        with open(os.path.join(root, "registry.json"), "w") as handle:
            json.dump({"models_dir": root, "logs_dir": root, "runs": {"r": {
                "frozen_glob": "r-step*-frozen.safetensors",
                "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261003", "segment_id": "seg-a"},
                             {"log": "b.txt", "cumstep_base": 1000, "date": "20261003", "segment_id": "seg-b"}]}}},
                      handle)
        self.saved_environment = {key: os.environ.get(key) for key in ("DCM_DASH_ROOT", "DCM_BIN", "STUB_MODE")}
        os.environ["DCM_DASH_ROOT"] = root
        os.environ["DCM_BIN"] = make_stub_binary(root)
        os.environ["STUB_MODE"] = "ok"
        import replay
        self.replay = importlib.reload(replay)
        self.root = root

    def tearDown(self):
        for key, value in self.saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.folder.cleanup()

    def probed_cum_steps(self):
        path = os.path.join(self.root, "data", "r.csv")
        if not os.path.exists(path):
            return []
        with open(path, newline="") as handle:
            return [int(row["cum_step"]) for row in csv.DictReader(handle) if row["pElo"]]

    def backfill(self, internals_cells):
        """probe_backfill("r"), returning the exception it raised (None if it returned)."""
        with mock.patch.object(self.replay, "internals_cells", internals_cells):
            try:
                self.replay.probe_backfill("r", verbose=False)
            except Exception as error:  # the test inspects whichever exception escaped
                return error
        return None

    def test_backfill_saves_probed_rows_when_a_later_file_cannot_be_filed(self):
        def internals_cells(path):
            if os.path.basename(path) == "seg-b-replay-step1000.safetensors":
                raise ValueError("some blocks use ReZero and some do not")
            return dict(bn1Mean=0.5, sae2="", eff_alpha="")
        raised = self.backfill(internals_cells)
        self.assertEqual(self.probed_cum_steps(), [500, 1000, 1500, 2500],
                         "every row probed and filed in the pass is saved")
        self.assertIsInstance(raised, self.replay.BackfillIncomplete)
        self.assertIn("seg-b-replay-step1000.safetensors", str(raised))
        self.assertIn("some blocks use ReZero and some do not", str(raised))

    def test_an_unreadable_lineage_file_is_reported_not_skipped(self):
        with open(os.path.join(self.root, "seg-b-replay-step2000.safetensors"), "wb") as handle:
            handle.write(struct.pack("<Q", 4096) + b'{"__metadata__": {')
        raised = self.backfill(lambda path: dict(bn1Mean=0.5, sae2="", eff_alpha=""))
        self.assertEqual(self.probed_cum_steps(), [500, 1000, 1500, 2000, 2500])
        self.assertIsInstance(raised, self.replay.BackfillIncomplete)
        self.assertIn("seg-b-replay-step2000.safetensors", str(raised))


class TickTests(unittest.TestCase):
    """tick.py: a probe failure is reported in the exit status, after the dashboard is rendered."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = self.folder.name
        with open(os.path.join(root, "registry.json"), "w") as handle:
            json.dump({"models_dir": root, "logs_dir": root, "runs": {"r": {
                "out_model": "r-replay-latest.safetensors", "frozen_glob": "r-step*-frozen.safetensors",
                "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261003"}]}}}, handle)
        self.saved_root = os.environ.get("DCM_DASH_ROOT")
        os.environ["DCM_DASH_ROOT"] = root
        import replay
        self.replay = importlib.reload(replay)

    def tearDown(self):
        if self.saved_root is None:
            os.environ.pop("DCM_DASH_ROOT", None)
        else:
            os.environ["DCM_DASH_ROOT"] = self.saved_root
        self.folder.cleanup()

    def test_tick_renders_when_probing_fails(self):
        rendered = []
        failure = self.replay.ProbeFailure("probe of x exited 3")
        with mock.patch.object(self.replay, "track", side_effect=failure), \
                mock.patch.object(self.replay, "probe_backfill", side_effect=failure), \
                mock.patch.object(self.replay, "render", side_effect=lambda: rendered.append(True)), \
                mock.patch.object(sys, "argv", ["tick.py", "r"]):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(os.path.join(DASH, "tick.py"), run_name="__main__")
        self.assertEqual(rendered, [True], "the dashboard is rendered despite the failures")
        self.assertEqual(caught.exception.code, 1)

    def test_tick_renders_when_backfill_is_incomplete(self):
        rendered = []
        incomplete = self.replay.BackfillIncomplete("r: 1 checkpoint(s) could not be probed or filed")
        with mock.patch.object(self.replay, "track"), \
                mock.patch.object(self.replay, "probe_backfill", side_effect=incomplete), \
                mock.patch.object(self.replay, "render", side_effect=lambda: rendered.append(True)), \
                mock.patch.object(sys, "argv", ["tick.py", "r"]):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(os.path.join(DASH, "tick.py"), run_name="__main__")
        self.assertEqual(rendered, [True])
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
