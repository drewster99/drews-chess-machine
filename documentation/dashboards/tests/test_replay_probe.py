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
import io
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
from test_lineage import v11_header, write_chain, write_header  # noqa: E402

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


class TrainerStepNamedFilesTests(unittest.TestCase):
    """The tracker on files from architecture format v11, whose names and `training_step`
    carry the trainer step: a resumed segment's file is filed at its segment's base plus its
    record's segment step, a stem holding several segments' files is refused rather than
    filed, and an enumerated file is used for a mark only when its header is the rolling
    file's state."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = self.folder.name
        self.root = root
        self.saved_environment = {key: os.environ.get(key) for key in ("DCM_DASH_ROOT", "DCM_BIN", "STUB_MODE")}
        os.environ["DCM_DASH_ROOT"] = root
        os.environ["DCM_BIN"] = make_stub_binary(root)
        os.environ["STUB_MODE"] = "ok"

    def tearDown(self):
        for key, value in self.saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.folder.cleanup()

    def load(self, runs):
        with open(os.path.join(self.root, "registry.json"), "w") as handle:
            json.dump({"models_dir": self.root, "logs_dir": self.root, "runs": runs}, handle)
        import replay
        self.replay = importlib.reload(replay)

    def probed(self, run):
        path = os.path.join(self.root, "data", f"{run}.csv")
        if not os.path.exists(path):
            return []
        with open(path, newline="") as handle:
            return [(int(row["cum_step"]), int(row["meta_step"])) for row in csv.DictReader(handle) if row["pElo"]]

    def backfill(self, run):
        with mock.patch.object(self.replay, "internals_cells", lambda path: dict(bn1Mean=0.5, sae2="", eff_alpha="")):
            try:
                self.replay.probe_backfill(run, verbose=False)
            except Exception as error:  # the test inspects whichever exception escaped
                return error
        return None

    def test_a_v11_resumed_segments_file_is_read_and_filed_by_its_segment_step(self):
        path = os.path.join(self.root, "T-replay-step2000.safetensors")
        write_header(path, v11_header(local_step=487, cum=2000))
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "T-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006", "enum_stem": "S"},
                                      {"log": "b.txt", "cumstep_base": 1000, "date": "20261006",
                                       "enum_stem": "T", "model_id": "20261006-2-TTTT"}]}})
        self.assertEqual(self.replay.meta_step_of(path), 487)
        self.assertEqual(self.replay._ckpt_index([self.root]), {("20261006-2-TTTT", 487): path})
        self.assertIsNone(self.backfill("r"))
        self.assertEqual(self.probed("r"), [(1487, 487)], "filed at the segment's base plus its segment step")

    def test_a_stem_holding_several_segments_files_is_reported_not_filed(self):
        write_header(os.path.join(self.root, "S-replay-step1000.safetensors"),
                     {"dcm_format_version": "10", "model_id": "20261006-1-SSSS", "creator": "replay",
                      "training_step": "1000", "trainer_completed_steps": "1000"} | {
                         "dcm_lineage": v11_header(local_step=1000, cum=1000)["dcm_lineage"]})
        write_header(os.path.join(self.root, "S-replay-step2000.safetensors"), v11_header(local_step=487, cum=2000))
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "S-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006", "enum_stem": "S"}]}})
        raised = self.backfill("r")
        self.assertIsInstance(raised, self.replay.BackfillIncomplete)
        self.assertIn("S-replay-step2000.safetensors", str(raised))
        self.assertIn("derive-registry", str(raised))
        self.assertEqual(self.probed("r"), [(1000, 1000)], "the pre-v11 file is filed as before")

    def test_discover_stems_refuses_a_stem_shared_by_two_segments(self):
        write_header(os.path.join(self.root, "S-replay-step1000.safetensors"),
                     {"dcm_format_version": "10", "model_id": "20261006-1-SSSS", "creator": "replay",
                      "training_step": "1000", "trainer_completed_steps": "1000",
                      "dcm_lineage": v11_header(local_step=1000, cum=1000)["dcm_lineage"]})
        write_header(os.path.join(self.root, "S-replay-step2000.safetensors"), v11_header(local_step=487, cum=2000))
        self.load({})
        cfg = {"out_model": "S-replay-latest.safetensors",
               "segments": [{"cumstep_base": 0, "label": "run"}, {"cumstep_base": 1513, "label": "resume1"}]}
        printed = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))):
            found = self.replay.discover_enum_stems(cfg, None, verbose=True)
        self.assertEqual(found, {})
        self.assertTrue(any("REFUSED S" in line and "spans 2 model_ids" in line for line in printed), printed)

    def test_freeze_uses_a_same_named_file_only_when_it_holds_the_rolling_state(self):
        rolling = v11_header(local_step=500, cum=2000, model_id="20261006-5-ROLL")
        write_header(os.path.join(self.root, "S-replay-latest.safetensors"), rolling)
        enumerated = os.path.join(self.root, "S-replay-step2000.safetensors")
        write_header(enumerated, v11_header(local_step=500, cum=2000, model_id="20261006-6-OTHR"))
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "S-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006"},
                                      {"log": "b.txt", "cumstep_base": 1500, "date": "20261006"}]}})
        cfg = self.replay.REG["runs"]["r"]
        cum, frozen = self.replay.freeze("r", cfg, 500, rolling)
        self.assertEqual(cum, 2000)
        self.assertEqual(os.path.basename(frozen), "r-step2000-frozen.safetensors",
                         "another model's file under the name is not this mark's checkpoint")
        write_header(enumerated, rolling)
        self.assertEqual(self.replay.freeze("r", cfg, 500, rolling), (2000, enumerated))

    # ----- a resumed v11 segment keeps its stem: earlier segments' files sit under the same name -----

    def resumed_registry(self, latest_segment_fields=None):
        """Run `r`: segment 0 ended at trainer step 1000 and segment 1 has just resumed from it under the
        same stem, `S` (base 1000, no file of its own yet). `latest_segment_fields` are added to segment 1."""
        latest = {"log": "b.txt", "cumstep_base": 1000, "date": "20261006"}
        latest.update(latest_segment_fields or {})
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "S-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006"}, latest]}})

    def test_backfill_does_not_file_an_earlier_segments_v11_file_under_the_latest_segment(self):
        write_header(os.path.join(self.root, "S-replay-step1000.safetensors"),
                     v11_header(local_step=1000, cum=1000, model_id="20261006-1-SEG0"))
        self.resumed_registry()
        raised = self.backfill("r")
        self.assertEqual(self.probed("r"), [], "segment 0's file is not segment 1's step 1000 (cum 2000)")
        self.assertIsInstance(raised, self.replay.BackfillIncomplete)
        self.assertIn("S-replay-step1000.safetensors", str(raised))
        self.assertIn("derive-registry", str(raised))

    def test_backfill_does_not_file_a_v11_file_under_a_segment_naming_another_model_id(self):
        write_header(os.path.join(self.root, "S-replay-step1000.safetensors"),
                     v11_header(local_step=1000, cum=1000, model_id="20261006-1-SEG0"))
        self.resumed_registry({"model_id": "20261006-2-SEG1"})
        raised = self.backfill("r")
        self.assertEqual(self.probed("r"), [])
        self.assertIsInstance(raised, self.replay.BackfillIncomplete)
        self.assertIn("S-replay-step1000.safetensors", str(raised))

    def track(self, run):
        """replay.track(run), returning what it printed to stderr."""
        printed = io.StringIO()
        with mock.patch.object(self.replay, "internals_cells", lambda path: dict(bn1Mean=0.5, sae2="", eff_alpha="")), \
                mock.patch.object(sys, "stderr", printed):
            self.replay.track(run)
        return printed.getvalue()

    def write_rolling(self, model_id="20261006-1-SEG0"):
        """The rolling file and its enumerated twin, both at local step 1000 of a segment whose record
        names segment `seg-b`."""
        for name in ("S-replay-latest.safetensors", "S-replay-step1000.safetensors"):
            write_header(os.path.join(self.root, name), v11_header(local_step=1000, cum=1000, model_id=model_id))

    def test_track_refuses_a_rolling_file_the_latest_segment_does_not_name(self):
        self.write_rolling()
        self.resumed_registry()
        printed = self.track("r")
        self.assertEqual(self.probed("r"), [], "the previous segment's final state is not a mark of segment 1")
        self.assertIn("derive-registry", printed)

    def test_track_refuses_a_rolling_file_of_another_segment_id_or_model_id(self):
        self.write_rolling()
        for fields in ({"segment_id": "seg-c"}, {"model_id": "20261006-2-SEG1"}):
            with self.subTest(fields=fields):
                self.resumed_registry(fields)
                printed = self.track("r")
                self.assertEqual(self.probed("r"), [])
                self.assertIn("derive-registry", printed)

    def test_track_files_a_rolling_file_the_latest_segment_names(self):
        self.write_rolling(model_id="20261006-2-SEG1")
        for fields in ({"segment_id": "seg-b"}, {"model_id": "20261006-2-SEG1"}):
            with self.subTest(fields=fields):
                data = os.path.join(self.root, "data", "r.csv")
                if os.path.exists(data):
                    os.remove(data)
                self.resumed_registry(fields)
                self.track("r")
                self.assertEqual(self.probed("r"), [(2000, 1000)])

    def import_records(self, records):
        """import_probes of `records` (without --ckpt-dir) into segment 1 of a run whose segment 1
        (model M) begins at 1513; returns the CSV's rows as (cum_step, meta_step)."""
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "S-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006"},
                                      {"log": "b.txt", "cumstep_base": 1513, "date": "20261006", "model_id": "M"}]}})
        probes = os.path.join(self.root, "p.jsonl")
        with open(probes, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        with mock.patch.object(sys, "stdout", io.StringIO()), mock.patch.object(sys, "stderr", io.StringIO()):
            self.replay.import_probes("r", probes, 1, ckpt_dirs=(), verbose=True)
        path = os.path.join(self.root, "data", "r.csv")
        if not os.path.exists(path):
            return []
        with open(path, newline="") as handle:
            return [(int(row["cum_step"]), int(row["meta_step"])) for row in csv.DictReader(handle)]

    def probe_record(self, **fields):
        record = {"step": 2000, "training_step": 2000, "model_id": "M", "modelID": "M", "pElo": 1000.0,
                  "nll": 2.0, "model": "/x/S-replay-step2000.safetensors"}
        record.update(fields)
        return record

    def test_import_probes_files_a_trainer_step_record_by_its_segment_step(self):
        rows = self.import_records([self.probe_record(trainer_step=2000, segment_step=487,
                                                      step_basis="trainer_step")])
        self.assertEqual(rows, [(2000, 487)], "base 1513 + segment step 487, not base + trainer step 2000")

    def test_import_probes_rejects_a_trainer_step_record_without_a_segment_step(self):
        rows = self.import_records([self.probe_record(trainer_step=2000, segment_step=None,
                                                      step_basis="trainer_step")])
        self.assertEqual(rows, [])

    def test_import_probes_files_a_legacy_segment_step_record_by_its_name_step(self):
        rows = self.import_records([self.probe_record(step=487, training_step=487, trainer_step=2000,
                                                      segment_step=487, step_basis="legacy_segment_step",
                                                      model="/x/S-replay-step487.safetensors")])
        self.assertEqual(rows, [(2000, 487)])

    def test_import_probes_rejects_a_record_whose_named_file_has_an_unreadable_header(self):
        with open(os.path.join(self.root, "S-replay-step300.safetensors"), "wb") as handle:
            handle.write(struct.pack("<Q", 4096) + b'{"__metadata__": {')
        future = v11_header(local_step=400, cum=1913, model_id="M")
        future["dcm_format_version"] = "99"
        write_header(os.path.join(self.root, "S-replay-step400.safetensors"), future)
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "S-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006"},
                                      {"log": "b.txt", "cumstep_base": 1513, "date": "20261006", "model_id": "M"}]}})
        probes = os.path.join(self.root, "p.jsonl")
        with open(probes, "w") as handle:
            for step in (300, 400, 500):
                handle.write(json.dumps({"step": step, "training_step": step, "modelID": "M", "pElo": 1000.0,
                                         "nll": 2.0, "model": f"/x/S-replay-step{step}.safetensors"}) + "\n")
        printed = io.StringIO()
        with mock.patch.object(sys, "stdout", io.StringIO()), mock.patch.object(sys, "stderr", printed):
            added = self.replay.import_probes("r", probes, 1, ckpt_dirs=(self.root,), verbose=True)
        self.assertEqual(added, 1, "the record whose file is not there is imported; the import is not aborted")
        self.assertEqual(self.probed("r"), [(2013, 500)])
        self.assertIn("S-replay-step300.safetensors: header cannot be read", printed.getvalue())
        self.assertIn("S-replay-step400.safetensors: header cannot be read", printed.getvalue())

    def test_import_probes_rejects_a_record_stating_an_unknown_step_basis(self):
        self.load({"r": {"frozen_glob": "r-step*-frozen.safetensors", "out_model": "S-replay-latest.safetensors",
                         "segments": [{"log": "a.txt", "cumstep_base": 0, "date": "20261006"},
                                      {"log": "b.txt", "cumstep_base": 1513, "date": "20261006", "model_id": "M"}]}})
        probes = os.path.join(self.root, "p.jsonl")
        with open(probes, "w") as handle:
            handle.write(json.dumps({"step": 500, "modelID": "M", "pElo": 1000.0, "segment_step": 500,
                                     "step_basis": "wall_clock", "model": "/x/S-replay-step500.safetensors"}) + "\n")
        with mock.patch.object(sys, "stdout", io.StringIO()), mock.patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(self.replay.import_probes("r", probes, 1, ckpt_dirs=(), verbose=True), 0)


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
