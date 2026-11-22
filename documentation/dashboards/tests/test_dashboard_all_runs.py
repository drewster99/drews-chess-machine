"""Forked lineage runs, carried derive copies, registry exclusions and the log
backfill (documentation/plans-active/MASTER_DASHBOARD_ALL_RUNS_PLAN.md).

R-elitecyc's lineage run forked four ways at trainer step 384,964 (the main line,
R-elitecyc-hilr, -fp32 and -fp32fix are each an exact resume of one save), and the
fp32 start file is a `--derive-model --set-compute-dtype` copy that keeps the source
segment's record under a new model ID. Before this, `derive_runs` keyed segments by
index (one fork overwrote the others) and refused the copy as "files disagree on
model_id", so `derive-registry` could not run at all on the real Models folder.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, DASH)
sys.path.insert(0, HERE)
import app_files  # noqa: E402
import dcm_lineage  # noqa: E402
import _lineage_registry  # noqa: E402
from test_lineage import SEGMENTS, record_for, write_chain, write_v7  # noqa: E402

# seg-x: a second exact resume of seg-a's 1000 save (a fork beside seg-b).
FORK = [SEGMENTS[0], ("seg-x", "20261003-9-XXXX", 1000, 1000, 4000, 260000, 800.0, 850.0, 1_790_150_000,
                      SEGMENTS[1][9])]


def tearDownModule():
    app_files.remove_base()


def write_fork(directory):
    """The three-segment chain plus seg-x, a fork of seg-a at 1000."""
    write_chain(directory)
    for local in (500, 1000):
        record = record_for(1, local, specs=FORK)
        write_v7(directory, f"seg-x-replay-step{local}.safetensors", record, FORK[1][1], local)


def write_carried_copy(directory, name="seg-a-fp32-start.safetensors", model_id="20261003-7-COPY"):
    """A derive output that kept seg-a's 1000 trainer state: seg-a's record, a new model ID."""
    record = record_for(0, 1000)
    record["derivation_history"] = [derivation(model_id, "seg-a")]
    write_v7(directory, name, record, model_id, 1000)


def derivation(model_id, carried_segment_id):
    """One `derivation_history` entry as `--derive-model --set-compute-dtype` writes it
    on a trainer-state file (shape of 20261112-zlra-elitecyc-fp32-start's)."""
    header, _ = app_files.read_file(app_files.base_file())
    architecture = header["__metadata__"]["architecture"]
    return {"build": "2669 (e9d4f722)", "created_at_unix": 1_790_000_500, "model_id": model_id,
            "operations": [{"arguments": {"value": "float32"}, "changed_architecture_fields": ["compute_data_type"],
                            "operation": "set-compute-dtype", "rewritten_tensors": []}],
            "parent_model_id": SEGMENTS[0][1], "source_file": "seg-a-replay-step1000.safetensors",
            "source_format_version": 7, "source_sha256": "00" * 32,
            "trainer_state": {"segment_id": carried_segment_id, "source_architecture": architecture,
                              "source_architecture_format_version": 12, "trainer_completed_steps": 1000}}


def registry(runs):
    return {"models_dir": "~/x", "logs_dir": "~/y", "runs": runs}


class ForkedLineageTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def derived(self):
        recorded, _, errors = dcm_lineage.scan(self.dir)
        self.assertEqual(errors, {})
        return dcm_lineage.derive_runs(recorded)["run-1"]

    def test_both_chains_of_a_fork_are_kept_with_their_parent(self):
        """Protects: every segment of a forked lineage run is derived.
        Fails if: segments are keyed by index, so seg-x (index 1, like seg-b) replaces
        seg-b or is lost."""
        write_fork(self.dir)
        run = self.derived()
        by_id = {s.fields["segment_id"]: s for s in run.segments}
        self.assertEqual(sorted(by_id), ["seg-a", "seg-b", "seg-c", "seg-x"])
        self.assertEqual((by_id["seg-b"].parent_segment_id, by_id["seg-x"].parent_segment_id,
                          by_id["seg-c"].parent_segment_id, by_id["seg-a"].parent_segment_id),
                         ("seg-a", "seg-a", "seg-b", None))
        self.assertEqual(sorted(s.fields["segment_id"] for s in run.children("seg-a")), ["seg-b", "seg-x"])
        self.assertEqual((by_id["seg-x"].fields["cumstep_base"], by_id["seg-x"].fields["model_id"]),
                         (1000, "20261003-9-XXXX"))

    def test_a_carried_trainer_state_copy_is_set_aside_not_refused(self):
        """Protects: a --derive-model copy that keeps a segment's record under a new
        model ID does not stop derivation and is not one of the segment's files.
        Fails if: it is grouped with seg-a (LineageError "files disagree on model_id")."""
        write_chain(self.dir)
        write_carried_copy(self.dir)
        run = self.derived()
        self.assertEqual(run.carried_copies, ["seg-a-fp32-start.safetensors"])
        seg_a = [s for s in run.segments if s.fields["segment_id"] == "seg-a"][0]
        self.assertEqual(seg_a.fields["model_id"], SEGMENTS[0][1])
        self.assertNotIn("seg-a-fp32-start.safetensors", seg_a.files)

    def test_a_file_of_another_model_without_a_carrying_derive_is_still_refused(self):
        """Protects: only a derive that names the segment's trainer state is set aside.
        Fails if: any second model ID in a segment is quietly dropped."""
        write_chain(self.dir)
        record = record_for(0, 1000)
        record["derivation_history"] = [derivation("20261003-8-ELSE", "seg-z")]
        write_v7(self.dir, "seg-a-other.safetensors", record, "20261003-8-ELSE", 1000)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.derive_runs(recorded)


class ForkRegistryPlanTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        write_fork(self.dir)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        self.runs = dcm_lineage.derive_runs(recorded)

    def main_run(self):
        return {"label": "main", "segments": [{"log": "a.txt", "model_id": SEGMENTS[0][1]},
                                              {"log": "b.txt", "model_id": SEGMENTS[1][1]},
                                              {"log": "c.txt", "model_id": SEGMENTS[2][1]}]}

    def test_a_fork_feeds_its_own_registry_run_on_the_parents_axis(self):
        """Protects: a registry run starting at a fork is matched and shares the
        lineage's step offset. Fails if: it is refused as a lineage run matching two
        registry runs, or its cumstep_base is left unrecorded (no segment 0 of its own)."""
        reg = registry({"main": self.main_run(),
                        "fork": {"label": "fork", "segments": [{"log": "x.txt", "model_id": FORK[1][1]}]}})
        updated, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual((report["conflicts"], report["unmatched_segments"], report["unmatched_runs"]), ([], [], {}))
        fork = updated["runs"]["fork"]["segments"][0]
        self.assertEqual((fork["segment_id"], fork["cumstep_base"], fork["lineage_run_id"]), ("seg-x", 1000, "run-1"))
        self.assertEqual(updated["runs"]["main"]["segments"][2]["cumstep_base"], 2500)

    def test_a_registry_run_not_starting_at_a_fork_is_still_refused(self):
        """Protects: splitting one chain across registry runs stays a contradiction.
        Fails if: any split of a forked lineage run is accepted (here seg-c, whose
        parent seg-b has one child)."""
        main = self.main_run()
        main["segments"].pop()
        reg = registry({"main": main, "tail": {"segments": [{"log": "c.txt", "model_id": SEGMENTS[2][1]}]}})
        with self.assertRaises(_lineage_registry.RegistryContradiction):
            _lineage_registry.plan(reg, self.runs)

    def test_an_unclaimed_fork_is_proposed_under_its_parents_registry_run(self):
        reg = registry({"main": self.main_run()})
        _, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual([(name, fields["segment_id"]) for name, fields in report["unmatched_segments"]],
                         [("main", "seg-x")])

    def test_excluded_segments_and_runs_are_reported_not_proposed(self):
        reg = registry({"main": self.main_run()})
        reg["excluded_segments"] = {"seg-x": "a test fork"}
        _, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual(report["unmatched_segments"], [])
        self.assertEqual(report["excluded"], [("run-1", "seg-x", "a test fork")])
        reg = registry({})
        reg["excluded_lineage_runs"] = {"run-1": "a test run"}
        _, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual((report["unmatched_runs"], report["excluded"]), ({}, [("run-1", None, "a test run")]))


def replay_line(time, step, trainer, games):
    return (f"{time}  [REPLAY] step={step} trainerStep={trainer} games={games} loss=3.0000 pLoss=2.0000 "
            f"vLoss=1.0000 pIllM=0.0100 gNorm=0.5000 ms=1000.0\n")


class LogBackfillTests(unittest.TestCase):
    """replay.log_backfill on a temporary registry and session logs."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = cls.folder.name
        os.mkdir(os.path.join(root, "data"))
        # Segment 0 starts at trainer step 500 (a resumed v11 segment): its saves are at
        # trainer 1000 and 2000 (segment steps 500 and 1500). It ran on to trainer 3200,
        # but segment 1 resumed from the 2000 save, so 3000 belongs to an abandoned tail.
        lines = [replay_line(f"10:{m:02d}:00.000", step, step + 500, step * 10)
                 for m, step in enumerate(range(100, 2701, 100))]
        for step in (500, 1500, 2500):
            lines.append(f"10:{step // 100:02d}:30.000  [REPLAY] saved trainer model (autosave) step={step} "
                         f"trainerStep={step + 500} nextGame=1 shard=0 epoch=0 -> r-latest.safetensors\n")
        lines.insert(6, "10:05:30.000  [CHECKPOINT] test sets trainer step 1000 (r-latest.safetensors) (300 ms): "
                        "lichess-200 n=200 top1=1 top5=2 nll=1.6000 pElo=1400.0; lichess-wide n=4435 top1=2 "
                        "top5=3 nll=1.8500 pElo=1800.5\n")
        lines.append("10:30:00.000  [CHECKPOINT] test sets trainer step 2000 (r-latest.safetensors) (300 ms): "
                     "lichess-200 n=200 top1=1 top5=2 nll=1.6000 pElo=1400.0; lichess-wide n=4435 top1=2 "
                     "top5=3 nll=1.8400 pElo=1810.0\n")
        lines.append("10:31:00.000  [CHECKPOINT] test sets trainer step 3000 (r-latest.safetensors) (300 ms): "
                     "lichess-200 n=200 top1=1 top5=2 nll=1.6000 pElo=1400.0; lichess-wide n=4435 top1=2 "
                     "top5=3 nll=1.9000 pElo=1700.0\n")
        with open(os.path.join(root, "dcm_log_20261101-100000.txt"), "w") as handle:
            handle.writelines(lines)
        with open(os.path.join(root, "dcm_log_20261102-100000.txt"), "w") as handle:
            handle.write(replay_line("10:00:00.000", 1, 2001, 0))
        reg = {"models_dir": root, "logs_dir": root, "runs": {"r": {
            "label": "r", "out_model": "r-latest.safetensors", "frozen_glob": "r-step*-frozen.safetensors",
            "segments": [{"log": "dcm_log_20261101-100000.txt", "date": "20261101", "cumstep_base": 500},
                         {"log": "dcm_log_20261102-100000.txt", "date": "20261102", "cumstep_base": 2000}]}}}
        with open(os.path.join(root, "registry.json"), "w") as handle:
            json.dump(reg, handle)
        os.environ["DCM_DASH_ROOT"] = root
        import importlib
        import replay
        cls.replay = importlib.reload(replay)
        cls.replay.log_backfill("r", verbose=False)
        cls.rows = {int(r["cum_step"]): r for r in cls.replay.read_csv("r")}

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DCM_DASH_ROOT")
        cls.folder.cleanup()

    def test_marks_are_the_saves_the_log_records(self):
        """Protects: a resumed v11 segment's rows sit where its saves are (trainer-step
        thousands). Fails if: marks are taken at segment-step thousands (trainer 1500,
        2500) instead of the logged saves."""
        self.assertEqual(sorted(self.rows), [1000, 2000])
        self.assertEqual((self.rows[1000]["meta_step"], self.rows[1000]["pLoss"]), ("500", "2.0"))

    def test_the_writers_test_set_result_fills_pelo(self):
        """Protects: a save whose file is gone still gets the writer's measured pElo.
        Fails if: the [CHECKPOINT] test sets line is not read, or read from the wrong set."""
        self.assertEqual((self.rows[1000]["pElo"], self.rows[1000]["nll"]), ("1800.5", "1.85"))
        self.assertEqual(self.rows[2000]["pElo"], "1810.0")
        self.assertIn(self.replay.LOG_TEST_SETS_NOTE, self.rows[1000]["note"])
        self.assertEqual(self.rows[1000]["probe_build"], "")

    def test_a_tail_past_the_next_segments_start_adds_nothing(self):
        """Protects: an abandoned tail (trained past the save the next segment resumed
        from) adds no row. Fails if: trainer 3000's save lands on the axis."""
        self.assertNotIn(3000, self.rows)

    def test_a_second_pass_changes_nothing(self):
        added, measured = self.replay.log_backfill("r", verbose=False)
        self.assertEqual((added, measured), (0, 0))


class LogBackfillProbedRowTests(unittest.TestCase):
    """A row a probe filed keeps its measurements; the log adds its blank games and wall cells."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = cls.folder.name
        os.mkdir(os.path.join(root, "data"))
        with open(os.path.join(root, "dcm_log_20261001-100000.txt"), "w") as handle:
            handle.writelines(f"10:{m:02d}:00.000  [REPLAY] step={step} games={step * 10} loss=3.0000 pLoss=2.0000 "
                              f"vLoss=1.0000 pIllM=0.0100 gNorm=0.5000 ms=1000.0\n"
                              for m, step in enumerate(range(100, 2101, 100)))
        reg = {"models_dir": root, "logs_dir": root, "runs": {"p": {
            "label": "p", "out_model": "p-replay-latest.safetensors", "frozen_glob": "p-step*-frozen.safetensors",
            "segments": [{"log": "dcm_log_20261001-100000.txt", "date": "20261001", "cumstep_base": 0,
                          "games_base": 0}]}}}
        with open(os.path.join(root, "registry.json"), "w") as handle:
            json.dump(reg, handle)
        os.environ["DCM_DASH_ROOT"] = root
        import importlib
        import replay
        cls.replay = importlib.reload(replay)
        _, snapshot = cls.replay.read_csv_for_update("p")
        cls.replay.write_csv("p", [dict(cum_step=1000, meta_step=1000, segment=0, pElo=1500.0, nll=2.0,
                                        note="probe-backfill")], snapshot)
        cls.replay.log_backfill("p", verbose=False)
        cls.rows = {int(r["cum_step"]): r for r in cls.replay.read_csv("p")}

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DCM_DASH_ROOT")
        cls.folder.cleanup()

    def test_blank_games_and_wall_are_filled_and_the_measurement_kept(self):
        """Protects: a probed pre-lineage checkpoint's row gets the compute axis from the
        log. Fails if: probe rows stay without games_fed / wall_sec, or the pass
        overwrites the probe's pElo."""
        row = self.rows[1000]
        self.assertEqual((row["pElo"], row["games_fed"], row["note"]), ("1500.0", "10000", "probe-backfill"))
        self.assertNotEqual(row["wall_sec"], "")
        self.assertEqual(sorted(self.rows), [1000, 2000])


if __name__ == "__main__":
    unittest.main()
