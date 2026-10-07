"""Tests for lineage reading (scripts/dcm_lineage.py), registry reconciliation
(documentation/dashboards/_lineage_registry.py) and the checkpoint inventory's lineage
columns (ckpt_inventory.py).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic header-only .safetensors files in a temporary folder.
"""
import copy
import datetime
import json
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, DASH)
import dcm_lineage  # noqa: E402
import _lineage_registry  # noqa: E402
import ckpt_inventory  # noqa: E402

SWIFT = os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachine")

DEVICE_M4 = {"hw_model": "Mac16,8", "chip": "Apple M4 Pro", "is_vm": False,
             "os_version": "Version 27.2", "gpu_name": "Apple M4 Pro"}
DEVICE_VM = {"hw_model": "VirtualMac2,1", "chip": "Apple M5", "is_vm": True,
             "os_version": "Version 27.1", "gpu_name": "Apple Paravirtual device"}
BUILD = {"build_number": 2300, "git_hash": "abc1234", "git_branch": "main", "git_dirty": False}

# A three-segment corpus-replay run: a fresh segment 0, then two exact resumes.
# Each segment: (segment_id, model_id, start step, local steps, games, positions,
#                train-step seconds, wall seconds, started unix, device).
SEGMENTS = [
    ("seg-a", "20261003-1-AAAA", 0, 1000, 5000, 330000, 900.5, 1000.25, 1_790_000_000, DEVICE_VM),
    ("seg-b", "20261003-2-BBBB", 1000, 1500, 7400, 480000, 1300.0, 1450.0, 1_790_100_000, DEVICE_M4),
    ("seg-c", "20261003-3-CCCC", 2500, 1500, 7300, 470000, 1250.75, 1400.5, 1_790_200_000, DEVICE_M4),
]


def summary_of(index, spec, cum_end):
    seg_id, _, start, steps, games, positions, train_s, wall_s, started, device = spec
    return {"segment_index": index, "segment_id": seg_id, "start": "fresh" if index == 0 else "resume",
            "started_unix": started, "recorded_unix": started + int(wall_s), "start_trainer_step": start,
            "end_trainer_step": cum_end, "segment_local_step": steps, "segment_games": games,
            "segment_positions": positions, "segment_train_step_sec": train_s, "segment_wall_sec": wall_s,
            "exact_resume": index > 0, "build": BUILD, "device": device}


def record_for(index, local_step, run_id="run-1", null_totals=False, specs=SEGMENTS, schema=2):
    """The record segment `index` writes after `local_step` steps of its own."""
    seg_id, model_id, start, steps, games, positions, train_s, wall_s, started, device = specs[index]
    fraction = local_step / steps
    prior = specs[:index]
    seg_games = round(games * fraction)
    seg_positions = round(positions * fraction)
    seg_train = train_s * fraction
    seg_wall = wall_s * fraction
    history = []
    end = 0
    for i, spec in enumerate(prior):
        end = spec[2] + spec[3]
        history.append(summary_of(i, spec, end))
    return {
        "schema": schema,
        "run": {"lineage_run_id": run_id, "segment_index": index, "segment_id": seg_id,
                "segment_started_unix": started, "start": "fresh" if index == 0 else "resume",
                "exact_resume": index > 0, "not_exact_items": [],
                "continues_unrecorded_history": null_totals, "recorded_unix": started + 10 + local_step},
        "parent": None if index == 0 else {"model_id": prior[-1][1], "content_sha256": "00" * 32,
                                           "trainer_completed_steps": start, "lineage_run_id": run_id,
                                           "segment_id": prior[-1][0]},
        "steps": {"cum_trainer_step": start + local_step, "segment_start_trainer_step": start,
                  "segment_local_step": local_step},
        "fed": {"cum_games": None if null_totals else sum(s[4] for s in prior) + seg_games,
                "cum_positions": None if null_totals else sum(s[5] for s in prior) + seg_positions,
                "segment_games": seg_games, "segment_positions": seg_positions, "corpus": None},
        "time": {"cum_train_step_sec": None if null_totals else sum(s[6] for s in prior) + seg_train,
                 "cum_wall_sec": None if null_totals else sum(s[7] for s in prior) + seg_wall,
                 "segment_train_step_sec": seg_train, "segment_wall_sec": seg_wall},
        "parameters": None, "build": BUILD,
        "invocation": {"argv": ["DrewsChessMachine", "--replay-corpus", "x"], "path_kind": "replay"},
        "device": device,
        "rng": {"dropout_philox_state": None, "streams": None, "init_seed": None, "init_scheme": None},
        "segments": history,
    }


def write_header(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header)


def write_v7(directory, name, record, model_id, training_step):
    write_header(os.path.join(directory, name), {
        "dcm_format_version": "7", "model_id": model_id, "training_step": str(training_step),
        "dcm_lineage": json.dumps(record, sort_keys=True),
        # Flat mirrors, which the readers must ignore: deliberately wrong here.
        "lineage_run_id": "MIRROR-IGNORED", "cum_trainer_step": "-1"})


def write_chain(directory, specs=SEGMENTS, run_id="run-1", null_totals=False, skip_segments=()):
    """Enumerated checkpoints every 500 local steps for each segment."""
    for index, spec in enumerate(specs):
        if index in skip_segments:
            continue
        for local in range(500, spec[3] + 1, 500):
            record = record_for(index, local, run_id=run_id, null_totals=null_totals, specs=specs)
            write_v7(directory, f"{spec[0]}-replay-step{local}.safetensors", record, spec[1], local)


def local_date(unix):
    return datetime.datetime.fromtimestamp(unix).strftime("%Y%m%d")


class SwiftMirrorTests(unittest.TestCase):
    """The Python constants mirror the app's; a change on one side must be made on both."""

    def test_constants_match_the_app(self):
        with open(os.path.join(SWIFT, "Network", "ArchitectureFormat.swift")) as handle:
            fmt = handle.read()
        with open(os.path.join(SWIFT, "Persistence", "LineageRecord.swift")) as handle:
            rec = handle.read()
        self.assertEqual(int(re.search(r"static let lineageRequiredFromVersion = (\d+)", fmt).group(1)),
                         dcm_lineage.LINEAGE_REQUIRED_FROM_VERSION)
        self.assertEqual(int(re.search(r"static let unversionedLegacyVersion = (\d+)", fmt).group(1)),
                         dcm_lineage.UNVERSIONED_LEGACY_VERSION)
        self.assertEqual(int(re.search(r"static let currentSchema = (\d+)", rec).group(1)),
                         dcm_lineage.SUPPORTED_SCHEMA)
        self.assertEqual(re.search(r'static let metadataKey = "([^"]+)"', rec).group(1), dcm_lineage.METADATA_KEY)

    def test_step_reading_constants_match_the_app(self):
        with open(os.path.join(SWIFT, "Network", "ArchitectureFormat.swift")) as handle:
            fmt = handle.read()
        with open(os.path.join(SWIFT, "Persistence", "ModelCheckpointFile.swift")) as handle:
            meta = handle.read()
        with open(os.path.join(SWIFT, "Persistence", "SessionSaveTrigger.swift")) as handle:
            trigger = handle.read()
        with open(os.path.join(SWIFT, "Persistence", "ModelFileStepReading.swift")) as handle:
            reading = handle.read()
        self.assertEqual(int(re.search(r"static let trainingStepIsTrainerStepFromVersion = (\d+)", fmt).group(1)),
                         dcm_lineage.TRAINING_STEP_IS_TRAINER_STEP_FROM_VERSION)
        import dcm_arch
        self.assertEqual(int(re.search(r"static let currentVersion = (\d+)", fmt).group(1)),
                         dcm_arch.CURRENT_FORMAT_VERSION)
        replay = re.search(r'static let corpusReplayCreator = "([^"]+)"', meta).group(1)
        vsuci = re.search(r'static let trainVsUciCreator = "([^"]+)"', meta).group(1)
        self.assertEqual({replay, vsuci}, dcm_lineage.SEGMENT_STEP_CREATORS)
        gui_literal = re.search(r"static let guiCreators: Set<String> = \[([^\]]+)\]", meta).group(1)
        promotion = re.search(r'static let promotionDiskTag = "([^"]+)"', trigger).group(1)
        gui = set(re.findall(r'"([^"]+)"', gui_literal))
        if "SessionSaveTrigger.promotionDiskTag" in gui_literal:
            gui.add(promotion)
        self.assertEqual(gui, dcm_lineage.GUI_CREATORS)
        raw_values = dict(re.findall(r'case (\w+) = "([^"]+)"', reading))
        self.assertEqual(raw_values, {"trainerStep": dcm_lineage.BASIS_TRAINER_STEP,
                                      "legacySegmentStep": dcm_lineage.BASIS_LEGACY_SEGMENT_STEP,
                                      "legacyGUITrainerStep": dcm_lineage.BASIS_LEGACY_GUI_TRAINER_STEP,
                                      "legacyUnknownWriter": dcm_lineage.BASIS_LEGACY_UNKNOWN_WRITER})

    def test_session_trainer_filename_matches_the_app(self):
        with open(os.path.join(SWIFT, "Persistence", "SessionCheckpointFile.swift")) as handle:
            layout = handle.read()
        self.assertEqual(re.search(r'static let trainerFilename = "([^"]+)"', layout).group(1),
                         dcm_lineage.SESSION_TRAINER_FILENAME)


class LineageReadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_three_segment_chain_derives_the_expected_registry_segments(self):
        write_chain(self.dir)
        recorded, unrecorded, errors = dcm_lineage.scan(self.dir)
        self.assertEqual((len(recorded), unrecorded, errors), (2 + 3 + 3, {}, {}))
        table = dcm_lineage.segment_table(dcm_lineage.derive_runs(recorded))
        self.assertEqual(list(table), ["run-1"])
        expected = [
            dict(segment_index=0, source="file", lineage_run_id="run-1", segment_id="seg-a",
                 model_id="20261003-1-AAAA", date=local_date(1_790_000_000), cumstep_base=0, games_base=0,
                 elapsed_base_sec=0.0, wall_base_sec=0.0, device="M5 (VM)", unrecorded=[],
                 files=["seg-a-replay-step500.safetensors", "seg-a-replay-step1000.safetensors"]),
            dict(segment_index=1, source="file", lineage_run_id="run-1", segment_id="seg-b",
                 model_id="20261003-2-BBBB", date=local_date(1_790_100_000), cumstep_base=1000, games_base=5000,
                 elapsed_base_sec=900.5, wall_base_sec=1000.25, device="M4 Pro", unrecorded=[],
                 files=[f"seg-b-replay-step{n}.safetensors" for n in (500, 1000, 1500)]),
            dict(segment_index=2, source="file", lineage_run_id="run-1", segment_id="seg-c",
                 model_id="20261003-3-CCCC", date=local_date(1_790_200_000), cumstep_base=2500,
                 games_base=12400, elapsed_base_sec=2200.5, wall_base_sec=2450.25, device="M4 Pro",
                 unrecorded=[], files=[f"seg-c-replay-step{n}.safetensors" for n in (500, 1000, 1500)]),
        ]
        segments = table["run-1"]["segments"]
        for got, want in zip(segments, expected):
            for key, value in want.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(got[key], value, places=6, msg=key)
                else:
                    self.assertEqual(got[key], value, msg=key)
        self.assertEqual(len(segments), 3)
        self.assertEqual(table["run-1"]["latest_file"], "seg-c-replay-step1500.safetensors")
        self.assertFalse(table["run-1"]["continues_unrecorded_history"])

    def test_bases_plus_segment_totals_reproduce_every_files_cumulative_values(self):
        write_chain(self.dir)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        runs = dcm_lineage.derive_runs(recorded)
        bases = {s.fields["segment_id"]: s.fields for s in runs["run-1"].segments}
        for file in recorded:
            record = file.record
            base = bases[record["run"]["segment_id"]]
            self.assertEqual(base["cumstep_base"] + record["steps"]["segment_local_step"],
                             record["steps"]["cum_trainer_step"])
            self.assertEqual(base["games_base"] + record["fed"]["segment_games"], record["fed"]["cum_games"])
            self.assertAlmostEqual(base["elapsed_base_sec"] + record["time"]["segment_train_step_sec"],
                                   record["time"]["cum_train_step_sec"], places=6)

    def test_segment_known_only_from_history_is_derived_from_the_later_record(self):
        write_chain(self.dir, skip_segments=(0, 1))
        recorded, _, _ = dcm_lineage.scan(self.dir)
        segments = dcm_lineage.derive_runs(recorded)["run-1"].segments
        self.assertEqual([(s.segment_index, s.source) for s in segments], [(0, "history"), (1, "history"), (2, "file")])
        seg_b = segments[1]
        self.assertEqual((seg_b.fields["cumstep_base"], seg_b.fields["games_base"]), (1000, 5000))
        self.assertAlmostEqual(seg_b.fields["elapsed_base_sec"], 900.5, places=6)
        self.assertNotIn("model_id", seg_b.fields)
        self.assertIn("model_id", seg_b.unrecorded)

    def test_null_totals_stay_unrecorded_and_are_never_filled(self):
        write_chain(self.dir, null_totals=True)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        run = dcm_lineage.derive_runs(recorded)["run-1"]
        self.assertTrue(run.continues_unrecorded_history)
        for segment in run.segments:
            for field in ("games_base", "elapsed_base_sec", "wall_base_sec"):
                self.assertNotIn(field, segment.fields)
                self.assertIn(field, segment.unrecorded)
            # The trainer clock is still recorded, so the step base is.
            self.assertIn("cumstep_base", segment.fields)

    def test_older_files_are_unrecorded_not_reconstructed(self):
        write_header(os.path.join(self.dir, "v6.safetensors"), {"dcm_format_version": "6", "model_id": "m6"})
        write_header(os.path.join(self.dir, "legacy.safetensors"), {"model_id": "m3"})
        recorded, unrecorded, errors = dcm_lineage.scan(self.dir)
        self.assertEqual((recorded, unrecorded, errors), ([], {"legacy.safetensors": 3, "v6.safetensors": 6}, {}))

    def test_refusals(self):
        write_header(os.path.join(self.dir, "missing.safetensors"), {"dcm_format_version": "7", "model_id": "m"})
        bad_schema = record_for(0, 500, schema=3)
        write_v7(self.dir, "schema.safetensors", bad_schema, "m", 500)
        no_key = record_for(0, 500)
        del no_key["time"]["cum_wall_sec"]
        write_v7(self.dir, "nokey.safetensors", no_key, "m", 500)
        write_header(os.path.join(self.dir, "old-with-record.safetensors"),
                     {"dcm_format_version": "6", "dcm_lineage": "{}"})
        write_header(os.path.join(self.dir, "badversion.safetensors"), {"dcm_format_version": "seven"})
        with open(os.path.join(self.dir, "huge.safetensors"), "wb") as handle:
            handle.write(struct.pack("<Q", dcm_lineage.MAX_HEADER_BYTES + 1))
        _, _, errors = dcm_lineage.scan(self.dir)
        self.assertEqual(sorted(errors), sorted(["missing.safetensors", "schema.safetensors", "nokey.safetensors",
                                                 "old-with-record.safetensors", "badversion.safetensors",
                                                 "huge.safetensors"]))
        self.assertIn("time.cum_wall_sec", errors["nokey.safetensors"])

    def test_files_of_one_segment_that_disagree_are_refused(self):
        write_chain(self.dir)
        record = record_for(1, 1500)
        record["steps"]["segment_start_trainer_step"] = 1001
        record["steps"]["cum_trainer_step"] = 2501
        write_v7(self.dir, "seg-b-replay-step1500.safetensors", record, "20261003-2-BBBB", 1500)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.derive_runs(recorded)

    def test_device_label(self):
        self.assertEqual(dcm_lineage.device_label(DEVICE_M4), "M4 Pro")
        self.assertEqual(dcm_lineage.device_label(DEVICE_VM), "M5 (VM)")
        self.assertIsNone(dcm_lineage.device_label(dict(DEVICE_M4, chip=None)))
        self.assertIsNone(dcm_lineage.device_label(dict(DEVICE_M4, is_vm=None)))

    def test_checkpoint_facts(self):
        write_chain(self.dir)
        facts = dcm_lineage.checkpoint_facts(os.path.join(self.dir, "seg-c-replay-step1000.safetensors"))
        self.assertEqual((facts["segment_id"], facts["segment_local_step"], facts["cum_trainer_step"],
                          facts["run_origin"], facts["cum_games"]), ("seg-c", 1000, 3500, 0, 12400 + round(7300 * 2 / 3)))
        write_header(os.path.join(self.dir, "v6.safetensors"), {"dcm_format_version": "6"})
        self.assertIsNone(dcm_lineage.checkpoint_facts(os.path.join(self.dir, "v6.safetensors")))


def registry_with(segments):
    return {"models_dir": "~/x", "logs_dir": "~/y",
            "runs": {"r": {"label": "r", "out_model": "seg-c-replay-latest.safetensors", "segments": segments}}}


class RegistryPlanTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write_chain(self.dir)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        self.runs = dcm_lineage.derive_runs(recorded)

    def test_fills_missing_fields_and_matches_by_model_id(self):
        reg = registry_with([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0},
                             {"log": "b.txt", "model_id": "20261003-2-BBBB"},
                             {"log": "c.txt", "model_id": "20261003-3-CCCC"}])
        before = copy.deepcopy(reg)
        updated, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual(reg, before, "plan must not mutate its input")
        self.assertEqual(report["conflicts"], [])
        seg_c = updated["runs"]["r"]["segments"][2]
        self.assertEqual((seg_c["lineage_run_id"], seg_c["segment_id"], seg_c["cumstep_base"], seg_c["games_base"],
                          seg_c["device"]), ("run-1", "seg-c", 2500, 12400, "M4 Pro"))
        self.assertEqual(seg_c["log"], "c.txt")
        self.assertIn(("r", 0, "cumstep_base", 0, 0), report["same"])

    def test_a_differing_hand_value_is_a_conflict_and_is_left_alone(self):
        reg = registry_with([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0},
                             {"log": "b.txt", "model_id": "20261003-2-BBBB", "games_base": 4999}])
        updated, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual(report["conflicts"], [("r", 1, "games_base", 4999, 5000)])
        self.assertEqual(updated["runs"]["r"]["segments"][1]["games_base"], 4999)
        self.assertEqual([(s["segment_index"], s["segment_id"]) for _, s in report["unmatched_segments"]],
                         [(2, "seg-c")])

    def test_unmatched_run_is_proposed_not_created(self):
        reg = registry_with([{"log": "z.txt", "model_id": "unrelated", "cumstep_base": 0}])
        updated, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual(updated, reg)
        self.assertEqual(list(report["unmatched_runs"]), ["run-1"])

    def test_step_axis_anchors_on_the_registry_when_the_run_is_not_the_first_segment(self):
        reg = registry_with([{"log": "old.txt", "cumstep_base": 0},
                             {"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 70000},
                             {"log": "b.txt", "model_id": "20261003-2-BBBB"}])
        updated, report = _lineage_registry.plan(reg, self.runs)
        self.assertEqual(updated["runs"]["r"]["segments"][2]["cumstep_base"], 71000)
        self.assertEqual(report["conflicts"], [])

    def test_step_axis_without_an_anchor_stays_unrecorded(self):
        reg = registry_with([{"log": "old.txt", "cumstep_base": 0},
                             {"log": "b.txt", "model_id": "20261003-2-BBBB"}])
        updated, report = _lineage_registry.plan(reg, self.runs)
        self.assertNotIn("cumstep_base", updated["runs"]["r"]["segments"][1])
        self.assertIn(("r", 1, "cumstep_base"), report["unrecorded"])

    def test_a_lineage_run_matching_two_registry_runs_is_refused(self):
        reg = registry_with([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0}])
        reg["runs"]["other"] = {"segments": [{"log": "b.txt", "model_id": "20261003-2-BBBB"}]}
        with self.assertRaises(_lineage_registry.RegistryContradiction):
            _lineage_registry.plan(reg, self.runs)


class DeriveRegistryCommandTests(unittest.TestCase):
    """The command around `plan`: it writes only on --write, only fills, and refuses on conflict."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write_chain(self.dir)
        self.paths = sorted(os.path.join(self.dir, n) for n in os.listdir(self.dir))
        self.registry = os.path.join(self.dir, "registry.json")

    def write_registry(self, segments):
        text = json.dumps(registry_with(segments), indent=2, ensure_ascii=False)
        with open(self.registry, "w") as handle:
            handle.write(text)
        return text

    def read_registry(self):
        with open(self.registry) as handle:
            return handle.read()

    def run_command(self, write, path_kind="replay"):
        lines = []
        status = _lineage_registry.derive_registry(self.registry, self.paths, path_kind, write, out=lines.append)
        return status, lines

    def test_proposal_only_changes_nothing(self):
        before = self.write_registry([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0},
                                      {"log": "b.txt", "model_id": "20261003-2-BBBB"},
                                      {"log": "c.txt", "model_id": "20261003-3-CCCC"}])
        status, lines = self.run_command(write=False)
        self.assertEqual(status, 0)
        self.assertEqual(self.read_registry(), before)
        self.assertTrue(any("proposal only" in line for line in lines))

    def test_write_fills_and_a_second_run_finds_everything_agreeing(self):
        self.write_registry([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0},
                             {"log": "b.txt", "model_id": "20261003-2-BBBB"},
                             {"log": "c.txt", "model_id": "20261003-3-CCCC"}])
        status, _ = self.run_command(write=True)
        self.assertEqual(status, 0)
        written = json.loads(self.read_registry())["runs"]["r"]["segments"]
        self.assertEqual([s["segment_id"] for s in written], ["seg-a", "seg-b", "seg-c"])
        self.assertEqual([s["cumstep_base"] for s in written], [0, 1000, 2500])
        self.assertEqual(written[1]["log"], "b.txt")
        after_first = self.read_registry()
        status, lines = self.run_command(write=True)
        self.assertEqual(status, 0)
        self.assertEqual(self.read_registry(), after_first)
        self.assertTrue(any("nothing to write" in line for line in lines))

    def test_conflict_refuses_the_whole_write(self):
        before = self.write_registry([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0},
                                      {"log": "b.txt", "model_id": "20261003-2-BBBB", "games_base": 1},
                                      {"log": "c.txt", "model_id": "20261003-3-CCCC"}])
        status, lines = self.run_command(write=True)
        self.assertEqual(status, 1)
        self.assertEqual(self.read_registry(), before)
        self.assertTrue(any(line.startswith("  CONFLICT r seg 1: games_base") for line in lines))

    def test_other_path_kinds_are_ignored(self):
        before = self.write_registry([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0}])
        status, lines = self.run_command(write=True, path_kind="vsuci")
        self.assertEqual((status, self.read_registry()), (0, before))
        self.assertTrue(lines[0].startswith(f"scanned {len(self.paths)} file(s): 0 with a vsuci lineage record"))


class SessionFolderTests(unittest.TestCase):
    """Train-vs-UCI and GUI runs save .dcmsession folders, each holding a
    trainer.safetensors and a champion.safetensors: the scanners read them beside
    top-level step files, and name each one by its folder."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # Two vs-UCI session saves of segment 0, then top-level step files of segment 1.
        for local, stamp in ((500, "20261003-010000"), (1000, "20261003-020000")):
            record = record_for(0, local)
            record["invocation"]["path_kind"] = "vsuci"
            folder = os.path.join(self.dir, f"{stamp}-20261003-1-AAAA-vsuci-periodic.dcmsession")
            os.mkdir(folder)
            for name in ("trainer.safetensors", "champion.safetensors"):
                write_v7(folder, name, record, SEGMENTS[0][1], local)
        for local in (500, 1000):
            record = record_for(1, local)
            record["invocation"]["path_kind"] = "vsuci"
            write_v7(self.dir, f"seg-b-vsuci-step{local}.safetensors", record, SEGMENTS[1][1], local)
        # Neither is read: a folder that is not a session, and a staging folder.
        os.mkdir(os.path.join(self.dir, "notes"))
        write_v7(os.path.join(self.dir, "notes"), "stray.safetensors", record_for(0, 500), SEGMENTS[0][1], 500)
        staging = os.path.join(self.dir, "20261003-030000-20261003-1-AAAA-vsuci-final.dcmsession.tmp")
        os.mkdir(staging)
        write_v7(staging, "trainer.safetensors", record_for(0, 1000), SEGMENTS[0][1], 1000)

    def test_model_paths_cover_top_level_files_and_session_folders_only(self):
        names = [dcm_lineage.display_name(p) for p in dcm_lineage.model_paths(self.dir)]
        self.assertEqual(names, [
            "20261003-010000-20261003-1-AAAA-vsuci-periodic.dcmsession/champion.safetensors",
            "20261003-010000-20261003-1-AAAA-vsuci-periodic.dcmsession/trainer.safetensors",
            "20261003-020000-20261003-1-AAAA-vsuci-periodic.dcmsession/champion.safetensors",
            "20261003-020000-20261003-1-AAAA-vsuci-periodic.dcmsession/trainer.safetensors",
            "seg-b-vsuci-step1000.safetensors",
            "seg-b-vsuci-step500.safetensors",
        ])

    def test_session_files_derive_segments_named_by_their_folder(self):
        recorded, unrecorded, errors = dcm_lineage.scan(self.dir)
        self.assertEqual((len(recorded), unrecorded, errors), (6, {}, {}))
        table = dcm_lineage.segment_table(dcm_lineage.derive_runs(recorded))
        segments = table["run-1"]["segments"]
        self.assertEqual([s["segment_id"] for s in segments], ["seg-a", "seg-b"])
        self.assertIn("20261003-020000-20261003-1-AAAA-vsuci-periodic.dcmsession/trainer.safetensors",
                      segments[0]["files"])
        self.assertEqual(segments[1]["cumstep_base"], 1000)

    def test_derive_registry_reads_session_folders_for_vsuci(self):
        registry = os.path.join(tempfile.mkdtemp(), "vsuci_registry.json")
        with open(registry, "w") as handle:
            handle.write(json.dumps(registry_with([{"log": "a.txt", "model_id": "20261003-1-AAAA", "cumstep_base": 0},
                                                   {"log": "b.txt", "model_id": "20261003-2-BBBB"}]), indent=2))
        lines = []
        status = _lineage_registry.derive_registry(registry, dcm_lineage.model_paths(self.dir), "vsuci", True,
                                                   out=lines.append)
        self.assertEqual(status, 0, lines)
        with open(registry) as handle:
            written = json.load(handle)["runs"]["r"]["segments"]
        self.assertEqual([s["segment_id"] for s in written], ["seg-a", "seg-b"])
        self.assertEqual([s["cumstep_base"] for s in written], [0, 1000])


def gui_record(local_step, segment_index=0, specs=SEGMENTS):
    record = record_for(segment_index, local_step, specs=specs)
    record["invocation"]["path_kind"] = "gui"
    return record


class GuiSessionLineageTests(unittest.TestCase):
    """A GUI session folder holds trainer.safetensors (the trainer generation's ID) and
    champion.safetensors (the champion's ID), and the trainer ID changes at every
    promotion within a segment: a GUI segment has several model IDs by design."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def session(self, stamp, files):
        folder = os.path.join(self.dir, f"{stamp}-20261003-1-AAAA-periodic.dcmsession")
        os.mkdir(folder)
        for name, record, model_id, step in files:
            write_v7(folder, name, record, model_id, step)
        return folder

    def test_gui_session_folder_with_differing_champion_and_trainer_ids_derives(self):
        record = gui_record(500)
        self.session("20261003-010000", [("trainer.safetensors", record, "20261003-1-AAAA-4", 500),
                                         ("champion.safetensors", record, "20261003-1-AAAA-3", 500)])
        recorded, _, errors = dcm_lineage.scan(self.dir)
        self.assertEqual(errors, {})
        segment = dcm_lineage.segment_table(dcm_lineage.derive_runs(recorded))["run-1"]["segments"][0]
        self.assertEqual(segment["model_ids"], ["20261003-1-AAAA-3", "20261003-1-AAAA-4"])
        self.assertNotIn("model_id", segment)
        self.assertEqual((segment["segment_id"], segment["cumstep_base"]), ("seg-a", 0))

    def test_gui_segment_whose_trainer_id_changes_at_a_promotion_derives(self):
        self.session("20261003-010000", [("trainer.safetensors", gui_record(500), "20261003-1-AAAA-4", 500)])
        self.session("20261003-020000", [("trainer.safetensors", gui_record(1000), "20261003-1-AAAA-6", 1000)])
        recorded, _, _ = dcm_lineage.scan(self.dir)
        run = dcm_lineage.derive_runs(recorded)["run-1"]
        self.assertEqual(run.segments[0].fields["model_ids"], ["20261003-1-AAAA-4", "20261003-1-AAAA-6"])
        self.assertEqual(len(run.segments[0].files), 2)

    def test_a_gui_file_without_a_model_id_is_refused(self):
        folder = self.session("20261003-010000", [("trainer.safetensors", gui_record(500), "20261003-1-AAAA-4", 500)])
        write_header(os.path.join(folder, "champion.safetensors"), {
            "dcm_format_version": "7", "training_step": "500",
            "dcm_lineage": json.dumps(gui_record(500), sort_keys=True)})
        recorded, _, _ = dcm_lineage.scan(self.dir)
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.derive_runs(recorded)

    def test_replay_and_vsuci_segments_keep_one_model_id(self):
        for path_kind in ("replay", "vsuci"):
            directory = tempfile.mkdtemp()
            for local, model_id in ((500, "20261003-1-AAAA"), (1000, "20261003-1-ZZZZ")):
                record = record_for(0, local)
                record["invocation"]["path_kind"] = path_kind
                write_v7(directory, f"seg-a-step{local}.safetensors", record, model_id, local)
            recorded, _, _ = dcm_lineage.scan(directory)
            with self.assertRaises(dcm_lineage.LineageError, msg=path_kind):
                dcm_lineage.derive_runs(recorded)

    def test_a_segment_written_by_two_path_kinds_is_refused(self):
        self.session("20261003-010000", [("trainer.safetensors", gui_record(500), "20261003-1-AAAA", 500)])
        write_v7(self.dir, "seg-a-replay-step1000.safetensors", record_for(0, 1000), "20261003-1-AAAA", 1000)
        recorded, _, _ = dcm_lineage.scan(self.dir)
        with self.assertRaises(dcm_lineage.LineageError) as caught:
            dcm_lineage.derive_runs(recorded)
        self.assertIn("path_kind", str(caught.exception))

    def test_lineage_table_reads_trainer_files_only(self):
        import contextlib
        import io
        import selfplay
        # The champion file's record describes another run (the file the champion came
        # from), which is not this run's progress and must not join its segments.
        other = gui_record(1000, segment_index=1)
        other["run"]["lineage_run_id"] = "run-earlier"
        self.session("20261003-010000", [("trainer.safetensors", gui_record(500), "20261003-1-AAAA-4", 500),
                                         ("champion.safetensors", other, "20261003-1-AAAA-3", 1000)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = selfplay.print_lineage_table(self.dir)
        text = out.getvalue()
        self.assertEqual(status, 0, text)
        self.assertTrue(text.startswith("scanned 1 session trainer file(s)"), text)
        table = json.loads(text[text.index("{"):])
        self.assertEqual(list(table), ["run-1"])
        self.assertEqual(table["run-1"]["segments"][0]["model_ids"], ["20261003-1-AAAA-4"])


class TrackerCellTests(unittest.TestCase):
    """replay.py's lineage cells and header-identified checkpoints, on a temporary registry."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = cls.folder.name
        write_chain(root)
        write_header(os.path.join(root, "old-replay-step1000.safetensors"),
                     {"dcm_format_version": "6", "model_id": "old", "training_step": "1000"})
        with open(os.path.join(root, "registry.json"), "w") as handle:
            json.dump({"models_dir": root, "logs_dir": root, "runs": {}}, handle)
        os.environ["DCM_DASH_ROOT"] = root
        import importlib
        import replay
        cls.replay = importlib.reload(replay)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DCM_DASH_ROOT")
        cls.folder.cleanup()

    def test_cells_come_from_the_record(self):
        path = os.path.join(self.root, "seg-b-replay-step1000.safetensors")
        cells = self.replay.lineage_cells(path, {"segment_id": "seg-b", "games_base": 5000}, 1000)
        self.assertEqual(cells["games_fed"], 5000 + round(7400 * 1000 / 1500))
        self.assertAlmostEqual(cells["train_step_sec"], 900.5 + 1300.0 * 1000 / 1500, places=3)

    def test_a_file_without_a_record_gives_no_cells(self):
        path = os.path.join(self.root, "old-replay-step1000.safetensors")
        self.assertEqual(self.replay.lineage_cells(path, {}, 1000), {})

    def test_cells_refuse_a_file_filed_under_the_wrong_segment_or_step(self):
        path = os.path.join(self.root, "seg-b-replay-step1000.safetensors")
        with self.assertRaises(dcm_lineage.LineageError):
            self.replay.lineage_cells(path, {"segment_id": "seg-c"}, 1000)
        with self.assertRaises(dcm_lineage.LineageError):
            self.replay.lineage_cells(path, {"segment_id": "seg-b"}, 999)
        with self.assertRaises(dcm_lineage.LineageError):
            self.replay.lineage_cells(path, {"segment_id": "seg-b", "games_base": 4000}, 1000)

    def test_checkpoints_are_found_by_segment_id_and_skipped_by_enum_specs(self):
        cfg = {"out_model": "seg-c-replay-latest.safetensors",
               "segments": [{"cumstep_base": 0, "enum_stem": "seg-a"},
                            {"cumstep_base": 1000, "segment_id": "seg-b"},
                            {"cumstep_base": 2500, "segment_id": "seg-c"}]}
        found = self.replay.lineage_checkpoints(cfg)
        self.assertEqual([(si, os.path.basename(p), n) for si, p, n in found],
                         [(1, "seg-b-replay-step500.safetensors", 500), (1, "seg-b-replay-step1000.safetensors", 1000),
                          (1, "seg-b-replay-step1500.safetensors", 1500), (2, "seg-c-replay-step500.safetensors", 500),
                          (2, "seg-c-replay-step1000.safetensors", 1000), (2, "seg-c-replay-step1500.safetensors", 1500)])
        self.assertEqual(self.replay.enum_specs(cfg), [(0, "seg-a-replay-step*.safetensors")])


def v11_header(local_step=487, cum=2000, creator="replay", stated=None, clock=True, model_id="20261006-2-TTTT"):
    """A format v11 corpus-replay header of segment 1 (seg-b) `local_step` steps in, at
    trainer step `cum`: `training_step` is the trainer step (unless `stated` says
    otherwise) and the record's `segment_local_step` the segment step."""
    record = record_for(1, 500)
    record["steps"] = {"cum_trainer_step": cum, "segment_start_trainer_step": cum - local_step,
                       "segment_local_step": local_step}
    header = {"dcm_format_version": "11", "model_id": model_id, "creator": creator,
              "training_step": str(cum if stated is None else stated),
              "dcm_lineage": json.dumps(record, sort_keys=True)}
    if clock:
        header["trainer_completed_steps"] = str(cum)
    return header


class StepReadingTests(unittest.TestCase):
    """`dcm_lineage.step_reading`, the mirror of the app's ModelFileStepReading: from format
    v11 `training_step` is the trainer step; before v11 it is read by the file's `creator`."""

    def test_a_v11_header_states_the_trainer_step_with_the_segment_step_as_the_sidecar(self):
        reading = dcm_lineage.step_reading(v11_header(), "v11")
        self.assertEqual((reading.basis, reading.stated_training_step, reading.trainer_step, reading.segment_step),
                         (dcm_lineage.BASIS_TRAINER_STEP, 2000, 2000, 487))
        self.assertIsNone(reading.legacy_note)

    def test_a_v11_trainer_header_whose_step_is_not_its_clock_is_refused(self):
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.step_reading(v11_header(stated=487), "bad")
        header = v11_header()
        del header["training_step"]
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.step_reading(header, "bad")

    def test_a_negative_trainer_clock_is_refused(self):
        for header in (v11_header(), {"dcm_format_version": "10", "model_id": "m", "creator": "replay",
                                      "training_step": "5"}):
            header = dict(header, trainer_completed_steps="-5")
            if header["dcm_format_version"] == "11":
                header["training_step"] = "-5"
            with self.subTest(version=header["dcm_format_version"]), \
                    self.assertRaisesRegex(dcm_lineage.LineageError, "below 0"):
                dcm_lineage.step_reading(header, "negative")

    def test_trainer_schedule_keys_without_the_clock_are_refused(self):
        for key in ("trainer_lr_warmup_steps", "trainer_lr_momentum_cycle", "trainer_lr_momentum_cycle_envelope"):
            header = v11_header(clock=False)
            header[key] = "0" if key == "trainer_lr_warmup_steps" else "{}"
            with self.subTest(key=key), self.assertRaisesRegex(dcm_lineage.LineageError,
                                                               f"trainer_completed_steps is missing beside {key}"):
                dcm_lineage.step_reading(header, "half")

    def test_a_version_newer_than_the_tools_is_refused(self):
        import dcm_arch
        header = v11_header()
        header["dcm_format_version"] = str(dcm_arch.CURRENT_FORMAT_VERSION + 1)
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.step_reading(header, "future")

    def test_a_pre_v11_replay_header_reads_its_step_as_the_segment_step(self):
        header = v11_header(stated=487)
        header["dcm_format_version"] = "10"
        reading = dcm_lineage.step_reading(header, "v10")
        self.assertEqual((reading.basis, reading.trainer_step, reading.segment_step),
                         (dcm_lineage.BASIS_LEGACY_SEGMENT_STEP, 2000, 487))
        self.assertIn("training_step 487 is the writing segment's step", reading.legacy_note)
        self.assertIn("trainer step 2000 (trainer_completed_steps)", reading.legacy_note)

    def test_a_pre_v11_plain_replay_header_takes_the_records_total_and_a_schedule_less_v3_one_none(self):
        header = v11_header(stated=487, clock=False)
        header["dcm_format_version"] = "9"
        reading = dcm_lineage.step_reading(header, "v9")
        self.assertEqual((reading.trainer_step, reading.segment_step), (2000, 487))
        v3 = {"model_id": "m", "creator": "replay", "training_step": "41000"}
        reading = dcm_lineage.step_reading(v3, "v3")
        self.assertIsNone(reading.trainer_step)
        self.assertEqual((reading.segment_step, reading.trainer_step_or_stated_step), (41000, 41000))

    def test_the_creator_names_the_writer_not_the_records_path_kind(self):
        header = v11_header(creator="manual", clock=False)
        header["dcm_format_version"] = "8"
        self.assertEqual(json.loads(header["dcm_lineage"])["invocation"]["path_kind"], "replay")
        reading = dcm_lineage.step_reading(header, "v8 GUI champion")
        self.assertEqual((reading.basis, reading.trainer_step), (dcm_lineage.BASIS_LEGACY_GUI_TRAINER_STEP, 2000))

    def test_a_creator_less_pre_v11_header_without_a_record_gives_its_step_as_both(self):
        reading = dcm_lineage.step_reading({"dcm_format_version": "6", "model_id": "m", "training_step": "40"}, "v6")
        self.assertEqual((reading.basis, reading.trainer_step, reading.segment_step),
                         (dcm_lineage.BASIS_LEGACY_UNKNOWN_WRITER, 40, 40))
        self.assertIn("was stated by writer ''", reading.legacy_note)

    def test_a_header_stating_no_step_is_never_flagged(self):
        reading = dcm_lineage.step_reading({"dcm_format_version": "10", "model_id": "m", "creator": "new-model"},
                                           "seed")
        self.assertIsNone(reading.stated_training_step)
        self.assertIsNone(reading.legacy_note)

    def test_a_malformed_step_is_refused(self):
        with self.assertRaises(dcm_lineage.LineageError):
            dcm_lineage.step_reading({"model_id": "m", "training_step": " 40"}, "bad")


class InventoryTests(unittest.TestCase):
    def test_inventory_reports_lineage_and_unrecorded(self):
        directory = tempfile.mkdtemp()
        write_chain(directory)
        write_header(os.path.join(directory, "v6.safetensors"),
                     {"dcm_format_version": "6", "model_id": "m6", "training_step": "7",
                      "replay_epoch": "1", "replay_next_game_index": "42"})
        entries = {os.path.basename(e["path"]): e for e in ckpt_inventory.scan([directory], False)}
        old = entries["v6.safetensors"]
        self.assertEqual((old["lineage"], old["replay_epoch"], old["replay_next_game_index"]),
                         ("unrecorded (format v6)", "1", "42"))
        new = entries["seg-b-replay-step1000.safetensors"]
        self.assertEqual((new["step_basis"], new["trainer_step"], new["segment_step"]),
                         (dcm_lineage.BASIS_LEGACY_UNKNOWN_WRITER, 1000, 1000))
        self.assertEqual(new["lineage"]["lineage_run_id"], "run-1")
        self.assertEqual(new["lineage"]["segment_index"], 1)
        self.assertEqual(new["lineage"]["cum_trainer_step"], 2000)
        self.assertNotIn("cum_trainer_step", new, "flat mirror keys are not carried")


if __name__ == "__main__":
    unittest.main()
