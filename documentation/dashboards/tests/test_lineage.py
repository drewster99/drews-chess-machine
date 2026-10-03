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


def record_for(index, local_step, run_id="run-1", null_totals=False, specs=SEGMENTS, schema=1):
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
        "rng": {"seed_mode": "unseeded", "dropout_philox_state": None},
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
        bad_schema = record_for(0, 500, schema=2)
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
        self.assertEqual(new["lineage"]["lineage_run_id"], "run-1")
        self.assertEqual(new["lineage"]["segment_index"], 1)
        self.assertEqual(new["lineage"]["cum_trainer_step"], 2000)
        self.assertNotIn("cum_trainer_step", new, "flat mirror keys are not carried")


if __name__ == "__main__":
    unittest.main()
