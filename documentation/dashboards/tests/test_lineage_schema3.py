"""Tests for lineage schema 3 in scripts/dcm_lineage.py (hyperparameter recording
plan P4): the supported schema range, the keys schema 3 added (required at 3,
refused at 2), the corpus shapes, and `weights_totals` (plan B4, O-15).

Run: python3 -m unittest discover -s documentation/dashboards/tests
"""
import copy
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)
import dcm_lineage  # noqa: E402
import test_lineage  # noqa: E402  the schema-2 record builder

SWIFT = os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachine")
SWIFT_TESTS = os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachineTests")

SCHEMA_3_BUILD_EXTRAS = {"git_diff_sha256": {"recorded": True, "value": None},
                         "xcode_build": {"recorded": True, "value": "17A5241e"},
                         "sdk_build": {"recorded": True, "value": "26A5300a"},
                         "configuration": {"recorded": True, "value": "Release"}}
SUMMARY_EXTRAS = {"configuration": {"recorded": False}, "parameters": {"recorded": False},
                  "corpus_identity": {"recorded": False}, "segment_start_corpus": {"recorded": False},
                  "path_kind": {"recorded": False}, "argv": {"recorded": False},
                  "run_seeds": {"recorded": False}}


def schema_three(record, ancestry=None):
    """`record` (a schema-2 record from `test_lineage.record_for`) in the
    schema-3 shape: the three top-level keys, the build identity and every
    summary's added fields."""
    upgraded = copy.deepcopy(record)
    upgraded["schema"] = 3
    upgraded["configuration"] = {"recorded": False}
    upgraded["run_seeds"] = {"recorded": False}
    upgraded["ancestry"] = ancestry or {"history_before_oldest_run": "none", "runs": []}
    upgraded["build"] = dict(upgraded["build"], **SCHEMA_3_BUILD_EXTRAS)
    upgraded["segments"] = [dict(summary, **SUMMARY_EXTRAS) for summary in upgraded["segments"]]
    return upgraded


def totals(step, games, positions, train_sec, wall_sec):
    return {"cum_trainer_step": step, "cum_games": games, "cum_positions": positions,
            "cum_train_step_sec": train_sec, "cum_wall_sec": wall_sec}


def ancestor(run_id, left_by, departure):
    return {"lineage_run_id": run_id, "left_by": left_by,
            "left_at": {"model_id": f"{run_id}-model", "content_sha256": None,
                        "trainer_completed_steps": departure["cum_trainer_step"]},
            "totals_at_departure": departure, "architecture_at_departure": None,
            "initialization": {"recorded": True, "value": None}, "segments": []}


def own_totals(step, games, positions, train_sec, wall_sec, history="none", runs=()):
    """A record carrying only what `weights_totals` reads."""
    return {"steps": {"cum_trainer_step": step}, "fed": {"cum_games": games, "cum_positions": positions},
            "time": {"cum_train_step_sec": train_sec, "cum_wall_sec": wall_sec},
            "ancestry": {"history_before_oldest_run": history, "runs": list(runs)}}


class SchemaRangeTests(unittest.TestCase):
    def test_oldest_supported_schema_matches_the_app(self):
        with open(os.path.join(SWIFT, "Persistence", "LineageRecord.swift")) as handle:
            rec = handle.read()
        self.assertEqual(int(re.search(r"static let oldestDecodableSchema = (\d+)", rec).group(1)),
                         dcm_lineage.OLDEST_SUPPORTED_SCHEMA)
        self.assertEqual(int(re.search(r"static let currentSchema = (\d+)", rec).group(1)),
                         dcm_lineage.SUPPORTED_SCHEMA)

    def test_a_schema_above_the_supported_range_is_refused(self):
        record = schema_three(test_lineage.record_for(1, 500))
        record["schema"] = dcm_lineage.SUPPORTED_SCHEMA + 1
        with self.assertRaisesRegex(dcm_lineage.LineageError, "not a supported schema"):
            dcm_lineage.validated_record(record, "above.safetensors")
        record["schema"] = dcm_lineage.OLDEST_SUPPORTED_SCHEMA - 1
        with self.assertRaisesRegex(dcm_lineage.LineageError, "not a supported schema"):
            dcm_lineage.validated_record(record, "below.safetensors")

    def test_both_supported_schemas_are_read(self):
        schema_two = test_lineage.record_for(1, 500)
        self.assertIs(dcm_lineage.validated_record(schema_two, "two.safetensors"), schema_two)
        schema_3 = schema_three(schema_two)
        self.assertIs(dcm_lineage.validated_record(schema_3, "three.safetensors"), schema_3)

    def test_the_real_schema_two_fixtures_are_read(self):
        """The two real records the Swift tests pin (`LineageSchemaTwoFixtures`)
        are read here too, so both readers accept the same files."""
        with open(os.path.join(SWIFT_TESTS, "LineageSchemaTwoFixtures.swift")) as handle:
            source = handle.read()
        literals = re.findall(r'static let \w+ = #"(.*?)"#', source)
        self.assertEqual(len(literals), 2)
        import json
        for text in literals:
            record = json.loads(text)
            self.assertEqual(record["schema"], 2)
            dcm_lineage.validated_record(record, "fixture")
            self.assertEqual(dcm_lineage.corpus_ids(record["fed"]["corpus"]), ["20260624-192615-w3aA5b"])
            self.assertIsNone(dcm_lineage.weights_totals(record), "a schema-2 record has no ancestry")


class SchemaThreeKeyTests(unittest.TestCase):
    def test_schema_three_summary_keys_are_required(self):
        record = schema_three(test_lineage.record_for(1, 500))
        self.assertEqual(len(record["segments"]), 1)
        for key in SUMMARY_EXTRAS:
            broken = copy.deepcopy(record)
            del broken["segments"][0][key]
            with self.assertRaisesRegex(dcm_lineage.LineageError, f"segments\\[0\\] has no {key}"):
                dcm_lineage.validated_record(broken, "s.safetensors")

    def test_schema_three_top_level_and_build_keys_are_required(self):
        record = schema_three(test_lineage.record_for(0, 500))
        for key in ("configuration", "run_seeds", "ancestry"):
            broken = copy.deepcopy(record)
            del broken[key]
            with self.assertRaisesRegex(dcm_lineage.LineageError, f"has no {key}"):
                dcm_lineage.validated_record(broken, "t.safetensors")
        for key in SCHEMA_3_BUILD_EXTRAS:
            broken = copy.deepcopy(record)
            del broken["build"][key]
            with self.assertRaisesRegex(dcm_lineage.LineageError, f"build has no {key}"):
                dcm_lineage.validated_record(broken, "b.safetensors")

    def test_a_schema_two_record_carrying_a_schema_three_key_is_refused(self):
        record = test_lineage.record_for(1, 500)
        record["ancestry"] = {"history_before_oldest_run": "none", "runs": []}
        with self.assertRaisesRegex(dcm_lineage.LineageError, "never wrote"):
            dcm_lineage.validated_record(record, "mixed.safetensors")
        record = test_lineage.record_for(1, 500)
        record["segments"][0]["argv"] = {"recorded": False}
        with self.assertRaisesRegex(dcm_lineage.LineageError, "never wrote"):
            dcm_lineage.validated_record(record, "mixed-summary.safetensors")


SCHEMA_2_CORPUS = {"corpus_id": "a", "corpus_path": "/a", "epoch": 0}
SCHEMA_3_CORPUS = {"corpus_identity": {"first_only": {"corpus_id": "a", "corpus_path": "/a"}},
                   "segment_start": {"recorded": False}, "epoch": 0}


class CorpusPositionSchemaTests(unittest.TestCase):
    """`validated_record` checks a corpus position's keys by schema, as CorpusPosition's decoder does."""

    def record(self, schema, corpus):
        record = test_lineage.record_for(1, 500)
        if schema == 3:
            record = schema_three(record)
        record["fed"]["corpus"] = copy.deepcopy(corpus)
        return record

    def test_each_schemas_corpus_position_is_read(self):
        dcm_lineage.validated_record(self.record(2, SCHEMA_2_CORPUS), "two")
        dcm_lineage.validated_record(self.record(3, SCHEMA_3_CORPUS), "three")

    def test_a_schema_three_position_with_schema_two_keys_is_refused(self):
        for key in ("corpus_id", "corpus_path"):
            corpus = dict(SCHEMA_3_CORPUS, **{key: SCHEMA_2_CORPUS[key]})
            with self.subTest(key=key), self.assertRaisesRegex(dcm_lineage.LineageError, f"carries {key}"):
                dcm_lineage.validated_record(self.record(3, corpus), "three")

    def test_a_schema_three_position_without_its_keys_is_refused(self):
        for key in ("corpus_identity", "segment_start"):
            corpus = {k: v for k, v in SCHEMA_3_CORPUS.items() if k != key}
            with self.subTest(key=key), self.assertRaisesRegex(dcm_lineage.LineageError, f"has no {key}"):
                dcm_lineage.validated_record(self.record(3, corpus), "three")

    def test_a_schema_two_position_with_schema_three_keys_is_refused(self):
        for key in ("corpus_identity", "segment_start"):
            corpus = dict(SCHEMA_2_CORPUS, **{key: SCHEMA_3_CORPUS[key]})
            with self.subTest(key=key), self.assertRaisesRegex(dcm_lineage.LineageError, "never wrote"):
                dcm_lineage.validated_record(self.record(2, corpus), "two")

    def test_a_schema_two_position_without_its_keys_is_refused(self):
        for key in ("corpus_id", "corpus_path"):
            corpus = {k: v for k, v in SCHEMA_2_CORPUS.items() if k != key}
            with self.subTest(key=key), self.assertRaisesRegex(dcm_lineage.LineageError, f"has no {key}"):
                dcm_lineage.validated_record(self.record(2, corpus), "two")

    def test_a_corpus_position_that_is_not_an_object_is_refused(self):
        with self.assertRaisesRegex(dcm_lineage.LineageError, "not an object"):
            dcm_lineage.validated_record(self.record(2, ["a"]), "two")


class SeedOriginSchemaTests(unittest.TestCase):
    def record(self, schema, seed_origin):
        record = test_lineage.record_for(1, 500)
        if schema == 3:
            record = schema_three(record)
        record["rng"]["streams"] = {"master_seed": "1", "seed_origin": seed_origin}
        return record

    def test_a_schema_two_record_with_a_command_line_seed_origin_is_refused(self):
        with self.assertRaisesRegex(dcm_lineage.LineageError, "seed_origin command_line is a schema-3 value"):
            dcm_lineage.validated_record(self.record(2, "command_line"), "two")

    def test_other_seed_origins_are_read(self):
        dcm_lineage.validated_record(self.record(2, "configured"), "two")
        dcm_lineage.validated_record(self.record(2, "drawn"), "two")
        dcm_lineage.validated_record(self.record(3, "command_line"), "three")


class CorpusShapeTests(unittest.TestCase):
    def test_corpus_ids_read_every_shape(self):
        self.assertEqual(dcm_lineage.corpus_ids({"corpus_id": "a", "corpus_path": "/a"}), ["a"])
        self.assertEqual(dcm_lineage.corpus_ids({"corpus_identity": {"first_only": {"corpus_id": "a",
                                                                                     "corpus_path": "/a"}}}), ["a"])
        listed = {"corpus_identity": {"listed": [{"corpus_id": "a", "corpus_path": "/a", "shard_count": 3},
                                                 {"corpus_id": "b", "corpus_path": "/b", "shard_count": 1}]}}
        self.assertEqual(dcm_lineage.corpus_ids(listed), ["a", "b"])


class WeightsTotalsTests(unittest.TestCase):
    def test_weights_totals_sum_along_ancestry_and_stop_at_unrecorded(self):
        # A (100 steps) -> branch -> B (50): the weights carry both runs.
        record = own_totals(50, 5, 500, 25.0, 30.0,
                            runs=[ancestor("A", "branch", totals(100, 10, 1000, 50.0, 60.0))])
        self.assertEqual(dcm_lineage.weights_totals(record),
                         totals(150, 15, 1500, 75.0, 90.0))
        # A null total the sum needs, in the record or an ancestor, is unrecorded.
        self.assertIsNone(dcm_lineage.weights_totals(own_totals(50, None, 500, 25.0, 30.0)))
        record = own_totals(50, 5, 500, 25.0, 30.0,
                            runs=[ancestor("A", "branch", totals(100, None, 1000, 50.0, 60.0))])
        self.assertIsNone(dcm_lineage.weights_totals(record))
        # A fresh run with no ancestors is its own total.
        self.assertEqual(dcm_lineage.weights_totals(own_totals(7, 1, 70, 3.5, 4.0)), totals(7, 1, 70, 3.5, 4.0))

    def test_a_derive_is_not_counted_twice(self):
        # train -> derive: the derived run already carries the source's totals.
        record = own_totals(110, 11, 1100, 55.0, 66.0,
                            runs=[ancestor("A", "derive", totals(100, 10, 1000, 50.0, 60.0))])
        self.assertEqual(dcm_lineage.weights_totals(record)["cum_trainer_step"], 110)
        # Repeated derives: still only the record's own (carried) totals.
        record = own_totals(110, 11, 1100, 55.0, 66.0,
                            runs=[ancestor("A", "derive", totals(100, 10, 1000, 50.0, 60.0)),
                                  ancestor("B", "derive", totals(100, 10, 1000, 50.0, 60.0))])
        self.assertEqual(dcm_lineage.weights_totals(record)["cum_trainer_step"], 110)
        # A (100) -> branch -> B (50) -> derive -> C (+10): C's own total is 60
        # (B's 50 carried), the weights' total 60 + A's 100 = 160.
        record = own_totals(60, 6, 600, 30.0, 36.0,
                            runs=[ancestor("A", "branch", totals(100, 10, 1000, 50.0, 60.0)),
                                  ancestor("B", "derive", totals(50, 5, 500, 25.0, 30.0))])
        self.assertEqual(dcm_lineage.weights_totals(record)["cum_trainer_step"], 160)
        # derive -> branch: A (100) -> derive -> B (carries 100, trains to 120)
        # -> branch -> C (30): 30 + B's 120 (which already holds A's 100).
        record = own_totals(30, 3, 300, 15.0, 18.0,
                            runs=[ancestor("A", "derive", totals(100, 10, 1000, 50.0, 60.0)),
                                  ancestor("B", "branch", totals(120, 12, 1200, 60.0, 72.0))])
        self.assertEqual(dcm_lineage.weights_totals(record)["cum_trainer_step"], 150)
        # A resumed derived run: a resume adds no ancestry entry, so the
        # derived run's carried total grows by its own steps only.
        record = own_totals(125, 12, 1250, 62.0, 75.0,
                            runs=[ancestor("A", "derive", totals(100, 10, 1000, 50.0, 60.0))])
        self.assertEqual(dcm_lineage.weights_totals(record)["cum_trainer_step"], 125)

    def test_an_unrecorded_oldest_run_gives_unrecorded(self):
        record = own_totals(50, 5, 500, 25.0, 30.0, history="unrecorded",
                            runs=[ancestor("A", "branch", totals(100, 10, 1000, 50.0, 60.0))])
        self.assertIsNone(dcm_lineage.weights_totals(record))


if __name__ == "__main__":
    unittest.main()
