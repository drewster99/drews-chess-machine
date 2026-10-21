"""Tests for the lineage-schema-3 analysis in scripts/dcm_lineage.py: `corpus_ids`
over every corpus shape and `weights_totals` along a record's ancestry
(hyperparameter recording plan B4, O-15: totals behind the weights, summed over
branched-from runs, never over derived-from ones).

Which keys each schema requires or refuses is the app's rule (LineageRecord's
decoder; LineageSchema3Tests in Swift); dcm_lineage reads records the app accepted.

Run: python3 -m unittest discover -s documentation/dashboards/tests
"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_lineage  # noqa: E402

SWIFT_TESTS = os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachineTests")


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


class RealRecordTests(unittest.TestCase):
    def test_the_real_schema_two_fixtures_read_their_corpus_and_have_no_weights_totals(self):
        """The two real records the Swift tests pin (`LineageSchemaTwoFixtures`).
        Fails if: `corpus_ids` misses the schema-2 corpus shape, or `weights_totals`
        invents a total for a record that carries no ancestry."""
        with open(os.path.join(SWIFT_TESTS, "LineageSchemaTwoFixtures.swift")) as handle:
            source = handle.read()
        literals = re.findall(r'static let \w+ = #"(.*?)"#', source)
        self.assertEqual(len(literals), 2)
        for text in literals:
            record = json.loads(text)
            self.assertEqual(record["schema"], 2)
            self.assertEqual(dcm_lineage.corpus_ids(record["fed"]["corpus"]), ["20260624-192615-w3aA5b"])
            self.assertIsNone(dcm_lineage.weights_totals(record), "a schema-2 record has no ancestry")


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
