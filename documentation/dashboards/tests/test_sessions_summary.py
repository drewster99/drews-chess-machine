"""Tests for scripts/sessions_summary.py: it reads each .dcmsession folder's
session.json (and, from format v2, its lineage record) the way the app does.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic session folders in a temporary folder.
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)
import sessions_summary  # noqa: E402
from test_lineage import record_for  # noqa: E402

SWIFT_SESSION_FILE = os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachine", "Persistence",
                                  "SessionCheckpointFile.swift")


def state(session_id, format_version=2, lineage=None, arena_promotions=(), has_buffer=False):
    """A session.json body with every field the summary requires."""
    body = {
        "formatVersion": format_version, "sessionID": session_id, "savedAtUnix": 1_790_030_637,
        "sessionStartUnix": 1_790_005_838, "elapsedTrainingSec": 7200.0, "trainingSteps": 15901,
        "selfPlayGames": 103551, "selfPlayMoves": 19206232, "trainingPositionsSeen": 65130496,
        "batchSize": 4096, "learningRate": 0.001, "promoteThreshold": 0.53, "arenaGames": 400,
        "selfPlayTau": {"startTau": 0.2, "decayPerPly": 0.02, "floorTau": 0.02},
        "arenaTau": {"startTau": 0.2, "decayPerPly": 0.02, "floorTau": 0.02},
        "selfPlayWorkerCount": 175, "championID": f"{session_id}-3", "trainerID": f"{session_id}-4",
        "arenaHistory": [{"promoted": promoted} for promoted in arena_promotions],
        "hasReplayBuffer": has_buffer,
    }
    if lineage is not None:
        body["lineage"] = lineage
    return body


def vsuci_record():
    record = record_for(0, 500)
    record["invocation"]["path_kind"] = "vsuci"
    return record


class SessionsSummaryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, folder_name, body, extra_files=()):
        folder = os.path.join(self.dir, folder_name)
        os.mkdir(folder)
        if body is not None:
            with open(os.path.join(folder, "session.json"), "w") as handle:
                handle.write(body if isinstance(body, str) else json.dumps(body))
        for name, size in extra_files:
            with open(os.path.join(folder, name), "wb") as handle:
                handle.write(b"\0" * size)
        return folder

    def rows(self):
        return {row["name"]: row for row in sessions_summary.collect(self.dir)}

    def test_constants_match_the_app(self):
        with open(SWIFT_SESSION_FILE) as handle:
            swift = handle.read()
        self.assertEqual(int(re.search(r"static let currentFormatVersion: Int = (\d+)", swift).group(1)),
                         sessions_summary.CURRENT_FORMAT_VERSION)
        self.assertEqual(int(re.search(r"static let lineageRequiredFromFormatVersion: Int = (\d+)", swift).group(1)),
                         sessions_summary.LINEAGE_REQUIRED_FROM_FORMAT_VERSION)
        self.assertEqual(re.search(r'static let stateFilename = "([^"]+)"', swift).group(1),
                         sessions_summary.STATE_FILENAME)
        self.assertEqual(re.search(r'static let replayBufferFilename = "([^"]+)"', swift).group(1),
                         sessions_summary.REPLAY_BUFFER_FILENAME)

    def test_a_gui_save_with_a_lineage_record(self):
        name = "20261003-010000-20261003-1-AAAA-promote.dcmsession"
        self.write(name, state("20261003-1-AAAA", lineage=record_for(1, 1000),
                               arena_promotions=(False, True, True), has_buffer=True),
                   extra_files=[("replay_buffer.bin", 2048), ("trainer.safetensors", 1024)])
        row = self.rows()[name]
        self.assertNotIn("error", row)
        self.assertEqual((row["trigger"], row["session_id"], row["training_steps"]),
                         ("promote", "20261003-1-AAAA", 15901))
        self.assertEqual((row["arenas"], row["promotions"]), (3, 2))
        self.assertEqual((row["has_replay_buffer"], row["replay_buffer_file"]), (True, True))
        self.assertGreaterEqual(row["size_bytes"], 2048 + 1024)
        self.assertEqual(row["lineage"]["path_kind"], "replay")
        self.assertEqual(row["lineage"]["segment_index"], 1)
        self.assertEqual(row["lineage"]["cum_trainer_step"], 2000)
        self.assertTrue(row["lineage"]["exact_resume"])
        self.assertIn("promote", sessions_summary.render_table([row])[1])

    def test_a_train_vs_uci_save_is_named_by_its_trigger_and_path_kind(self):
        name = "20261003-020000-20261003-9-TeSt-vsuci-final.dcmsession"
        self.write(name, state("20261003-9-TeSt", lineage=vsuci_record()))
        row = self.rows()[name]
        self.assertEqual((row["trigger"], row["lineage"]["path_kind"]), ("vsuci-final", "vsuci"))
        self.assertIn("vsuci seg=0", sessions_summary.render_table([row])[1])

    def test_a_format_v1_save_has_no_lineage_and_a_renamed_folder_has_no_trigger(self):
        self.write("old-Ko63-try-to-resume-from-here.dcmsession", state("20260514-2-Ko63", format_version=1))
        self.write("20260531-024912-20260529-11-G5w2-promote-keep.dcmsession",
                   state("20260529-11-G5w2", format_version=1))
        rows = self.rows()
        renamed = rows["old-Ko63-try-to-resume-from-here.dcmsession"]
        self.assertEqual((renamed["trigger"], renamed["lineage"]), (sessions_summary.RENAMED, None))
        self.assertIn("none (v1)", sessions_summary.render_table([renamed])[1])
        self.assertEqual(rows["20260531-024912-20260529-11-G5w2-promote-keep.dcmsession"]["trigger"], "promote-keep")

    def test_a_folder_name_naming_another_session_is_not_trusted_for_the_trigger(self):
        name = "20261003-010000-20261003-1-AAAA-manual.dcmsession"
        self.write(name, state("20261003-2-BBBB", format_version=1))
        self.assertEqual(self.rows()[name]["trigger"], sessions_summary.RENAMED)

    def test_what_the_app_would_refuse_is_reported_not_guessed(self):
        self.write("20261003-030000-20261003-1-AAAA-manual.dcmsession", state("20261003-1-AAAA"))
        broken = vsuci_record()
        del broken["steps"]["cum_trainer_step"]
        self.write("20261003-040000-20261003-1-AAAA-manual.dcmsession", state("20261003-1-AAAA", lineage=broken))
        self.write("20261003-050000-20261003-1-AAAA-manual.dcmsession", None)
        self.write("20261003-060000-20261003-1-AAAA-manual.dcmsession", "{not json")
        self.write("20261003-070000-20261003-1-AAAA-manual.dcmsession",
                   {k: v for k, v in state("20261003-1-AAAA", lineage=vsuci_record()).items() if k != "trainingSteps"})
        self.write("20261003-080000-20261003-1-AAAA-manual.dcmsession",
                   state("20261003-1-AAAA", format_version=3, lineage=vsuci_record()))
        errors = {name: row.get("error") for name, row in self.rows().items()}
        self.assertIn("has no lineage", errors["20261003-030000-20261003-1-AAAA-manual.dcmsession"])
        self.assertIn("lineage has no steps.cum_trainer_step", errors["20261003-040000-20261003-1-AAAA-manual.dcmsession"])
        self.assertIn("no session.json", errors["20261003-050000-20261003-1-AAAA-manual.dcmsession"])
        self.assertIn("is not JSON", errors["20261003-060000-20261003-1-AAAA-manual.dcmsession"])
        self.assertIn("has no trainingSteps", errors["20261003-070000-20261003-1-AAAA-manual.dcmsession"])
        self.assertIn("unsupported session.json formatVersion 3", errors["20261003-080000-20261003-1-AAAA-manual.dcmsession"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = sessions_summary.main(["--sessions-dir", self.dir])
        self.assertEqual(status, 1)
        self.assertEqual(out.getvalue().count("ERROR:"), 6)

    def test_lineage_without_not_exact_items_is_a_row_error(self):
        broken = vsuci_record()
        del broken["run"]["not_exact_items"]
        self.write("20261003-010000-20261003-1-AAAA-manual.dcmsession", state("20261003-1-AAAA", lineage=broken))
        self.write("20261003-020000-20261003-1-AAAA-manual.dcmsession",
                   state("20261003-1-AAAA", lineage=vsuci_record()))
        rows = self.rows()
        self.assertIn("lineage has no run.not_exact_items",
                      rows["20261003-010000-20261003-1-AAAA-manual.dcmsession"]["error"])
        self.assertNotIn("error", rows["20261003-020000-20261003-1-AAAA-manual.dcmsession"],
                         "one bad session.json does not stop the listing")

    def test_only_session_folders_are_read(self):
        self.write("20261003-010000-20261003-1-AAAA-manual.dcmsession", state("20261003-1-AAAA", lineage=vsuci_record()))
        self.write("20261003-020000-20261003-1-AAAA-manual.dcmsession.tmp", state("20261003-1-AAAA"))
        self.write("notes", None)
        with open(os.path.join(self.dir, "readme.txt"), "w") as handle:
            handle.write("not a session")
        self.assertEqual(list(self.rows()), ["20261003-010000-20261003-1-AAAA-manual.dcmsession"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = sessions_summary.main(["--sessions-dir", self.dir, "--json"])
        self.assertEqual(status, 0)
        self.assertEqual([row["trigger"] for row in json.loads(out.getvalue())], ["manual"])

    def test_the_buffer_column_checks_session_json_against_the_folder(self):
        cases = [
            ("20261003-010000-20261003-1-AAAA-manual.dcmsession", True, 64, "yes"),
            ("20261003-020000-20261003-1-AAAA-manual.dcmsession", False, 0, "no"),
            ("20261003-030000-20261003-1-AAAA-manual.dcmsession", True, 0, "MISSING"),
            ("20261003-040000-20261003-1-AAAA-manual.dcmsession", False, 64, "UNLISTED"),
        ]
        for name, declared, buffer_bytes, _ in cases:
            files = [("replay_buffer.bin", buffer_bytes)] if buffer_bytes else []
            self.write(name, state("20261003-1-AAAA", lineage=vsuci_record(), has_buffer=declared), extra_files=files)
        rows = self.rows()
        for name, _, _, label in cases:
            self.assertEqual(sessions_summary.buffer_label(rows[name]), label, name)
            self.assertIn(f"  {label}", sessions_summary.render_table([rows[name]])[1])

    def test_the_table_aligns_every_column(self):
        self.write("20261003-010000-20261003-1-AAAA-manual.dcmsession",
                   state("20261003-1-AAAA", lineage=vsuci_record(), arena_promotions=(True,)))
        self.write("old-renamed-save.dcmsession", state("20260514-2-Ko63", format_version=1))
        self.write("20261003-020000-20261003-1-AAAA-manual.dcmsession", None)
        lines = sessions_summary.render_table(sessions_summary.collect(self.dir))
        self.assertEqual(len(lines), 4)
        header, saved, broken, renamed = lines
        # Search past the folder name, which also holds the trigger.
        self.assertEqual(header.index("trigger"), saved.index("manual", header.index("size")))
        self.assertEqual(header.index("trigger"), renamed.index("(renamed)"))
        self.assertEqual(header.index("lineage"), saved.index("vsuci seg=0"))
        self.assertEqual(header.index("lineage"), renamed.index("none (v1)"))
        self.assertEqual(header.index("size") + len("size"), broken.index("ERROR:") - 2,
                         "an error row's size lines up and its message follows it")

    def test_sizes_are_base_2(self):
        self.assertEqual(sessions_summary.binary_size(1024 ** 2), "1.0 MB")
        self.assertEqual(sessions_summary.binary_size(11 * 1024 ** 3), "11.0 GB")


if __name__ == "__main__":
    unittest.main()
