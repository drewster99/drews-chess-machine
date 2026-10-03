"""Tests for _guarded_csv: compare-and-swap, no-silent-shrink, and the folder lock.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works in its own temporary folder; nothing under the repository is written.
"""
import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import _guarded_csv as G  # noqa: E402

FIELDS = ["cum_step", "meta_step", "segment", "pElo", "note"]


def write(path, rows, fieldnames=FIELDS):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row(meta, pelo="", segment=0, note=""):
    return {"cum_step": str(meta), "meta_step": str(meta), "segment": str(segment), "pElo": pelo, "note": note}


class GuardedCsvTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "run.csv")

    def tearDown(self):
        self.folder.cleanup()

    def test_replace_after_unchanged_read_writes_rows(self):
        write(self.path, [row(1000, "900")])
        rows, _, snapshot = G.read_rows(self.path)
        rows.append(row(2000, "950"))
        G.replace_rows(self.path, rows, FIELDS, snapshot, G.replay_row_key)
        self.assertEqual([r["pElo"] for r in G.read_rows(self.path)[0]], ["900", "950"])

    def test_concurrent_change_is_refused_and_file_kept(self):
        write(self.path, [row(1000, "900")])
        rows, _, snapshot = G.read_rows(self.path)
        write(self.path, [row(1000, "900"), row(2000, "960")])  # another writer saves first
        before = open(self.path, "rb").read()
        rows.append(row(2000, "950"))
        with self.assertRaises(G.ConcurrentModification):
            G.replace_rows(self.path, rows, FIELDS, snapshot, G.replay_row_key)
        self.assertEqual(open(self.path, "rb").read(), before)

    def test_dropping_a_row_is_refused_unless_allowed(self):
        write(self.path, [row(1000, "900"), row(2000, "950")])
        rows, _, snapshot = G.read_rows(self.path)
        with self.assertRaises(G.SourceOfTruthShrink):
            G.replace_rows(self.path, rows[:1], FIELDS, snapshot, G.replay_row_key)
        self.assertEqual(len(G.read_rows(self.path)[0]), 2)
        G.replace_rows(self.path, rows[:1], FIELDS, snapshot, G.replay_row_key, allow_shrink=True)
        self.assertEqual(len(G.read_rows(self.path)[0]), 1)

    def test_blanking_a_value_is_refused_but_note_and_named_columns_are_not(self):
        write(self.path, [row(1000, "900", note="first")])
        rows, _, snapshot = G.read_rows(self.path)
        blanked = [dict(rows[0], pElo="")]
        with self.assertRaises(G.SourceOfTruthShrink):
            G.replace_rows(self.path, blanked, FIELDS, snapshot, G.replay_row_key)
        G.replace_rows(self.path, [dict(rows[0], note="")], FIELDS, snapshot, G.replay_row_key)
        rows, _, snapshot = G.read_rows(self.path)
        G.replace_rows(self.path, [dict(rows[0], pElo="")], FIELDS, snapshot, G.replay_row_key,
                       allowed_blank_columns=frozenset({"pElo"}))
        self.assertEqual(G.read_rows(self.path)[0][0]["pElo"], "")

    def test_a_column_the_old_header_lacks_is_absent_not_blanked(self):
        write(self.path, [{"cum_step": "1000", "meta_step": "1000", "segment": "0", "pElo": "900"}],
              fieldnames=["cum_step", "meta_step", "segment", "pElo"])
        rows, _, snapshot = G.read_rows(self.path)
        G.replace_rows(self.path, rows, FIELDS, snapshot, G.replay_row_key)
        self.assertEqual(G.read_rows(self.path)[1], FIELDS)

    def test_rows_with_columns_outside_the_schema_are_refused(self):
        write(self.path, [row(1000, "900")])
        rows, _, snapshot = G.read_rows(self.path)
        with self.assertRaises(ValueError):
            G.replace_rows(self.path, [dict(rows[0], stray="x")], FIELDS, snapshot, G.replay_row_key)

    def test_writing_a_new_file_needs_an_absent_snapshot(self):
        _, _, snapshot = G.read_rows(self.path)
        self.assertEqual(snapshot.digest, G.ABSENT)
        G.replace_rows(self.path, [row(1000, "900")], FIELDS, snapshot, G.replay_row_key)
        with self.assertRaises(G.ConcurrentModification):
            G.replace_rows(self.path, [row(1000, "900")], FIELDS, snapshot, G.replay_row_key)

    def test_append_only_text_replace(self):
        with open(self.path, "w", newline="") as handle:
            handle.write("step,pElo\r\n1,2\r\n")
        text, snapshot = G.read_text(self.path)
        G.replace_text_if_unchanged(self.path, text + "3,4\r\n", snapshot, must_extend=True)
        self.assertEqual(open(self.path, "rb").read(), b"step,pElo\r\n1,2\r\n3,4\r\n")
        text, snapshot = G.read_text(self.path)
        with self.assertRaises(G.SourceOfTruthShrink):
            G.replace_text_if_unchanged(self.path, "step,pElo\r\n", snapshot, must_extend=True)

    def test_folder_lock_excludes_another_process_and_does_not_nest(self):
        script = textwrap.dedent(f"""
            import fcntl, os, sys
            fd = os.open({self.folder.name!r}, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("acquired")
            except BlockingIOError:
                print("blocked")
        """)
        with G.folder_lock(self.folder.name):
            held = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout.strip()
            with self.assertRaises(RuntimeError):
                with G.folder_lock(self.folder.name):
                    pass
        free = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout.strip()
        self.assertEqual((held, free), ("blocked", "acquired"))

    def test_row_keys(self):
        self.assertEqual(G.replay_row_key({"segment": "2", "meta_step": "3000"}), ("2", 3000))
        self.assertEqual(G.selfplay_row_key({"cum_step": "376977"}), 377)


if __name__ == "__main__":
    unittest.main()
