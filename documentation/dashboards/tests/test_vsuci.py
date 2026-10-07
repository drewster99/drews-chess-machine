"""Tests for the train-vs-UCI tracker's step-line parsing (documentation/dashboards/vsuci.py):
the current [VS-UCI] line format (with pLogitMean= / vLogitMean= and a trailing trainerStep=)
is read, a log whose lines carry trainerStep= gets its 1000-step marks on trainer-step
multiples of 1000, and a log from older builds keeps its marks at segment multiples of 1000.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic logs in a temporary folder.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, DASH)
import vsuci  # noqa: E402


def current_line(second, step, trainer_step, ms=200.0):
    """A [VS-UCI] step line in the format the runner writes today (TrainVsUciRunner)."""
    return (f"00:{second // 60:02d}:{second % 60:02d}.000 [VS-UCI] step={step} loss=2.5000 pLoss=1.2000 "
            f"vLoss=0.9000 pEnt=3.100 playedP=0.250 pLogitMean=0.1234 vLogitMean=-0.0100 gNorm=1.500 "
            f"lr=0.001 ms={ms:.1f} buf=4096 mom=0.9000 lrCyc[0.0005,0.002] trainerStep={trainer_step}\n")


def older_line(second, step, ms=200.0):
    """A [VS-UCI] step line from a build before pLogitMean= and trainerStep=."""
    return (f"00:{second // 60:02d}:{second % 60:02d}.000 [VS-UCI] step={step} loss=2.5000 pLoss=1.2000 "
            f"vLoss=0.9000 pEnt=-- playedP=-- gNorm=1.500 lr=0.001 ms={ms:.1f} buf=4096\n")


class VsUciMarkTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()

    def write(self, name, lines):
        path = os.path.join(self.folder.name, name)
        with open(path, "w") as handle:
            handle.writelines(lines)
        return path

    def test_the_current_line_format_is_read_with_its_trainer_step(self):
        path = self.write("current.txt", [current_line(1, 1, 514)])
        per_step, _ = vsuci.parse_segment(path, 0, {})
        self.assertEqual(list(per_step), [1])
        self.assertEqual(per_step[1]["trainerStep"], 514)
        self.assertEqual(per_step[1]["gNorm"], 1.5)

    def test_a_resumed_segments_marks_are_on_trainer_step_multiples_of_1000(self):
        # Resumed at trainer step 513: the segment's first line, the dense and fixed
        # lines at trainer steps 1000 and 2000, and a time line between them.
        lines = [current_line(1, 1, 514), current_line(10, 487, 1000), current_line(20, 700, 1213),
                 current_line(30, 987, 1500), current_line(40, 1487, 2000)]
        per_step, _ = vsuci.parse_segment(self.write("resumed.txt", lines), 0, {})
        self.assertEqual([meta for meta, _ in vsuci.marks_of(per_step)], [487, 1487])

    def test_a_log_without_trainer_steps_keeps_its_segment_step_marks(self):
        lines = [older_line(second, step) for second, step in enumerate(range(50, 2001, 50), 1)]
        per_step, _ = vsuci.parse_segment(self.write("older.txt", lines), 0, {})
        marks = vsuci.marks_of(per_step)
        self.assertEqual([meta for meta, _ in marks], [1000, 2000])
        self.assertIsNone(marks[0][1]["trainerStep"])


if __name__ == "__main__":
    unittest.main()
