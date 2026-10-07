"""Tests for the tensor comparator (scripts/safetensors_tensor_compare.py).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic .safetensors files in a temporary folder.
"""
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import safetensors_tensor_compare as compare_module  # noqa: E402

FLT_MIN = compare_module.FLT_MIN


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


def f16(values):
    return struct.pack(f"<{len(values)}e", *values)


def bf16(values):
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16) for value in values)


def write(path, tensors, metadata=None):
    """tensors: name -> (dtype, shape, raw bytes)."""
    header = {"__metadata__": metadata if metadata is not None else {"model_id": os.path.basename(path)}}
    data = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + data)


class TensorCompareTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)

    def pair(self, reference, other, reference_metadata=None, other_metadata=None):
        reference_path = os.path.join(self.folder, "reference.safetensors")
        other_path = os.path.join(self.folder, "other.safetensors")
        write(reference_path, reference, reference_metadata)
        write(other_path, other, other_metadata)
        return reference_path, other_path

    def run_main(self, *arguments):
        return compare_module.main(list(arguments))

    def test_identical_files_pass_bit_exact_whatever_the_metadata(self):
        tensors = {"w": ("F32", [3], f32([1.0, -2.0, 3.5])), "b": ("F32", [1], f32([0.25]))}
        paths = self.pair(tensors, dict(tensors), {"model_id": "A"}, {"model_id": "B", "extra": "x"})
        self.assertEqual(self.run_main(*paths), 0)

    def test_one_ulp_apart_fails_bit_exact_and_passes_with_a_tolerance(self):
        one = 1.0
        next_up = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", one))[0] + 1))[0]
        paths = self.pair({"w": ("F32", [2], f32([one, 2.0]))}, {"w": ("F32", [2], f32([next_up, 2.0]))})
        self.assertEqual(self.run_main(*paths), 1)
        ratio = (next_up - one) / 2.0
        self.assertEqual(self.run_main(*paths, "--max-relative", repr(ratio)), 0)
        self.assertEqual(self.run_main(*paths, "--max-relative", repr(ratio / 2)), 1)

    def test_signed_zero_differs_bit_exact_only(self):
        paths = self.pair({"w": ("F32", [2], f32([0.0, 1.0]))}, {"w": ("F32", [2], f32([-0.0, 1.0]))})
        self.assertEqual(self.run_main(*paths), 1)
        self.assertEqual(self.run_main(*paths, "--max-relative", "1e-9"), 0)

    def test_an_all_zero_reference_tensor_uses_the_flt_min_floor(self):
        tiny = struct.unpack("<f", f32([2 * FLT_MIN]))[0]
        paths = self.pair({"w": ("F32", [2], f32([0.0, 0.0]))}, {"w": ("F32", [2], f32([0.0, tiny]))})
        # |a - b| / max(max|a|, FLT_MIN) = 2 FLT_MIN / FLT_MIN = 2, never a division by zero.
        passed, messages = compare_module.compare(*paths, max_relative=1.9)
        self.assertFalse(passed, messages)
        passed, messages = compare_module.compare(*paths, max_relative=2.0)
        self.assertTrue(passed, messages)

    def test_a_nan_fails_unless_byte_identical(self):
        nan = float("nan")
        same = self.pair({"w": ("F32", [2], f32([nan, 1.0]))}, {"w": ("F32", [2], f32([nan, 1.0]))})
        self.assertEqual(self.run_main(*same, "--max-relative", "0.5"), 0)
        different = self.pair({"w": ("F32", [2], f32([nan, 1.0]))}, {"w": ("F32", [2], f32([nan, 1.5]))})
        self.assertEqual(self.run_main(*different, "--max-relative", "1e6"), 1)
        infinite = self.pair({"w": ("F32", [2], f32([1.0, 1.0]))}, {"w": ("F32", [2], f32([math.inf, 1.0]))})
        self.assertEqual(self.run_main(*infinite, "--max-relative", "1e6"), 1)

    def test_a_shape_mismatch_fails(self):
        paths = self.pair({"w": ("F32", [2, 1], f32([1.0, 2.0]))}, {"w": ("F32", [1, 2], f32([1.0, 2.0]))})
        self.assertEqual(self.run_main(*paths), 1)
        self.assertEqual(self.run_main(*paths, "--max-relative", "1"), 1)

    def test_a_dtype_mismatch_fails(self):
        paths = self.pair({"w": ("F32", [2], f32([1.0, 2.0]))}, {"w": ("I32", [2], struct.pack("<2i", 1, 2))})
        self.assertEqual(self.run_main(*paths, "--max-relative", "1"), 1)

    def test_a_missing_tensor_fails(self):
        paths = self.pair({"w": ("F32", [1], f32([1.0])), "b": ("F32", [1], f32([0.0]))},
                          {"w": ("F32", [1], f32([1.0]))})
        self.assertEqual(self.run_main(*paths), 1)
        self.assertEqual(self.run_main(*reversed(paths)), 1)

    def test_each_float_dtype_is_decoded(self):
        for dtype, encode in (("F32", f32), ("F16", f16), ("BF16", bf16)):
            with self.subTest(dtype=dtype):
                paths = self.pair({"w": (dtype, [2], encode([2.0, -4.0]))}, {"w": (dtype, [2], encode([2.0, -3.0]))})
                # Worst |a - b| = 1, reference scale 4: ratio 0.25.
                self.assertEqual(self.run_main(*paths, "--max-relative", "0.25"), 0)
                self.assertEqual(self.run_main(*paths, "--max-relative", "0.24"), 1)

    def test_a_non_float_dtype_must_be_byte_identical(self):
        paths = self.pair({"n": ("I64", [1], struct.pack("<q", 5))}, {"n": ("I64", [1], struct.pack("<q", 6))})
        self.assertEqual(self.run_main(*paths, "--max-relative", "1e6"), 1)

    def test_an_unreadable_file_exits_two(self):
        reference = os.path.join(self.folder, "reference.safetensors")
        write(reference, {"w": ("F32", [1], f32([1.0]))})
        garbage = os.path.join(self.folder, "garbage.safetensors")
        with open(garbage, "wb") as handle:
            handle.write(b"\x01")
        self.assertEqual(self.run_main(reference, garbage), 2)
        self.assertEqual(self.run_main(reference, os.path.join(self.folder, "absent.safetensors")), 2)

    def test_a_tensor_whose_bytes_do_not_match_its_shape_is_unreadable(self):
        reference = os.path.join(self.folder, "reference.safetensors")
        write(reference, {"w": ("F32", [3], f32([1.0, 2.0]))})
        self.assertEqual(self.run_main(reference, reference), 2)

    def write_entry(self, entry):
        """A file holding one 2-element F32 tensor `w` whose header entry is `entry`."""
        path = os.path.join(self.folder, "malformed.safetensors")
        encoded = json.dumps({"__metadata__": {}, "w": entry}).encode()
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)) + encoded + f32([1.0, 2.0]))
        return path

    def test_a_malformed_shape_dtype_or_offsets_is_unreadable_not_a_difference(self):
        good = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        malformed = [("shape", 2), ("shape", "2"), ("shape", [2.0]), ("shape", ["2"]), ("shape", [-2]),
                     ("shape", [True, 2]), ("shape", None), ("data_offsets", [0.0, 8]), ("data_offsets", ["0", "8"]),
                     ("data_offsets", [False, 8]), ("data_offsets", [-1, 8]), ("dtype", ["F32"]), ("dtype", 4)]
        for key, value in malformed:
            with self.subTest(key=key, value=value):
                path = self.write_entry(dict(good, **{key: value}))
                with self.assertRaises(compare_module.UnreadableFile):
                    compare_module.read_tensors(path)
                self.assertEqual(self.run_main(path, path), 2)
        self.assertEqual(self.run_main(self.write_entry(good), self.write_entry(good)), 0)


if __name__ == "__main__":
    unittest.main()
