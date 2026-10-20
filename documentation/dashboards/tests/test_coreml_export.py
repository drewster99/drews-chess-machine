"""Tests for scripts/dcm_coreml_export.py: what it refuses and how it folds constants.

The translation of every operation kind is proven end to end against the app's own
forward pass by CoreMLExportParityTests (Swift); these cover the script's own rules.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Tests that build a MIL program need coremltools and skip without it.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_coreml_export as export  # noqa: E402

HAVE_COREMLTOOLS = importlib.util.find_spec("coremltools") is not None


def write_graph(folder, graph, weights_bytes):
    with open(os.path.join(folder, export.GRAPH_FILE), "w") as handle:
        json.dump(graph, handle)
    with open(os.path.join(folder, export.WEIGHTS_FILE), "wb") as handle:
        handle.write(weights_bytes)


def one_weight_graph(**overrides):
    graph = {
        "format_version": export.GRAPH_FORMAT_VERSION,
        "input": {"id": "board", "shape": [-1, 2, 8, 8], "data_type": "float32"},
        "weights": [{"id": "w0", "name": "stem.conv.weight", "shape": [1, 2, 1, 1],
                     "byte_offset": 0, "element_count": 2}],
        "operations": [],
        "outputs": [],
    }
    graph.update(overrides)
    return graph


class LoadGraphTests(unittest.TestCase):
    """Protects: a graph folder is read only when it is the format and size the script
    understands. Fails if: a newer graph format, or a weights file that does not match
    the graph, is read anyway (silently wrong weights)."""

    def load(self, graph, weights_bytes):
        with tempfile.TemporaryDirectory() as folder:
            write_graph(folder, graph, weights_bytes)
            return export.load_graph(folder)

    def test_reads_weights_in_their_graph_shape(self):
        _, weights = self.load(one_weight_graph(), np.array([1.5, -2.0], dtype="<f4").tobytes())
        self.assertEqual(weights["w0"].shape, (1, 2, 1, 1))
        self.assertEqual(weights["w0"].ravel().tolist(), [1.5, -2.0])

    def test_refuses_another_graph_format(self):
        with self.assertRaisesRegex(export.ExportError, "graph format"):
            self.load(one_weight_graph(format_version=export.GRAPH_FORMAT_VERSION + 1),
                      np.zeros(2, dtype="<f4").tobytes())

    def test_refuses_a_weights_file_longer_than_the_graph(self):
        with self.assertRaisesRegex(export.ExportError, "holds 3 values"):
            self.load(one_weight_graph(), np.zeros(3, dtype="<f4").tobytes())

    def test_refuses_a_weight_past_the_end_of_the_file(self):
        with self.assertRaisesRegex(export.ExportError, "does not fit"):
            self.load(one_weight_graph(), np.zeros(1, dtype="<f4").tobytes())

    def test_refuses_a_weight_whose_shape_disagrees_with_its_count(self):
        graph = one_weight_graph()
        graph["weights"][0]["shape"] = [3]
        with self.assertRaisesRegex(export.ExportError, "does not fit"):
            self.load(graph, np.zeros(2, dtype="<f4").tobytes())


@unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools is not installed")
class TranslateTests(unittest.TestCase):
    """Protects: the translator refuses what it does not know and folds constant-only
    chains. Fails if: an unknown operation kind is skipped or guessed instead of refused,
    or the ReZero chain C*tanh(a/C) (a weight and constants only) is not evaluated."""

    def build(self, operations, outputs, weights=None):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        graph = {"input": {"id": "board"}, "operations": operations, "outputs": outputs}
        build, _ = export.translate(graph, weights or {}, mb)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 8, 8))], opset_version=ct.target.macOS15)
        def program(board):
            return tuple(build(board).values())
        return program

    def test_an_unknown_operation_kind_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "unknown operation kind 'swish'"):
            self.build([{"id": "t0", "data_type": "float32", "kind": {"swish": {"input": "board"}}}],
                       [{"name": "policy", "id": "t0"}])

    def test_an_operand_read_before_it_is_defined_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "'t9' is not defined"):
            self.build([{"id": "t0", "data_type": "float32", "kind": {"reLU": {"input": "t9"}}}],
                       [{"name": "policy", "id": "t0"}])

    def test_an_output_that_does_not_depend_on_the_board_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "does not depend on the board"):
            self.build([{"id": "t0", "data_type": "float32", "kind": {"constant": {"values": [1.0], "shape": [1]}}}],
                       [{"name": "policy", "id": "t0"}])

    def test_the_rezero_scale_folds_to_one_constant(self):
        cap, alpha = 0.5, np.array([0.8], dtype=np.float32)
        ops = [
            {"id": "t0", "data_type": "float32", "kind": {"constant": {"values": [cap], "shape": []}}},
            {"id": "t1", "data_type": "float32", "kind": {"division": {"primary": "w0", "secondary": "t0"}}},
            {"id": "t2", "data_type": "float32", "kind": {"tanh": {"input": "t1"}}},
            {"id": "t3", "data_type": "float32", "kind": {"multiplication": {"primary": "t0", "secondary": "t2"}}},
            {"id": "t4", "data_type": "float32", "kind": {"multiplication": {"primary": "board", "secondary": "t3"}}},
        ]
        program = self.build(ops, [{"name": "policy", "id": "t4"}], {"w0": alpha})
        mul = [op for op in program.functions["main"].operations if op.op_type == "mul"]
        self.assertEqual(len(mul), 1, "only the board multiplication is left in the program")
        self.assertAlmostEqual(float(np.asarray(mul[0].y.val).ravel()[0]), cap * np.tanh(0.8 / cap), places=6)


class ComputePrecisionTests(unittest.TestCase):
    """Protects: mirror precision runs an op in fp16 exactly when the MPSGraph ran it in
    bf16/fp16, and an op a lowering adds follows the op it came from. Fails if: the
    selector stops reading the recorded type, or stops mapping `<id>__<part>` to `<id>`
    (the policy tail's normalization parts would then drop to fp16), or lets an op the
    converter adds run in fp16 inside an all-fp32 model."""

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools is not installed")
    def test_mirror_selects_fp16_by_the_recorded_type(self):
        import coremltools as ct
        selector = export.compute_precision(ct, "mirror", {"t1": "bfloat16", "t2": "float32"}).op_selector

        class Op:
            def __init__(self, name):
                self.name = name
        self.assertTrue(selector(Op("t1")))
        self.assertTrue(selector(Op("t1__scale")))
        self.assertFalse(selector(Op("t2")))
        self.assertFalse(selector(Op("t2__inv_std")))
        self.assertTrue(selector(Op("converter_added")), "follows a reduced-precision network")

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools is not installed")
    def test_mirror_keeps_an_all_fp32_network_in_fp32(self):
        import coremltools as ct
        selector = export.compute_precision(ct, "mirror", {"t1": "float32", "policy": "float32"}).op_selector

        class Op:
            def __init__(self, name):
                self.name = name
        self.assertFalse(selector(Op("t1")))
        self.assertFalse(selector(Op("converter_added")))


if __name__ == "__main__":
    unittest.main()
