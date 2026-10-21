"""Tests for scripts/dcm_coreml_export.py: what it refuses, how it folds constants,
and the types mirror precision gives each op.

The translation of every operation kind is proven end to end against the app's own
forward pass by CoreMLExportParityTests (Swift); these cover the script's own rules,
on graph files written the way `--export-inference-graph` writes them.

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


def weight(weight_id, name, shape, offset, data_type="float32"):
    return {"id": weight_id, "name": name, "shape": shape, "data_type": data_type,
            "byte_offset": offset, "element_count": int(np.prod(shape))}


def graph_with(weights=(), operations=(), outputs=()):
    return {"format_version": export.GRAPH_FORMAT_VERSION,
            "input": {"id": "board", "shape": [-1, 2, 8, 8], "data_type": "float32"},
            "weights": list(weights), "operations": list(operations), "outputs": list(outputs)}


class LoadGraphTests(unittest.TestCase):
    """Protects: a graph folder is read only when it is the format the script reads,
    its ids are unique and its weights tile weights.bin exactly. Fails if: a newer
    format, overlapping or out-of-range weights, a short or long weights file, or a
    reused id is read anyway (silently wrong weights or operands)."""

    def load(self, graph, weights_bytes):
        with tempfile.TemporaryDirectory() as folder:
            write_graph(folder, graph, weights_bytes)
            return export.load_graph(folder)

    def test_reads_weights_in_their_graph_shape(self):
        graph = graph_with([weight("w0", "a", [1, 2, 1, 1], 0), weight("w1", "b", [2], 8)])
        _, weights = self.load(graph, np.array([1.5, -2.0, 3.0, 4.0], dtype="<f4").tobytes())
        self.assertEqual(weights["w0"].shape, (1, 2, 1, 1))
        self.assertEqual(weights["w0"].ravel().tolist(), [1.5, -2.0])
        self.assertEqual(weights["w1"].tolist(), [3.0, 4.0])

    def test_refuses_another_graph_format(self):
        graph = graph_with([weight("w0", "a", [2], 0)])
        graph["format_version"] = export.GRAPH_FORMAT_VERSION + 1
        with self.assertRaisesRegex(export.ExportError, "graph format"):
            self.load(graph, np.zeros(2, dtype="<f4").tobytes())

    def test_refuses_a_weights_file_longer_than_the_graph(self):
        with self.assertRaisesRegex(export.ExportError, "holds 3 values"):
            self.load(graph_with([weight("w0", "a", [2], 0)]), np.zeros(3, dtype="<f4").tobytes())

    def test_refuses_a_weight_past_the_end_of_the_file(self):
        with self.assertRaisesRegex(export.ExportError, "past the end"):
            self.load(graph_with([weight("w0", "a", [2], 0)]), np.zeros(1, dtype="<f4").tobytes())

    def test_refuses_overlapping_weights(self):
        graph = graph_with([weight("w0", "a", [2], 0), weight("w1", "b", [2], 0)])
        with self.assertRaisesRegex(export.ExportError, "does not continue"):
            self.load(graph, np.zeros(4, dtype="<f4").tobytes())

    def test_refuses_a_weight_whose_shape_disagrees_with_its_count(self):
        graph = graph_with([weight("w0", "a", [2], 0)])
        graph["weights"][0]["shape"] = [3]
        with self.assertRaisesRegex(export.ExportError, "does not continue"):
            self.load(graph, np.zeros(2, dtype="<f4").tobytes())

    def test_refuses_an_id_used_twice(self):
        graph = graph_with([weight("w0", "a", [2], 0)],
                           [{"id": "w0", "label": None, "data_type": "float32", "kind": {"reLU": {"input": "board"}}}])
        with self.assertRaisesRegex(export.ExportError, "used more than once: w0"):
            self.load(graph, np.zeros(2, dtype="<f4").tobytes())


@unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools is not installed")
class TranslateTests(unittest.TestCase):
    """Protects: the translator refuses what it does not know, folds constant-only
    chains, and gives each op the type the precision mode asks for. Fails if: an
    unknown operation kind is skipped or guessed instead of refused, the ReZero chain
    C*tanh(a/C) is not evaluated, or mirror stops running a recorded bf16 op in fp16
    and a recorded fp32 op in fp32."""

    def program(self, graph, weights, precision="fp32"):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        build = export.translate(graph, weights, mb, types, precision)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 8, 8))], opset_version=ct.target.macOS15)
        def program(board):
            return tuple(build(board).values())
        return program

    def op(self, op_id, kind, args, data_type="float32"):
        return {"id": op_id, "label": None, "data_type": data_type, "kind": {kind: args}}

    def test_an_unknown_operation_kind_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "unknown operation kind 'swish'"):
            self.program(graph_with([], [self.op("t0", "swish", {"input": "board"})],
                                    [{"name": "policy", "id": "t0"}]), {})

    def test_an_operand_read_before_it_is_defined_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "'t9' is not defined"):
            self.program(graph_with([], [self.op("t0", "reLU", {"input": "t9"})], [{"name": "policy", "id": "t0"}]), {})

    def test_an_output_that_does_not_depend_on_the_board_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "does not depend on the board"):
            self.program(graph_with([], [self.op("t0", "constant", {"values": [1.0], "shape": [1]})],
                                    [{"name": "policy", "id": "t0"}]), {})

    def test_the_rezero_scale_folds_to_one_constant(self):
        cap = 0.5
        ops = [self.op("t0", "constant", {"values": [cap], "shape": []}),
               self.op("t1", "division", {"primary": "w0", "secondary": "t0"}),
               self.op("t2", "tanh", {"input": "t1"}),
               self.op("t3", "multiplication", {"primary": "t0", "secondary": "t2"}),
               self.op("t4", "multiplication", {"primary": "board", "secondary": "t3"})]
        program = self.program(graph_with([weight("w0", "alpha", [1], 0)], ops, [{"name": "policy", "id": "t4"}]),
                               {"w0": np.array([0.8], dtype=np.float32)})
        mul = [op for op in program.functions["main"].operations if op.op_type == "mul"]
        self.assertEqual(len(mul), 1, "only the board multiplication is left in the program")
        self.assertAlmostEqual(float(np.asarray(mul[0].y.val).ravel()[0]), cap * np.tanh(0.8 / cap), places=6)

    def test_mirror_runs_each_op_in_its_recorded_type(self):
        """A bf16 conv, widened to fp32 for its BN, as a policy tail is recorded."""
        from coremltools.converters.mil.mil import types
        graph = graph_with(
            [weight("w0", "conv.weight", [3, 2, 1, 1], 0, "bfloat16")] +
            [weight(f"w{i}", f"bn.{name}", [1, 3, 1, 1], 24 + 12 * (i - 1), "float32")
             for i, name in enumerate(("weight", "bias", "running_mean", "running_var"), start=1)],
            [self.op("t0", "cast", {"input": "board"}, "bfloat16"),
             self.op("t1", "convolution2D", {"input": "t0", "weights": "w0", "strideY": 1, "strideX": 1,
                                             "dilationY": 1, "dilationX": 1, "groups": 1, "paddingTop": 0,
                                             "paddingBottom": 0, "paddingLeft": 0, "paddingRight": 0}, "bfloat16"),
             self.op("t2", "cast", {"input": "t1"}, "float32"),
             self.op("t3", "normalize", {"input": "t2", "mean": "w3", "variance": "w4", "gamma": "w1", "beta": "w2",
                                         "epsilon": 1e-5}, "float32")],
            [{"name": "policy", "id": "t3"}])
        weights = {"w0": np.ones((3, 2, 1, 1), np.float32), "w1": np.ones((1, 3, 1, 1), np.float32),
                   "w2": np.zeros((1, 3, 1, 1), np.float32), "w3": np.zeros((1, 3, 1, 1), np.float32),
                   "w4": np.ones((1, 3, 1, 1), np.float32)}
        operations = {op.name: op for op in self.program(graph, weights, "mirror").functions["main"].operations}
        self.assertEqual(operations["t1"].outputs[0].dtype, types.fp16)
        self.assertEqual(operations["t2"].outputs[0].dtype, types.fp32)
        self.assertEqual(operations["t3"].outputs[0].dtype, types.fp32)

    def test_an_unknown_recorded_type_is_refused(self):
        with self.assertRaisesRegex(export.ExportError, "unknown recorded data type 'int8'"):
            export.mil_type("int8", "mirror")


if __name__ == "__main__":
    unittest.main()
