"""Turn a DCM model file into a Core ML model (.mlpackage).

The network is never described here. The app records its own inference forward
pass from the graph builders (`--export-inference-graph`, `InferenceGraph` in
Network/InferenceGraph.swift): every primitive op in build order, the weights in
the layout the MPSGraph variables hold, and the data type each op computes in.
This script only translates each primitive to its Core ML MIL equivalent, so a
new architecture option reaches Core ML without a change here; an op kind it does
not know is refused, never guessed.

    dcm_coreml_export.py MODEL.safetensors --out MODEL.mlpackage [--app BINARY]
    dcm_coreml_export.py --graph EXPORT_FOLDER --out MODEL.mlpackage

Precision (--precision):
  mirror  (default) ops the MPSGraph runs in bf16/fp16 run in fp16; ops it runs
          in fp32 (the policy/value tails, an fp32 model) stay fp32. fp16 has
          a far smaller range than bf16 (max 65504): an activation the bf16
          network carries above it becomes inf in Core ML. --parity reports
          that as non_finite_coreml_positions.
  fp16    every op in fp16 (what the Neural Engine runs).
  fp32    every op in fp32 (CPU/GPU only; the reference for parity checks).

--parity runs the app's `--coreml-parity` on the result (with --tolerance, the
run fails when a difference exceeds it).

Needs coremltools and numpy (Python 3.13; see documentation/coreml-export.md).
The app binary: --app, else $DCM_BINARY, else the newest built binary in
DerivedData (Release or Debug).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcm_arch  # noqa: E402

# InferenceGraph.currentFormatVersion: the graph format this script reads.
GRAPH_FORMAT_VERSION = 1
GRAPH_FILE = "graph.json"
WEIGHTS_FILE = "weights.bin"
OUTPUT_NAMES = ("policy", "value", "value_probs")
REDUCED_PRECISION_TYPES = ("bfloat16", "float16")
PRECISIONS = ("mirror", "fp16", "fp32")
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 128, 256)


class ExportError(Exception):
    """A graph this script cannot translate, or an export that failed."""


def app_binary(explicit):
    """The app binary to run: `explicit`, else `dcm_arch.app_binary()`."""
    if explicit:
        return explicit
    try:
        return dcm_arch.app_binary()
    except dcm_arch.ArchitectureError as error:
        raise ExportError(f"{error}; or pass --app") from None


def run_app(binary, args):
    result = subprocess.run([binary] + args, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if result.returncode != 0:
        raise ExportError(f"{os.path.basename(binary)} {' '.join(args[:1])} exited {result.returncode}: "
                          f"{result.stderr.strip()}")
    return result


def load_graph(folder):
    """The recorded graph and its weights (id -> float32 array in graph shape)."""
    with open(os.path.join(folder, GRAPH_FILE)) as handle:
        graph = json.load(handle)
    if graph.get("format_version") != GRAPH_FORMAT_VERSION:
        raise ExportError(f"graph format {graph.get('format_version')!r}; this script reads {GRAPH_FORMAT_VERSION}")
    raw = np.fromfile(os.path.join(folder, WEIGHTS_FILE), dtype="<f4")
    weights = {}
    for weight in graph["weights"]:
        start = weight["byte_offset"] // 4
        count = weight["element_count"]
        if weight["byte_offset"] % 4 or start + count > raw.size or math.prod(weight["shape"]) != count:
            raise ExportError(f"weight {weight['name']} does not fit {WEIGHTS_FILE}")
        weights[weight["id"]] = raw[start:start + count].reshape(weight["shape"]).astype(np.float32)
    expected = sum(w["element_count"] for w in graph["weights"])
    if raw.size != expected:
        raise ExportError(f"{WEIGHTS_FILE} holds {raw.size} values; the graph's weights are {expected}")
    return graph, weights


def _is_constant(value):
    return isinstance(value, np.ndarray)


def _erf(array):
    return np.vectorize(math.erf, otypes=[np.float32])(array)


# Constant folding: an op whose operands are all constants (weights, constants
# and what is computed only from them, such as the ReZero scale C*tanh(a/C) or
# a weight widened to fp32) is evaluated here in float32.
_FOLD = {
    "cast": lambda a: a["input"],
    "reshape": lambda a: a["input"].reshape(a["shape"]),
    "addition": lambda a: a["primary"] + a["secondary"],
    "subtraction": lambda a: a["primary"] - a["secondary"],
    "multiplication": lambda a: a["primary"] * a["secondary"],
    "division": lambda a: a["primary"] / a["secondary"],
    "tanh": lambda a: np.tanh(a["input"]),
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a["input"])),
    "erf": lambda a: _erf(a["input"]),
}


def translate(graph, weights, mb):
    """Emits the graph's ops into the current MIL program; returns the output
    vars by name and the recorded data type of every emitted op, by op name."""
    env = {graph["input"]["id"]: None}  # set by the caller's program input
    dtype_by_name = {}

    def build(board):
        env[graph["input"]["id"]] = board
        env.update(weights)
        for op in graph["operations"]:
            (kind, args), = op["kind"].items()
            env[op["id"]] = emit(op["id"], kind, args)
            dtype_by_name[op["id"]] = op["data_type"]
        outputs = {}
        for output in graph["outputs"]:
            value = env[output["id"]]
            if _is_constant(value):
                raise ExportError(f"output {output['name']} does not depend on the board")
            outputs[output["name"]] = mb.identity(x=value, name=output["name"])
            dtype_by_name[output["name"]] = "float32"
        return outputs

    def operand(name):
        if name not in env:
            raise ExportError(f"operand {name!r} is not defined before it is read")
        return env[name]

    def emit(op_id, kind, args):
        tensor_args = {}
        for key, value in args.items():
            if key in ("input", "primary", "secondary", "weights", "mean", "variance", "gamma", "beta"):
                tensor_args[key] = operand(value)
            elif key == "inputs":
                tensor_args[key] = [operand(v) for v in value]
            else:
                tensor_args[key] = value
        a = tensor_args
        operands = [v for k, v in a.items() if k in ("input", "primary", "secondary", "inputs")]
        flat = [x for v in operands for x in (v if isinstance(v, list) else [v])]
        if kind == "constant":
            return np.array(a["values"], dtype=np.float32).reshape(a["shape"])
        if flat and all(_is_constant(v) for v in flat):
            if kind not in _FOLD:
                raise ExportError(f"{op_id}: cannot fold a constant {kind}")
            return np.asarray(_FOLD[kind](a), dtype=np.float32)
        n = lambda suffix=None: op_id if suffix is None else f"{op_id}__{suffix}"
        if kind == "cast":
            # Precision is applied per op from the recorded data types (see
            # `compute_precision`), so a cast is the identity in the program.
            return a["input"]
        if kind == "convolution2D":
            if not _is_constant(a["weights"]):
                raise ExportError(f"{op_id}: convolution weights are not constant")
            return mb.conv(x=a["input"], weight=a["weights"], strides=[a["strideY"], a["strideX"]],
                           dilations=[a["dilationY"], a["dilationX"]], groups=a["groups"], pad_type="custom",
                           pad=[a["paddingTop"], a["paddingBottom"], a["paddingLeft"], a["paddingRight"]],
                           name=n())
        if kind == "normalize":
            x, mean, var, gamma, beta, eps = a["input"], a["mean"], a["variance"], a["gamma"], a["beta"], a["epsilon"]
            if all(_is_constant(v) for v in (mean, var, gamma, beta)):
                # Inference BN: one per-channel scale and shift.
                scale = (gamma / np.sqrt(var + np.float32(eps))).astype(np.float32)
                shift = (beta - mean * scale).astype(np.float32)
                return mb.add(x=mb.mul(x=x, y=scale, name=n("scale")), y=shift, name=n())
            centered = mb.sub(x=x, y=mean, name=n("centered"))
            inv_std = mb.rsqrt(x=mb.add(x=var, y=np.float32(eps), name=n("var_eps")), name=n("inv_std"))
            normalized = mb.mul(x=centered, y=inv_std, name=n("normalized"))
            return mb.add(x=mb.mul(x=normalized, y=gamma, name=n("scaled")), y=beta, name=n())
        if kind == "mean":
            return mb.reduce_mean(x=a["input"], axes=a["axes"], keep_dims=True, name=n())
        if kind == "variance":
            # Population variance, as MPSGraph computes it.
            mean = mb.reduce_mean(x=a["input"], axes=a["axes"], keep_dims=True, name=n("mean"))
            deviation = mb.sub(x=a["input"], y=mean, name=n("deviation"))
            square = mb.mul(x=deviation, y=deviation, name=n("square"))
            return mb.reduce_mean(x=square, axes=a["axes"], keep_dims=True, name=n())
        if kind == "reductionSum":
            return mb.reduce_sum(x=a["input"], axes=[a["axis"]], keep_dims=True, name=n())
        if kind == "reshape":
            return mb.reshape(x=a["input"], shape=a["shape"], name=n())
        if kind == "matrixMultiplication":
            return mb.matmul(x=a["primary"], y=a["secondary"], name=n())
        binary = {"addition": mb.add, "subtraction": mb.sub, "multiplication": mb.mul, "division": mb.real_div}
        if kind in binary:
            return binary[kind](x=a["primary"], y=a["secondary"], name=n())
        unary = {"tanh": mb.tanh, "sigmoid": mb.sigmoid, "erf": mb.erf, "reLU": mb.relu}
        if kind in unary:
            return unary[kind](x=a["input"], name=n())
        if kind == "leakyReLU":
            return mb.leaky_relu(x=a["input"], alpha=a["alpha"], name=n())
        if kind == "softMax":
            return mb.softmax(x=a["input"], axis=a["axis"], name=n())
        if kind == "concat":
            return mb.concat(values=a["inputs"], axis=a["dimension"], name=n())
        if kind == "slice":
            rank = len(a["input"].shape)
            begin = [0] * rank
            size = [-1] * rank
            begin[a["dimension"]] = a["start"]
            size[a["dimension"]] = a["length"]
            return mb.slice_by_size(x=a["input"], begin=begin, size=size, name=n())
        raise ExportError(f"{op_id}: unknown operation kind {kind!r}")

    return build, dtype_by_name


def compute_precision(ct, precision, dtype_by_name):
    """The convert-time precision: everything fp32, everything fp16, or (mirror)
    fp16 exactly for the ops whose recorded MPSGraph data type is reduced.
    Ops a lowering adds are named `<op id>__<part>` and follow their op. An op
    the converter adds itself (no recorded id) runs in fp16 only when the
    network runs any op in reduced precision, so an fp32 model stays all fp32."""
    if precision == "fp32":
        return ct.precision.FLOAT32
    if precision == "fp16":
        return ct.precision.FLOAT16

    any_reduced = any(dtype in REDUCED_PRECISION_TYPES for dtype in dtype_by_name.values())

    def selector(op):
        recorded = dtype_by_name.get(op.name.split("__")[0])
        return any_reduced if recorded is None else recorded in REDUCED_PRECISION_TYPES

    return ct.transform.FP16ComputePrecision(op_selector=selector)


def convert(graph, weights, precision, batch_sizes):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol

    if precision not in PRECISIONS:
        raise ExportError(f"precision {precision!r} is not one of {', '.join(PRECISIONS)}")
    shape = graph["input"]["shape"]
    if len(shape) != 4 or shape[0] != -1 or graph["input"]["data_type"] != "float32":
        raise ExportError(f"unexpected graph input {graph['input']}")
    names = sorted(o["name"] for o in graph["outputs"])
    if names != sorted(OUTPUT_NAMES):
        raise ExportError(f"graph outputs {names}; expected {sorted(OUTPUT_NAMES)}")
    build, dtype_by_name = translate(graph, weights, mb)
    input_name = graph["input"]["id"]
    planes = shape[1:]

    @mb.program(input_specs=[mb.TensorSpec(shape=(get_new_symbol(), *planes))],
                opset_version=ct.target.macOS15)
    def program(board):
        outputs = build(board)
        return tuple(outputs[name] for name in OUTPUT_NAMES)

    if program.functions["main"].inputs.keys() != {input_name}:
        raise ExportError(f"program input is {list(program.functions['main'].inputs)}; expected {input_name}")
    sizes = sorted(set(batch_sizes))
    if 1 not in sizes:
        raise ExportError("batch sizes must include 1")
    return ct.convert(
        program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name=input_name, shape=ct.EnumeratedShapes(
            shapes=[(b, *planes) for b in sizes], default=(1, *planes)), dtype=np.float32)],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUT_NAMES],
        compute_precision=compute_precision(ct, precision, dtype_by_name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("model", nargs="?", help="DCM .safetensors / .dcmmodel file")
    source.add_argument("--graph", help="folder written by --export-inference-graph")
    parser.add_argument("--out", required=True, help="new .mlpackage path")
    parser.add_argument("--app", help="DrewsChessMachine binary (default: $DCM_BINARY or newest build)")
    parser.add_argument("--precision", choices=PRECISIONS, default="mirror")
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)),
                        help="comma-separated batch sizes the model accepts (must include 1)")
    parser.add_argument("--keep-graph", help="also write the app's export folder here (must not exist)")
    parser.add_argument("--parity", action="store_true", help="run the app's --coreml-parity on the result")
    parser.add_argument("--compute-units", default="cpu", choices=("cpu", "cpu-gpu", "cpu-ne", "all"))
    parser.add_argument("--tolerance", type=float, help="with --parity: fail above this difference")
    args = parser.parse_args(argv)

    out = os.path.abspath(args.out)
    if not out.endswith(".mlpackage"):
        raise ExportError("--out must end in .mlpackage")
    if os.path.lexists(out):
        raise ExportError(f"{out} already exists")
    if args.parity and not args.model:
        raise ExportError("--parity needs the model file")
    if args.keep_graph and not args.model:
        raise ExportError("--keep-graph needs the model file (with --graph the folder already exists)")
    try:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    except ValueError:
        raise ExportError(f"--batch-sizes {args.batch_sizes!r} is not a list of whole numbers") from None
    if any(b < 1 for b in batch_sizes):
        raise ExportError("batch sizes must be positive")

    binary = app_binary(args.app) if args.model else None
    with tempfile.TemporaryDirectory(prefix="dcm-coreml-") as scratch:
        if args.model:
            folder = os.path.abspath(args.keep_graph) if args.keep_graph else os.path.join(scratch, "graph")
            run_app(binary, ["--export-inference-graph", os.path.abspath(args.model), "--out", folder])
        else:
            folder = args.graph
        graph, weights = load_graph(folder)
        model = convert(graph, weights, args.precision, batch_sizes)
        model.short_description = f"DrewsChessMachine inference graph ({args.precision})"
        # Checked again just before writing: the conversion takes minutes, and
        # coremltools' save replaces whatever is at the path.
        if os.path.lexists(out):
            raise ExportError(f"{out} appeared while converting; not replacing it")
        model.save(out)
    print(f"wrote {out} ({args.precision}, batch sizes {','.join(map(str, sorted(set(batch_sizes))))})")

    if args.parity:
        parity_args = ["--coreml-parity", os.path.abspath(args.model), "--coreml", out,
                       "--compute-units", args.compute_units]
        if args.tolerance is not None:
            parity_args += ["--tolerance", repr(args.tolerance)]
        result = subprocess.run([binary] + parity_args, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            raise ExportError(f"parity check exited {result.returncode}: {result.stderr.strip()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
