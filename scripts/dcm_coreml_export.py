"""Turn a DCM model file into a Core ML model (.mlpackage).

The network is never described here. The app records its own inference forward
pass from the graph builders (`--export-inference-graph`, `InferenceGraph` in
Network/InferenceGraph.swift): every primitive op in build order, the data type
each op computes in, and the weights as the MPSGraph variables hold them. This
script only translates each primitive to its Core ML MIL equivalent, so a new
architecture option reaches Core ML without a change here; an op kind it does
not know is refused, never guessed.

    dcm_coreml_export.py MODEL.safetensors --out MODEL.mlpackage [--app BINARY]
    dcm_coreml_export.py --graph EXPORT_FOLDER --out MODEL.mlpackage

Precision (--precision). Every op is emitted in an explicit type and the
converter adds no precision pass, so the types below are what Core ML runs:
  mirror  (default) the MPSGraph's own types: an op it runs in bf16/fp16 runs in
          fp16, one it runs in fp32 (the policy/value tails, an fp32 model) in
          fp32, with the recorded casts between them. fp16 has a far smaller
          range than bf16 (max 65504): an activation the bf16 network carries
          above it becomes inf in Core ML. --parity reports that as
          non_finite_coreml_positions.
  fp16    every op in fp16 (what the Neural Engine runs).
  fp32    every op in fp32 (CPU/GPU only; the reference for parity checks).

--parity runs the app's `--coreml-parity` on the result (with --tolerance, the
run fails when a difference exceeds it).

Needs coremltools and numpy (Python 3.13; see documentation/coreml-export.md).
The app binary: --app, else `dcm_arch.app_binary()` ($DCM_BINARY, else the newest
build of this repository's project).
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
GRAPH_FORMAT_VERSION = 2
GRAPH_FILE = "graph.json"
WEIGHTS_FILE = "weights.bin"
OUTPUT_NAMES = ("policy", "value", "value_probs")
REDUCED_PRECISION_TYPES = ("bfloat16", "float16")
GRAPH_DATA_TYPES = REDUCED_PRECISION_TYPES + ("float32",)
PRECISIONS = ("mirror", "fp16", "fp32")
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 128, 256)
# The app's export and parity commands build an MPSGraph and run the 4,635
# test-set positions; minutes at most. A hung app is reported, not waited on.
APP_TIMEOUT_SECONDS = 1800
NUMPY_TYPE = {"fp16": np.float16, "fp32": np.float32}


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
    try:
        return subprocess.run([binary] + args, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=APP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise ExportError(f"{os.path.basename(binary)} {args[0]} did not finish in {APP_TIMEOUT_SECONDS} s") from None
    except OSError as error:
        raise ExportError(f"cannot run {binary}: {error}") from None


def load_graph(folder):
    """The recorded graph and its weights (id -> float32 array in graph shape),
    after checking that ids are unique and the weights tile weights.bin exactly."""
    with open(os.path.join(folder, GRAPH_FILE)) as handle:
        graph = json.load(handle)
    if graph.get("format_version") != GRAPH_FORMAT_VERSION:
        raise ExportError(f"graph format {graph.get('format_version')!r}; this script reads {GRAPH_FORMAT_VERSION}")
    ids = [graph["input"]["id"]] + [w["id"] for w in graph["weights"]] + [op["id"] for op in graph["operations"]]
    seen, duplicates = set(), set()
    for identifier in ids:
        (duplicates if identifier in seen else seen).add(identifier)
    if duplicates:
        raise ExportError(f"graph ids used more than once: {', '.join(sorted(duplicates))}")
    raw = np.fromfile(os.path.join(folder, WEIGHTS_FILE), dtype="<f4")
    weights = {}
    cursor = 0
    for weight in sorted(graph["weights"], key=lambda w: w["byte_offset"]):
        count = weight["element_count"]
        if weight["byte_offset"] != cursor * 4 or math.prod(weight["shape"]) != count:
            raise ExportError(f"weight {weight['name']} does not continue {WEIGHTS_FILE} at byte {cursor * 4}")
        if weight["data_type"] not in GRAPH_DATA_TYPES:
            raise ExportError(f"weight {weight['name']} has data type {weight['data_type']!r}")
        if cursor + count > raw.size:
            raise ExportError(f"weight {weight['name']} runs past the end of {WEIGHTS_FILE}")
        weights[weight["id"]] = raw[cursor:cursor + count].reshape(weight["shape"]).astype(np.float32)
        cursor += count
    if raw.size != cursor:
        raise ExportError(f"{WEIGHTS_FILE} holds {raw.size} values; the graph's weights are {cursor}")
    return graph, weights


def _is_constant(value):
    return isinstance(value, np.ndarray)


def _erf(array):
    return np.vectorize(math.erf, otypes=[np.float64])(array)


# Constant folding: an op whose operands are all constants (weights, constants
# and what is computed only from them, such as the ReZero scale C*tanh(a/C) or
# a weight widened to fp32) is evaluated here in float64, then stored in the
# op's type.
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


def mil_type(recorded, precision):
    """The MIL type ('fp16' / 'fp32') an op recorded as `recorded` runs in."""
    if recorded not in GRAPH_DATA_TYPES:
        raise ExportError(f"unknown recorded data type {recorded!r}")
    if precision not in PRECISIONS:
        raise ExportError(f"precision {precision!r} is not one of {', '.join(PRECISIONS)}")
    if precision != "mirror":
        return precision
    return "fp16" if recorded in REDUCED_PRECISION_TYPES else "fp32"


def translate(graph, weights, mb, types, precision):
    """A function that emits the graph's ops into the current MIL program and
    returns the output vars by name, each op in its explicit type (`mil_type`).
    `types` is coremltools' MIL types module."""
    weight_types = {w["id"]: mil_type(w["data_type"], precision) for w in graph["weights"]}
    mil_types = {"fp16": types.fp16, "fp32": types.fp32}

    def operand(env, name):
        if name not in env:
            raise ExportError(f"operand {name!r} is not defined before it is read")
        return env[name]

    def emit(env, op_id, kind, args, out_type):
        a = {}
        for key, value in args.items():
            if key in ("input", "primary", "secondary", "weights", "mean", "variance", "gamma", "beta"):
                a[key] = operand(env, value)
            elif key == "inputs":
                a[key] = [operand(env, v) for v in value]
            else:
                a[key] = value
        np_type = NUMPY_TYPE[out_type]
        if kind == "constant":
            return np.array(a["values"], dtype=np.float64).reshape(a["shape"]).astype(np_type)
        tensors = [v for k, v in a.items() if k in ("input", "primary", "secondary")] + a.get("inputs", [])
        if tensors and all(_is_constant(v) for v in tensors):
            if kind not in _FOLD:
                raise ExportError(f"{op_id}: cannot fold a constant {kind}")
            wide = {k: (v.astype(np.float64) if _is_constant(v) else v) for k, v in a.items()}
            return np.asarray(_FOLD[kind](wide)).astype(np_type)
        n = lambda suffix=None: op_id if suffix is None else f"{op_id}__{suffix}"
        if kind == "cast":
            if a["input"].dtype == mil_types[out_type]:
                return a["input"]
            return mb.cast(x=a["input"], dtype=out_type, name=n())
        if kind == "convolution2D":
            if not _is_constant(a["weights"]):
                raise ExportError(f"{op_id}: convolution weights are not constant")
            return mb.conv(x=a["input"], weight=a["weights"], strides=[a["strideY"], a["strideX"]],
                           dilations=[a["dilationY"], a["dilationX"]], groups=a["groups"], pad_type="custom",
                           pad=[a["paddingTop"], a["paddingBottom"], a["paddingLeft"], a["paddingRight"]],
                           name=n())
        if kind == "normalize":
            x, mean, var, gamma, beta, eps = a["input"], a["mean"], a["variance"], a["gamma"], a["beta"], a["epsilon"]
            centered = mb.sub(x=x, y=mean, name=n("centered"))
            if _is_constant(var) and _is_constant(gamma):
                # Inference BN: subtract the mean first, as MPSGraph does, then one
                # per-channel scale (gamma / sqrt(var + eps)), then beta.
                scale = (gamma.astype(np.float64) / np.sqrt(var.astype(np.float64) + eps)).astype(np_type)
                return mb.add(x=mb.mul(x=centered, y=scale, name=n("scaled")), y=beta, name=n())
            inv_std = mb.rsqrt(x=mb.add(x=var, y=np_type(eps), name=n("var_eps")), name=n("inv_std"))
            normalized = mb.mul(x=centered, y=inv_std, name=n("normalized"))
            return mb.add(x=mb.mul(x=normalized, y=gamma, name=n("gamma")), y=beta, name=n())
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

    def build(board):
        env = {}
        input_type = mil_type(graph["input"]["data_type"], precision)
        env[graph["input"]["id"]] = board if input_type == "fp32" else mb.cast(x=board, dtype=input_type,
                                                                             name="board__cast")
        for weight_id, values in weights.items():
            env[weight_id] = values.astype(NUMPY_TYPE[weight_types[weight_id]])
        for op in graph["operations"]:
            (kind, args), = op["kind"].items()
            env[op["id"]] = emit(env, op["id"], kind, args, mil_type(op["data_type"], precision))
        outputs = {}
        for output in graph["outputs"]:
            value = env[output["id"]]
            if _is_constant(value):
                raise ExportError(f"output {output['name']} does not depend on the board")
            if value.dtype != types.fp32:
                value = mb.cast(x=value, dtype="fp32", name=f"{output['name']}__fp32")
            outputs[output["name"]] = mb.identity(x=value, name=output["name"])
        return outputs

    return build


def convert(graph, weights, precision, batch_sizes):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol, types

    shape = graph["input"]["shape"]
    if len(shape) != 4 or shape[0] != -1 or graph["input"]["data_type"] != "float32":
        raise ExportError(f"unexpected graph input {graph['input']}")
    names = sorted(o["name"] for o in graph["outputs"])
    if names != sorted(OUTPUT_NAMES):
        raise ExportError(f"graph outputs {names}; expected {sorted(OUTPUT_NAMES)}")
    build = translate(graph, weights, mb, types, precision)
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
    # coremltools 9's fuse_elementwise_to_batchnorm builds a batch_norm whose
    # mean is fp32 beside fp16 gamma/beta when the BN constants are fp16, and
    # the op then fails validation; the BN is folded into the convolution by
    # the conv fusion passes either way.
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::fuse_elementwise_to_batchnorm"])
    # Every op already has its type; FLOAT32 tells the converter to add none.
    return ct.convert(
        program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name=input_name, shape=ct.EnumeratedShapes(
            shapes=[(b, *planes) for b in sizes], default=(1, *planes)), dtype=np.float32)],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUT_NAMES],
        compute_precision=ct.precision.FLOAT32, pass_pipeline=pipeline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("model", nargs="?", help="DCM .safetensors file")
    source.add_argument("--graph", help="folder written by --export-inference-graph")
    parser.add_argument("--out", required=True, help="new .mlpackage path")
    parser.add_argument("--app", help="DrewsChessMachine binary (default: dcm_arch.app_binary())")
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
            result = run_app(binary, ["--export-inference-graph", os.path.abspath(args.model), "--out", folder])
            if result.returncode != 0:
                raise ExportError(f"--export-inference-graph exited {result.returncode}: {result.stderr.strip()}")
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
        result = run_app(binary, parity_args)
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
