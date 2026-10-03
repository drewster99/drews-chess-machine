"""Shared loading, layout and per-unit helpers for the full-model dead-unit analysis.

Weights-only, CPU-only (numpy). Nothing here runs the app or touches the GPU.

Checkpoint identity is always the safetensors `__metadata__` (`model_id`,
`training_step`, `architecture`), never the filename. Filenames are only used
to *find* candidate files; every candidate is then re-identified from its
header and grouped by ModelID.

Tensor layouts (verified in `layout_check.py`, results in
`results/layout_check.md`):

- Saved weights follow the torch convention: conv `[O, I, kh, kw]`, linear
  `[out, in]`. BN / LayerNorm / bias vectors are 1-D.
- Optimizer velocity (`opt.<name>.velocity`) is stored flat in the *graph's*
  native layout: conv `[O, I, kh, kw]` (same as the saved weight), linear
  `[in, out]` (the transpose of the saved weight). `velocity_as_weight()`
  returns the velocity rearranged into the saved weight's shape so the two can
  be indexed the same way.

Optimizer facts used by the definitions (ChessTrainer.swift, SGD update):

- `v_new = mu * v_old + clip * grad` -- weight decay is *decoupled*, it never
  enters the velocity. A velocity of exactly 0 therefore means the gradient was
  exactly 0 for long enough that momentum decayed below the smallest fp32
  subnormal (hundreds to thousands of consecutive zero-gradient steps).
- `w -= lr * v_new + lr * wd * w` for decayed tensors (conv weights, FC
  weights); BN gamma/beta, LayerNorm gamma/beta, FC/conv biases and the ReZero
  alpha are not decayed. So a decayed weight slice with zero gradient keeps its
  direction exactly (cosine with init = 1) and shrinks by the same factor as
  every other zero-gradient decayed slice; an undecayed parameter with zero
  gradient stays bit-identical to its init.
"""
import json
import math
import os
import re
import struct

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "scripts"))
import dcm_arch  # noqa: E402  the single source of the version-gated architecture rules

MODELS_DIR = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))

# run key -> (description, filename stem shared by the fresh net and its enumerated checkpoints)
RUNS = {
    "leaky": ("leaky-FC1 (SE FC1 leaky_relu 0.01), seed 1 init", "20261001-test_SE_scale+bias-fc1leaky"),
    "relu": ("ReLU scale+bias seed 1 (same init as leaky)", "20260929-test_SE_scale+bias"),
    "relu_s2": ("ReLU scale+bias seed 2 (different init, saves velocity)", "20260929-test_SE_scale+bias-seed2"),
}

LEAKY_RELU_NEGATIVE_SLOPE = 0.01  # ActivationFunction.leakyReLUNegativeSlope

# --- Definitions (thresholds) -------------------------------------------------
# "Unmoved from init" for a decayed slice: direction kept (cosine with the fresh
# slice above this) -- TENSOR-STATS.md's "dead almost the whole run" test.
UNMOVED_COSINE = 0.999995
# ...and its norm ratio within this relative tolerance of the reference
# decay-only factor (measured on the stem weights reading always-zero planes).
DECAY_RATIO_RELATIVE_TOLERANCE = 2e-3
# "Low velocity": unit velocity L2 norm below this fraction of the site median.
LOW_VELOCITY_FRACTION = 0.05
# "High velocity outlier": unit velocity norm above this multiple of the site median.
HIGH_VELOCITY_MULTIPLE = 10.0
# BN followed by ReLU: beta/|gamma| bands (the channel's z-score where the ReLU
# switches; P(on) = Phi(beta/|gamma|) if the BN input is Gaussian).
DEAD_BETA_OVER_GAMMA = -3.0
MOSTLY_OFF_BETA_OVER_GAMMA = -2.0
ALWAYS_ON_BETA_OVER_GAMMA = 3.0
# BN running variance outlier: ratio to the BN's median.
RUNNING_VAR_HIGH_RATIO = 20.0
RUNNING_VAR_LOW_RATIO = 0.05
# LayerNorm gamma near zero (channel becomes a constant beta in the stream).
LN_GAMMA_NEAR_ZERO = 0.1

ALWAYS_ZERO_PLANES = [19, 20, 21, 22, 24, 26, 28]
PLANE_LABELS = (
    ["own P", "own N", "own B", "own R", "own Q", "own K",
     "opp P", "opp N", "opp B", "opp R", "opp Q", "opp K",
     "own O-O", "own O-O-O", "opp O-O", "opp O-O-O",
     "en passant", "halfmove clock", "rep >=1 before", "rep >=2 before"]
    + [f"pos {i + 1} plies ago" for i in range(10)]
)


# --- Reading ------------------------------------------------------------------

def read_header(path):
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    metadata = header.pop("__metadata__", {})
    return metadata, header, 8 + header_length


def _decode(raw, dtype):
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


class Checkpoint:
    """All tensors of one safetensors file, as float64 arrays in their saved
    shapes, plus fp32 bit patterns (for bf16-grid checks) and metadata."""

    def __init__(self, path):
        self.path = path
        self.file = os.path.basename(path)
        self.metadata, header, data_start = read_header(path)
        self.model_id = self.metadata["model_id"]
        self.training_step = int(self.metadata.get("training_step", "0"))
        # Version-gated: fields a file's format predates resolve the way the app
        # resolves them, and a v6+ file without `rezero_alpha_cap` is refused.
        self.architecture = dcm_arch.norm_arch_md(self.metadata)
        self.rezero_blocks = dcm_arch.rezero_blocks(self.metadata)
        self.tensors = {}
        self.f32 = {}
        with open(path, "rb") as handle:
            for name, info in header.items():
                begin, end = info["data_offsets"]
                handle.seek(data_start + begin)
                raw = handle.read(end - begin)
                if len(raw) != end - begin:
                    raise IOError(f"{path}: short read for {name}")
                values32 = _decode(raw, info["dtype"])
                self.f32[name] = values32
                self.tensors[name] = values32.astype(np.float64).reshape(info["shape"])
        self.has_velocity = any(n.startswith("opt.") for n in self.tensors)

    def __getitem__(self, name):
        return self.tensors[name]

    def __contains__(self, name):
        return name in self.tensors

    def velocity(self, name):
        """Velocity rearranged into the saved weight's shape, or None when the
        file carries no optimizer state."""
        key = f"opt.{name}.velocity"
        if key not in self.tensors:
            if self.has_velocity:
                raise KeyError(f"{self.file}: has optimizer state but no {key}")
            return None
        return velocity_as_weight(self.tensors[key], self.tensors[name].shape, name)


def is_linear_weight(name, shape):
    return len(shape) == 2 and name.endswith(".weight")


def velocity_as_weight(flat, weight_shape, name, linear_layout="in_out"):
    """Return velocity in the saved weight's shape. Conv / vector velocities are
    stored in the same order as the saved tensor; linear-weight velocities are
    stored in the graph's [in, out] layout (`linear_layout='in_out'`). The
    alternative 'out_in' exists only for `layout_check.py`."""
    flat = np.asarray(flat).reshape(-1)
    size = int(np.prod(weight_shape))
    if flat.size != size:
        raise ValueError(f"{name}: velocity has {flat.size} elements, weight {weight_shape}")
    if is_linear_weight(name, weight_shape):
        out_dim, in_dim = weight_shape
        if linear_layout == "in_out":
            return flat.reshape(in_dim, out_dim).T
        if linear_layout == "out_in":
            return flat.reshape(out_dim, in_dim)
        raise ValueError(linear_layout)
    return flat.reshape(weight_shape)


def discover(run):
    """{training_step: path} for one run's fresh net + enumerated checkpoints,
    identified by metadata. Raises if trained checkpoints span ModelIDs or two
    files claim the same step with different content."""
    stem = RUNS[run][1]
    pattern = re.compile(re.escape(stem) + r"-(fresh|replay-step\d+)\.safetensors$")
    found = {}
    trained_ids = set()
    fresh_id = None
    for file_name in sorted(os.listdir(MODELS_DIR)):
        if not pattern.match(file_name):
            continue
        path = os.path.join(MODELS_DIR, file_name)
        metadata, _, _ = read_header(path)
        step = int(metadata.get("training_step", "0"))
        if file_name.endswith("-fresh.safetensors"):
            if step != 0:
                raise ValueError(f"{file_name}: fresh net claims training_step {step}")
            fresh_id = metadata["model_id"]
        else:
            trained_ids.add(metadata["model_id"])
        if step in found:
            raise ValueError(f"{run}: two files claim step {step}: {found[step]} and {path}")
        found[step] = path
    if len(trained_ids) != 1:
        raise ValueError(f"{run}: trained checkpoints span ModelIDs {sorted(trained_ids)}")
    if fresh_id is None:
        raise FileNotFoundError(f"{run}: no fresh net")
    return dict(sorted(found.items())), fresh_id, next(iter(trained_ids))


# --- Small math helpers -------------------------------------------------------

def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rows_norm(a):
    a = np.asarray(a)
    return np.sqrt((a.reshape(a.shape[0], -1) ** 2).sum(1))


def rows_cos(a, b):
    a2 = a.reshape(a.shape[0], -1)
    b2 = b.reshape(b.shape[0], -1)
    denominator = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a2 * b2).sum(1) / denominator, np.nan)


def rows_all_zero(a):
    return np.all(a.reshape(a.shape[0], -1) == 0.0, axis=1)


def rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a, b):
    ra, rb = rank(np.asarray(a).ravel()), rank(np.asarray(b).ravel())
    return float(np.corrcoef(ra, rb)[0, 1])


def bf16_grid_fraction(values32):
    bits = np.asarray(values32, dtype=np.float32).view(np.uint32)
    return float(np.mean((bits & 0xFFFF) == 0))


def run_rezero_blocks(run):
    """Per-block ReZero settings (init, cap) of a run, read from its fresh net's
    metadata. Every checkpoint of a run shares one architecture, so the fresh net
    speaks for all of them."""
    found, _, _ = discover(run)
    if 0 not in found:
        raise FileNotFoundError(f"{run}: no fresh net to read the architecture from")
    return dcm_arch.rezero_blocks_of_file(found[0])
