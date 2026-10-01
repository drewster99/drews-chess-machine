"""Policy-head loading and static analysis for DCM checkpoints.

Weights-only (no forward pass). Everything here reads the checkpoint bytes
directly; nothing touches the GPU or the app.

Checkpoint identity is always the safetensors `__metadata__` (`model_id`,
`training_step`, `architecture`) or, for legacy `.dcmmodel` files, the
binary header's model ID + metadata JSON — never the filename.

Policy channel layout (PolicyEncoding.swift):
  0..55   queen-style: channel = direction*7 + (distance-1);
          directions N, NE, E, SE, S, SW, W, NW (encoder frame, "N" = toward
          row 0 = forward for the side to move; the frame is flipped
          vertically for black, so files are not mirrored and E is always
          the king side).
  56..63  knight jumps: up-right(-2,+1), right-up(-1,+2), right-down(+1,+2),
          down-right(+2,+1), down-left(+2,-1), left-down(+1,-2),
          left-up(-1,-2), up-left(-2,-1).
  64..72  underpromotion: 64 + piece*3 + direction, piece knight=0, rook=1,
          bishop=2; direction forward=0, capture-left=1, capture-right=2.
  73..75  queen promotion: 73 + direction.
Flat logit index = channel*64 + row*8 + col.
"""
import hashlib
import json
import math
import os
import struct

import numpy as np

POLICY_CHANNELS = 76
QUEEN_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
KNIGHT_JUMPS = ["N-up-right", "N-right-up", "N-right-down", "N-down-right",
                "N-down-left", "N-left-down", "N-left-up", "N-up-left"]
PROMOTION_DIRECTIONS = ["fwd", "capL", "capR"]
UNDERPROMOTION_PIECES = ["knight", "rook", "bishop"]

BF16_UNIT_ROUNDOFF = 2.0 ** -8  # bf16: 8-bit significand (7 stored + implicit)
LEAKY_RELU_NEGATIVE_SLOPE = 0.01  # ActivationFunction.leakyReLUNegativeSlope


def channel_label(channel):
    if channel < 56:
        return f"Q-{QUEEN_DIRECTIONS[channel // 7]}{channel % 7 + 1}"
    if channel < 64:
        return f"Kn-{channel - 56}({KNIGHT_JUMPS[channel - 56][2:]})"
    if channel < 73:
        offset = channel - 64
        return f"UP-{UNDERPROMOTION_PIECES[offset // 3]}-{PROMOTION_DIRECTIONS[offset % 3]}"
    return f"QP-{PROMOTION_DIRECTIONS[channel - 73]}"


def channel_families():
    """Named groups of policy channels -> list of channel indices."""
    families = {}
    for distance in range(1, 8):
        families[f"queen-style dist {distance}"] = [d * 7 + distance - 1 for d in range(8)]
    for direction_index, direction in enumerate(QUEEN_DIRECTIONS):
        families[f"queen-style dir {direction}"] = [direction_index * 7 + k for k in range(7)]
    families["queen-style all"] = list(range(56))
    families["knight"] = list(range(56, 64))
    for piece_index, piece in enumerate(UNDERPROMOTION_PIECES):
        families[f"underpromo {piece}"] = [64 + piece_index * 3 + d for d in range(3)]
    for direction_index, direction in enumerate(PROMOTION_DIRECTIONS):
        families[f"underpromo dir {direction}"] = [64 + p * 3 + direction_index for p in range(3)]
    families["underpromo all"] = list(range(64, 73))
    families["queen-promo"] = list(range(73, 76))
    return families


# Left-right mirror pairs in the encoder frame (file a<->h). A position and
# its mirror are both legal chess except for castling rights, so the move-type
# priors (biases) should be close to mirror-symmetric apart from castling
# (E2 carries O-O, W2 carries O-O-O).
def mirror_channel(channel):
    if channel < 56:
        mirror_direction = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}[channel // 7]
        return mirror_direction * 7 + channel % 7
    if channel < 64:
        return 56 + {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}[channel - 56]
    if channel < 73:
        offset = channel - 64
        return 64 + (offset // 3) * 3 + {0: 0, 1: 2, 2: 1}[offset % 3]
    return 73 + {0: 0, 1: 2, 2: 1}[channel - 73]


# ---------------------------------------------------------------- loading

def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_array(raw, dtype, shape):
    if dtype == "F32":
        array = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    elif dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        array = bits.view(np.float32).astype(np.float64)
    elif dtype == "F16":
        array = np.frombuffer(raw, dtype="<f2").astype(np.float64)
    else:
        raise ValueError(f"unsupported safetensors dtype {dtype}")
    return array.reshape(shape) if shape else array


def read_safetensors_policy(path):
    """Return (metadata, tensors) with every tensor whose name contains
    'policy' (weights, BN stats and optimizer velocities), as float64
    arrays, plus the raw float32 bit patterns for bf16-exactness checks."""
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
        data_start = 8 + header_length
        metadata = header.pop("__metadata__", {})
        tensors = {}
        raw_bits = {}
        for name, info in header.items():
            if "policy" not in name:
                continue
            begin, end = info["data_offsets"]
            handle.seek(data_start + begin)
            raw = handle.read(end - begin)
            if len(raw) != end - begin:
                raise IOError(f"{path}: short read for {name}")
            tensors[name] = _decode_array(raw, info["dtype"], info["shape"])
            if info["dtype"] == "F32":
                raw_bits[name] = np.frombuffer(raw, dtype="<u4")
    return metadata, tensors, raw_bits


# Legacy .dcmmodel: positional tensor list. Order = trainables, then BN
# running stats, then (trainer files only) one velocity per trainable.
# The policy head's trainables are the last ones before the value head's
# 7 trainables (value.conv.weight, value.bn.{weight,bias}, value.fc1.{weight,
# bias}, fc2.{weight,bias}); its pre_bn running stats are the 2 running
# tensors just before value.bn's 2. Counts per archHash (from
# NetworkArchitecture.legacyDcmmodelArchHashes + the preset recipes):
LEGACY_LAYOUTS = {
    0x13BA0B55: dict(preset="v3_8block_3x3", trainables=92, running=36,
                     style="simple_conv", channels=128, pre_channels=None,
                     compute="float32", activation="relu", blocks=8, kernel=3),
    0x5347C53D: dict(preset="v3_16block_3x3", trainables=175, running=70,
                     style="intermediate_conv", channels=128, pre_channels=128,
                     compute="float32", activation="relu", blocks=16, kernel=3),
    0xBAD32CED: dict(preset="v4_12block_3x3", trainables=149, running=56,
                     style="intermediate_conv", channels=128, pre_channels=128,
                     compute="bfloat16", activation="relu", blocks=12, kernel=3),
}


def read_dcmmodel_policy(path):
    data = open(path, "rb").read()
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError(f"{path}: trailing SHA-256 mismatch")
    if data[:8] != b"DCMMODEL":
        raise ValueError(f"{path}: bad magic")
    offset = 8
    version, arch_hash, tensor_count = struct.unpack_from("<III", data, offset)
    offset += 12
    created_at, = struct.unpack_from("<q", data, offset)
    offset += 8
    id_length, = struct.unpack_from("<I", data, offset)
    offset += 4
    model_id = data[offset:offset + id_length].decode()
    offset += id_length
    meta_length, = struct.unpack_from("<I", data, offset)
    offset += 4
    legacy_meta = json.loads(data[offset:offset + meta_length])
    offset += meta_length
    arrays = []
    for expected_index in range(tensor_count):
        index, count = struct.unpack_from("<II", data, offset)
        offset += 8
        if index != expected_index:
            raise ValueError(f"{path}: tensor index {index} != {expected_index}")
        arrays.append(np.frombuffer(data, dtype="<f4", count=count, offset=offset).astype(np.float64))
        offset += 4 * count
    if offset != len(data) - 32:
        raise ValueError(f"{path}: {len(data) - 32 - offset} trailing bytes")
    if arch_hash not in LEGACY_LAYOUTS:
        raise ValueError(f"{path}: unknown legacy archHash 0x{arch_hash:08x}")
    layout = LEGACY_LAYOUTS[arch_hash]
    trainable_count, running_count = layout["trainables"], layout["running"]
    has_velocity = tensor_count == 2 * trainable_count + running_count
    if not has_velocity and tensor_count != trainable_count + running_count:
        raise ValueError(f"{path}: tensor count {tensor_count} fits neither model nor trainer layout")
    channels = layout["channels"]
    value_trainables = 7
    tensors = {}
    if layout["style"] == "simple_conv":
        names = ["policy.conv.weight", "policy.conv.bias"]
        shapes = [(76, channels, 1, 1), (76,)]
    else:
        k = layout["pre_channels"]
        names = ["policy.pre_conv.weight", "policy.pre_bn.weight", "policy.pre_bn.bias",
                 "policy.conv.weight", "policy.conv.bias"]
        shapes = [(k, channels, 1, 1), (k,), (k,), (76, k, 1, 1), (76,)]
    first = trainable_count - value_trainables - len(names)
    for position, (name, shape) in enumerate(zip(names, shapes)):
        array = arrays[first + position]
        if array.size != int(np.prod(shape)):
            raise ValueError(f"{path}: {name} has {array.size} elements, expected shape {shape}")
        tensors[name] = array.reshape(shape)
        if has_velocity:
            velocity = arrays[trainable_count + running_count + first + position]
            if velocity.size != array.size:
                raise ValueError(f"{path}: velocity for {name} has wrong size")
            tensors[f"opt.{name}.velocity"] = velocity
    if layout["style"] != "simple_conv":
        k = layout["pre_channels"]
        running_mean = arrays[trainable_count + running_count - 4]
        running_var = arrays[trainable_count + running_count - 3]
        if running_mean.size != k or running_var.size != k:
            raise ValueError(f"{path}: policy pre_bn running stats have wrong size")
        tensors["policy.pre_bn.running_mean"] = running_mean
        tensors["policy.pre_bn.running_var"] = running_var
    # value.conv.weight right after the policy trainables: sanity check size.
    value_conv = arrays[trainable_count - value_trainables]
    if value_conv.size % channels != 0:
        raise ValueError(f"{path}: tensor after policy head is not value.conv.weight")
    architecture = dict(
        legacy_preset=layout["preset"], policy_head_style=layout["style"],
        policy_pre_conv_channels=layout["pre_channels"], compute_data_type=layout["compute"],
        activation_function=layout["activation"], channels=channels,
        num_blocks=layout["blocks"], block_conv1_kernel_size=layout["kernel"])
    metadata = dict(model_id=model_id, parent_model_id=legacy_meta.get("parentModelID", ""),
                    training_step=str(legacy_meta.get("trainingStep", "")),
                    creator=legacy_meta.get("creator", ""), created_at_unix=str(created_at),
                    architecture=json.dumps(architecture), dcmmodel_version=str(version),
                    dcmmodel_arch_hash=f"0x{arch_hash:08x}")
    return metadata, tensors, {}


def read_policy(path):
    if path.endswith(".dcmmodel"):
        return read_dcmmodel_policy(path)
    return read_safetensors_policy(path)


# ---------------------------------------------------------------- arch

def architecture_summary(architecture):
    """Compact description: tower widths/blocks/kernels, policy style, K,
    compute dtype, activation."""
    if "legacy_preset" in architecture:
        return (f"{architecture['legacy_preset']} ({architecture['num_blocks']}x{architecture['block_conv1_kernel_size']}x3"
                f" @{architecture['channels']})")
    groups = architecture.get("block_groups")
    if groups:
        parts = []
        for group in groups:
            parts.append(f"{group['count']}x{group['conv1_kernel_size']}x{group['conv1_kernel_size']}@{group['channels']}"
                         f" SE:{group.get('se_style', '?')}")
        tower = " + ".join(parts)
    else:
        tower = (f"{architecture.get('num_blocks')}x{architecture.get('block_conv1_kernel_size')}x"
                 f"{architecture.get('block_conv1_kernel_size')}@{architecture.get('channels')}"
                 f" SE:{architecture.get('block_se_style', '?')}")
    return f"stem{architecture.get('stem_conv_kernel_size')} {tower}"


def policy_input_width(architecture, tensors):
    if "policy.pre_conv.weight" in tensors:
        return tensors["policy.pre_conv.weight"].shape[1]
    return tensors["policy.conv.weight"].shape[1]


# ---------------------------------------------------------------- math

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(80)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def standard_normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


def activation_function(name):
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "leaky_relu":
        return lambda x: np.where(x >= 0, x, LEAKY_RELU_NEGATIVE_SLOPE * x)
    if name == "silu":
        return lambda x: x / (1.0 + np.exp(-x))
    if name == "gelu":
        erf = np.vectorize(math.erf)
        return lambda x: 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
    raise ValueError(f"unknown activation {name}")


def post_activation_moments(gamma, beta, activation):
    """E[f(beta + gamma*z)] and Var[...] for z ~ N(0,1), per channel, by
    Gauss-Hermite quadrature. This is the post-BN, post-activation feature
    distribution under the assumption that the BN input is normalized
    exactly (running stats match the batch)."""
    f = activation_function(activation)
    x = beta[:, None] + gamma[:, None] * _GH_NODES[None, :]
    values = f(x)
    mean = (values * _GH_WEIGHTS).sum(1)
    second = (values ** 2 * _GH_WEIGHTS).sum(1)
    return mean, np.maximum(second - mean ** 2, 0.0), second


def bf16_round(x):
    """Round-to-nearest-even to bf16, returned as float64."""
    as32 = np.asarray(x, dtype=np.float32)
    bits = as32.view(np.uint32).astype(np.uint64)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = ((bits + rounding_bias) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)


def bf16_spacing(magnitude):
    """Gap between adjacent bf16 values at |magnitude| (0 -> tiny)."""
    magnitude = np.abs(np.asarray(magnitude, dtype=np.float64))
    exponent = np.floor(np.log2(np.maximum(magnitude, 1e-30)))
    return 2.0 ** (exponent - 7)


def bf16_exact_fraction(raw_bits):
    if raw_bits is None:
        return None
    return float(np.mean((raw_bits & 0xFFFF) == 0))


def nonfinite_count(array):
    return int(np.size(array) - np.isfinite(array).sum())


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=float(values.min()), p05=float(np.percentile(values, 5)),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(values.max()), mean=float(values.mean()))
