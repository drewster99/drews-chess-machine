#!/usr/bin/env python3
"""
Inspect a .dcmmodel checkpoint file and dump per-tensor statistics.

The .dcmmodel format is a flat little-endian binary (defined in
DrewsChessMachine/ModelCheckpointFile.swift):

    [  0 ..  8 ]  magic "DCMMODEL"
    [  8 .. 12 ]  u32  formatVersion
    [ 12 .. 16 ]  u32  archHash
    [ 16 .. 20 ]  u32  numTensors
    [ 20 .. 28 ]  i64  createdAtUnix
    [ 28 .. 32 ]  u32  modelIDByteCount
    [ 32 ..  m ]  utf-8 modelID
    [  m ..  m+4] u32  metadataJSONByteCount
    [  m+4..  q ] utf-8 metadataJSON
    repeated `numTensors` times in declared order:
        [ .. +4 ]  u32  tensorIndex (must match its position)
        [ .. +4 ]  u32  elementCount
        [ .. .. ]  Float32 × elementCount
    [ last 32 bytes ] SHA-256 over all preceding bytes

Usage:
    tools/dump_dcmmodel.py path/to/file.dcmmodel
    tools/dump_dcmmodel.py path/to/session.dcmsession   # picks trainer.dcmmodel
    tools/dump_dcmmodel.py --compare a.dcmmodel b.dcmmodel

Per-tensor output reports n / mean / std / min / max / |max| and
flags NaN, Inf, all-zero, sparse (<0.1% non-zero) tensors. He-init
expected std for the tensor's fan-in is shown alongside so the eye
can pick out tensors that have actually moved off init.

This script is a debugging aid; it intentionally has no third-party
deps beyond numpy. Architecture constants are hard-coded to match
ChessNetwork.swift (channels=128, numBlocks=8, etc.). If the network
shape changes, update CONSTANTS below.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# Architecture constants. Must stay in sync with ChessNetwork.swift.
CHANNELS = 128
INPUT_PLANES = 20
NUM_BLOCKS = 8
POLICY_CHANNELS = 76
POLICY_SIZE = POLICY_CHANNELS * 8 * 8
SE_REDUCED = CHANNELS // 4    # seReductionRatio = 4

# Tensor layout (declared order, matches ChessNetwork.buildGraph).
# Trainables:
#   t0: stem conv W (20*128*9 = 23040)
#   t1, t2: stem BN gamma, beta (128 each)
#   For each of 8 residual blocks (10 tensors per block):
#     conv1 W (128*128*9), BN1 g/b (128 each),
#     conv2 W (128*128*9), BN2 g/b (128 each),
#     SE-FC1 W (128*32 = 4096), SE-FC1 b (32),
#     SE-FC2 W (32*128 = 4096), SE-FC2 b (128)
#   Then policy head:
#     policy 1x1 conv W (128*76 = 9728)
#     policy bias (76)
#   Then value head:
#     value 1x1 conv W (128 elements: shape [1,128,1,1])
#     value BN gamma, beta (1 each)
#     value FC1 W (64*64 = 4096), FC1 bias (64)
#     value FC2 W (64*1 = 64), FC2 bias (1)
# BN running stats appended after all trainables, in same BN order:
#   stem BN running_mean (128), running_var (128)
#   per block × 8: BN1 mean/var, BN2 mean/var (128 each)
#   value BN running_mean (1), running_var (1)


def kaiming_std(fan_in: int) -> float:
    """He init std for a given fan-in. Matches heInitData in ChessNetwork."""
    return math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0


def label_for_index(idx: int) -> str:
    """Best-effort human-readable name for tensor index."""
    if idx == 0:
        return "stem.conv.W       (fanIn=20*9=180)"
    if idx == 1:
        return "stem.BN.gamma     (init=1)"
    if idx == 2:
        return "stem.BN.beta      (init=0)"

    base = idx - 3
    block_size = 10
    if base < NUM_BLOCKS * block_size:
        block = base // block_size
        within = base % block_size
        names = [
            f"block{block}.conv1.W   (fanIn=128*9=1152)",
            f"block{block}.BN1.gamma (init=1)",
            f"block{block}.BN1.beta  (init=0)",
            f"block{block}.conv2.W   (fanIn=128*9=1152)",
            f"block{block}.BN2.gamma (init=1)",
            f"block{block}.BN2.beta  (init=0)",
            f"block{block}.SE.fc1.W  (fanIn=128)",
            f"block{block}.SE.fc1.b  (init=0)",
            f"block{block}.SE.fc2.W  (fanIn=32)",
            f"block{block}.SE.fc2.b  (init=0)",
        ]
        return names[within]

    after_blocks = idx - 3 - NUM_BLOCKS * block_size
    if after_blocks == 0:
        return "policy.conv.W     (fanIn=128)"
    if after_blocks == 1:
        return "policy.bias       (init=0)"
    if after_blocks == 2:
        return "value.conv.W      (fanIn=128)"
    if after_blocks == 3:
        return "value.BN.gamma    (init=1)"
    if after_blocks == 4:
        return "value.BN.beta     (init=0)"
    if after_blocks == 5:
        return "value.fc1.W       (fanIn=64)"
    if after_blocks == 6:
        return "value.fc1.b       (init=0)"
    if after_blocks == 7:
        return "value.fc2.W       (fanIn=64)"
    if after_blocks == 8:
        return "value.fc2.b       (init=0)"

    # BN running stats.
    bn_offset = idx - 3 - NUM_BLOCKS * block_size - 9
    if bn_offset < 0:
        return f"<unknown idx {idx}>"
    if bn_offset == 0:
        return "stem.BN.running_mean   (init=0)"
    if bn_offset == 1:
        return "stem.BN.running_var    (init=1)"
    if bn_offset >= 2 and bn_offset < 2 + NUM_BLOCKS * 4:
        block = (bn_offset - 2) // 4
        within = (bn_offset - 2) % 4
        layer = ["BN1.running_mean (0)", "BN1.running_var (1)",
                 "BN2.running_mean (0)", "BN2.running_var (1)"][within]
        return f"block{block}.{layer}"
    if bn_offset == 2 + NUM_BLOCKS * 4:
        return "value.BN.running_mean  (init=0)"
    if bn_offset == 2 + NUM_BLOCKS * 4 + 1:
        return "value.BN.running_var   (init=1)"
    return f"<bn-stat idx {idx}>"


def init_std_hint(idx: int, n: int) -> str:
    """Return ' [init std=…]' annotation when we can identify the tensor's
    expected initialization standard deviation. Empty string otherwise.
    """
    if idx == 0:
        return f" [init std={kaiming_std(INPUT_PLANES * 9):.4f}]"
    block_size = 10
    base = idx - 3
    if 0 <= base < NUM_BLOCKS * block_size:
        within = base % block_size
        if within in (0, 3):  # conv1, conv2
            return f" [init std={kaiming_std(CHANNELS * 9):.4f}]"
        if within == 6:        # SE-FC1
            return f" [init std={kaiming_std(CHANNELS):.4f}]"
        if within == 8:        # SE-FC2
            return f" [init std={kaiming_std(SE_REDUCED):.4f}]"
    after = idx - 3 - NUM_BLOCKS * block_size
    if after == 0:
        return f" [init std={kaiming_std(CHANNELS):.4f}]"
    if after == 2:
        return f" [init std={kaiming_std(CHANNELS):.4f}]"
    if after == 5:
        return f" [init std={kaiming_std(64):.4f}]"
    if after == 7:
        return f" [init std={kaiming_std(64):.4f}]"
    return ""


def parse_dcmmodel(path: Path):
    data = path.read_bytes()
    if len(data) < 8 + 4 + 4 + 4 + 8 + 4 + 4 + 32:
        raise ValueError(f"{path}: file too short ({len(data)} bytes)")
    content = data[:-32]
    sha = data[-32:]

    import hashlib
    if hashlib.sha256(content).digest() != sha:
        print(f"WARNING {path}: trailing SHA-256 does not match content. "
              "File may be corrupt; continuing for forensic purposes.",
              file=sys.stderr)

    off = 0

    def u32():
        nonlocal off
        v = struct.unpack_from("<I", content, off)[0]; off += 4; return v

    def i64():
        nonlocal off
        v = struct.unpack_from("<q", content, off)[0]; off += 8; return v

    magic = content[off:off+8]; off += 8
    if magic != b"DCMMODEL":
        raise ValueError(f"{path}: bad magic {magic!r}")
    version = u32()
    arch_hash = u32()
    num_tensors = u32()
    created_unix = i64()
    id_len = u32()
    model_id = content[off:off+id_len].decode("utf-8"); off += id_len
    meta_len = u32()
    meta_raw = content[off:off+meta_len].decode("utf-8"); off += meta_len
    metadata = json.loads(meta_raw)

    tensors = []
    while off < len(content):
        if len(content) - off < 8:
            break
        idx = u32()
        n = u32()
        raw = content[off:off+n*4]; off += n * 4
        arr = np.frombuffer(raw, dtype=np.float32)
        tensors.append((idx, arr))

    return {
        "version": version,
        "arch_hash": arch_hash,
        "num_tensors": num_tensors,
        "created_unix": created_unix,
        "model_id": model_id,
        "metadata": metadata,
        "tensors": tensors,
    }


def stat_line(idx: int, arr: np.ndarray) -> str:
    flags = []
    if np.isnan(arr).any():
        flags.append("NaN")
    if np.isinf(arr).any():
        flags.append("Inf")
    nz = int(np.count_nonzero(arr))
    pct = nz / len(arr) if len(arr) else 0.0
    if pct == 0.0:
        flags.append("ALL-ZERO")
    elif pct < 0.001:
        flags.append(f"sparse={pct:.4f}")
    abs_arr = np.abs(arr)
    flag_str = " ".join(flags)
    return (
        f"  t{idx:>3} n={len(arr):>7,} "
        f"mean={arr.mean():+.4e} std={arr.std():.4e} "
        f"min={arr.min():+.4e} max={arr.max():+.4e} "
        f"|max|={abs_arr.max():.4e}{init_std_hint(idx, len(arr))} "
        f"{flag_str}"
    ).rstrip()


def dump_one(path: Path) -> None:
    parsed = parse_dcmmodel(path)
    print(f"file: {path.name}  size={path.stat().st_size:,} bytes")
    print(f"  version={parsed['version']}  archHash=0x{parsed['arch_hash']:08x}  "
          f"numTensors={parsed['num_tensors']}")
    print(f"  createdAtUnix={parsed['created_unix']}  modelID={parsed['model_id']}")
    print(f"  metadata: {parsed['metadata']}")

    print(f"\n=== tensors ({len(parsed['tensors'])}) ===")
    for idx, arr in parsed["tensors"]:
        print(stat_line(idx, arr))
        # On the line right after, print a label for human-readable lookup.
        print(f"        ↳ {label_for_index(idx)}")


def compare(path_a: Path, path_b: Path) -> None:
    a = parse_dcmmodel(path_a)
    b = parse_dcmmodel(path_b)
    if a["arch_hash"] != b["arch_hash"]:
        print(f"WARNING archHash differs: a=0x{a['arch_hash']:08x}, b=0x{b['arch_hash']:08x}",
              file=sys.stderr)
    if len(a["tensors"]) != len(b["tensors"]):
        raise SystemExit(
            f"tensor count differs: a={len(a['tensors'])}, b={len(b['tensors'])}"
        )
    print(f"compare: {path_a.name} -> {path_b.name}")
    print(f"  a.modelID={a['model_id']} (step={a['metadata'].get('trainingStep')})")
    print(f"  b.modelID={b['model_id']} (step={b['metadata'].get('trainingStep')})")
    print()
    print("  per-tensor delta L2 / max |Δ| / Δstd")
    for (ai, av), (bi, bv) in zip(a["tensors"], b["tensors"]):
        if ai != bi or av.shape != bv.shape:
            print(f"  t{ai}: shape/index mismatch — skipping")
            continue
        d = bv - av
        l2 = float(np.linalg.norm(d))
        max_abs = float(np.max(np.abs(d)))
        d_std = float(bv.std() - av.std())
        print(f"  t{ai:>3} L2(Δ)={l2:.4e}  maxAbsΔ={max_abs:.4e}  "
              f"Δstd={d_std:+.4e}  ↳ {label_for_index(ai)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="Path to .dcmmodel or .dcmsession")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two checkpoint files; show per-tensor deltas.")
    args = parser.parse_args()

    if args.compare:
        a, b = (Path(p) for p in args.compare)
        compare(_resolve_to_model(a), _resolve_to_model(b))
        return

    if not args.path:
        parser.error("path argument required (or use --compare A B)")

    dump_one(_resolve_to_model(Path(args.path)))


def _resolve_to_model(p: Path) -> Path:
    """Accept either a .dcmmodel file or a .dcmsession directory.
    For sessions, prefer trainer.dcmmodel since that's the live training
    lineage; champion.dcmmodel is a snapshot at the last promotion.
    """
    if p.is_dir():
        trainer = p / "trainer.dcmmodel"
        if trainer.exists():
            return trainer
        champion = p / "champion.dcmmodel"
        if champion.exists():
            return champion
        raise SystemExit(f"{p} is a directory but contains no .dcmmodel files")
    return p


if __name__ == "__main__":
    main()
