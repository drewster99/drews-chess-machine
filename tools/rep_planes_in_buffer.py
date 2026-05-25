#!/usr/bin/env python3
"""
Replay-buffer rep-plane analysis.

(1) How often is each of planes 18..29 (the 12 repetition-related
    planes) actually set in stored training positions?

(2) Conditional outcome distribution: for each rep plane, what is
    P(outcome=win|plane on), P(outcome=draw|plane on), P(outcome=loss|plane on),
    compared to the same conditioned on plane OFF? If the network's
    "value goes up when rep planes are on" behavior reflects real
    training-data correlation, we should see win-skewed outcomes
    when the planes are on.

Reads the most recent .dcmsession/replay_buffer.bin per the format in
DrewsChessMachine/Training/ReplayBuffer.swift v7:

    [  0 ..  8 ]  magic "DCMRPBUF"
    [  8 .. 12 ]  u32 version (=7)
    [ 12 .. 16 ]  u32 pad
    [ 16 .. 24 ]  i64 floatsPerBoard (= 1920)
    [ 24 .. 32 ]  i64 capacity
    [ 32 .. 40 ]  i64 storedCount
    [ 40 .. 48 ]  i64 writeIndex
    [ 48 .. 56 ]  i64 totalPositionsAdded
    Body (only if storedCount > 0), in ring-walk order starting at
    `startIndex = (stored == cap) ? writeIndex : 0`:
        boards         : storedCount × 1920 × Float32 LE
        moves          : storedCount × Int32 LE
        outcomes       : storedCount × Float32 LE   (+1 / 0 / -1, STM-relative)
        plyIndex       : storedCount × UInt16 LE
        gameLength     : storedCount × UInt16 LE
        samplingTau    : storedCount × Float32 LE
        stateHash      : storedCount × UInt64 LE
        workerGameId   : storedCount × UInt32 LE
        materialCount  : storedCount × UInt8
    [last 32 bytes] SHA-256 trailer
"""

from __future__ import annotations

import argparse
import mmap
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np

INPUT_PLANES = 30
BOARD_CELLS = 64
PER_BOARD_FLOATS = INPUT_PLANES * BOARD_CELLS    # 1920
PER_BOARD_BYTES = PER_BOARD_FLOATS * 4           # 7680

# Per-position trailing metadata bytes (after boards).
# moves(4) + outcomes(4) + ply(2) + gameLen(2) + tau(4) + hash(8) + wgid(4) + mat(1) = 29
PER_POSITION_META_BYTES = 4 + 4 + 2 + 2 + 4 + 8 + 4 + 1

HEADER_SIZE = 8 + 4 + 4 + 8 * 5   # 52
TRAILER_SIZE = 32

REP_PLANES = list(range(18, 30))   # planes 18..29
PLANE_LABELS = {
    18: "rep>=1",
    19: "rep>=2",
    20: "mask t-1",
    21: "mask t-2",
    22: "mask t-3",
    23: "mask t-4",
    24: "mask t-5",
    25: "mask t-6",
    26: "mask t-7",
    27: "mask t-8",
    28: "mask t-9",
    29: "mask t-10",
}


def find_default_buffer() -> Path:
    root = Path.home() / "Library" / "Application Support" / "DrewsChessMachine" / "Sessions"
    if not root.exists():
        raise SystemExit(f"No Sessions directory at {root}")
    candidates: list[tuple[float, Path]] = []
    for d in root.glob("*.dcmsession"):
        bufp = d / "replay_buffer.bin"
        if bufp.exists():
            candidates.append((bufp.stat().st_mtime, bufp))
    if not candidates:
        raise SystemExit(f"No replay_buffer.bin found under {root}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_header(buf: mmap.mmap) -> dict:
    magic = bytes(buf[0:8])
    if magic != b"DCMRPBUF":
        raise SystemExit(f"bad magic: {magic!r}")
    version, _pad = struct.unpack_from("<II", buf, 8)
    floats_per_board, capacity, stored, write_idx, total_added = struct.unpack_from("<qqqqq", buf, 16)
    if floats_per_board != PER_BOARD_FLOATS:
        raise SystemExit(
            f"unexpected floatsPerBoard={floats_per_board}, expected {PER_BOARD_FLOATS}; "
            "input planes may have changed"
        )
    return {
        "version": version,
        "floats_per_board": floats_per_board,
        "capacity": capacity,
        "stored": stored,
        "write_idx": write_idx,
        "total_added": total_added,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="replay_buffer.bin file or .dcmsession dir")
    ap.add_argument("-n", "--sample-count", type=int, default=50000,
                    help="Number of positions to sample (default 50000)")
    ap.add_argument("--seed", type=int, default=20260524, help="RNG seed")
    args = ap.parse_args()

    if args.path:
        p = Path(args.path)
        if p.is_dir():
            p = p / "replay_buffer.bin"
        buffer_path = p
    else:
        buffer_path = find_default_buffer()
    print(f"reading: {buffer_path}")
    file_size = buffer_path.stat().st_size
    print(f"  file size: {file_size:,} bytes ({file_size/2**30:.2f} GiB)")

    with open(buffer_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            hdr = parse_header(mm)
            print(f"  version={hdr['version']}  capacity={hdr['capacity']:,}  "
                  f"stored={hdr['stored']:,}  writeIndex={hdr['write_idx']:,}  "
                  f"totalAdded={hdr['total_added']:,}")

            stored = hdr["stored"]
            if stored <= 0:
                raise SystemExit("buffer is empty")

            # Section offsets. All metadata sections follow boards, in declared order.
            boards_off = HEADER_SIZE
            moves_off = boards_off + stored * PER_BOARD_BYTES
            outcomes_off = moves_off + stored * 4
            ply_off = outcomes_off + stored * 4
            gamelen_off = ply_off + stored * 2
            tau_off = gamelen_off + stored * 2
            hash_off = tau_off + stored * 4
            wgid_off = hash_off + stored * 8
            mat_off = wgid_off + stored * 4
            expected_end = mat_off + stored * 1 + TRAILER_SIZE
            if expected_end != file_size:
                print(f"  WARNING: expected end={expected_end}, file size={file_size}, "
                      f"diff={file_size - expected_end}", file=sys.stderr)

            # Pick random sample indices in [0, stored).
            rng = random.Random(args.seed)
            n = min(args.sample_count, stored)
            sample_idx = sorted(rng.sample(range(stored), n))
            print(f"\n  sampling {n:,} positions of {stored:,} stored ({100*n/stored:.2f}%)")

            # Per-plane "on" tally and conditional outcome tallies.
            # planes 18..29 → 12 planes.
            # Each plane is broadcast (all 64 cells the same) so we
            # just read cell 0 (the first float of the plane).
            #
            # Outcomes are +1 (win), 0 (draw), -1 (loss) STM-relative.

            on_counts = np.zeros(len(REP_PLANES), dtype=np.int64)
            # outcome_on[k] = [wins, draws, losses] when plane k is ON
            # outcome_off[k] = same when plane k is OFF
            outcome_on = np.zeros((len(REP_PLANES), 3), dtype=np.int64)
            outcome_off = np.zeros((len(REP_PLANES), 3), dtype=np.int64)
            overall_outcome = np.zeros(3, dtype=np.int64)   # [W, D, L] across all sampled

            # Also count per-position number of mask bits set (bits 20..29 → bit count 0..10).
            mask_bits_histogram = np.zeros(11, dtype=np.int64)

            for i in sample_idx:
                # Read the first float of each rep plane (cell 0).
                # Plane k's cell 0 lives at offset boards_off + i * PER_BOARD_BYTES + k * 64 * 4.
                # We do 12 small reads. Each is 4 bytes. At ~50k positions
                # × 12 reads = 600k 4-byte reads. mmap makes this fast.
                base = boards_off + i * PER_BOARD_BYTES
                # We want planes 18..29. Read planes 18..29 cell 0 in one
                # slice — they're contiguous (12 planes × 64 floats apart).
                # Just read the 12 (cell 0)'s by stride.
                plane_on = np.zeros(len(REP_PLANES), dtype=bool)
                for j, k in enumerate(REP_PLANES):
                    val = struct.unpack_from("<f", mm, base + k * 64 * 4)[0]
                    plane_on[j] = val > 0.5

                # Outcome
                outcome_f = struct.unpack_from("<f", mm, outcomes_off + i * 4)[0]
                # Quantize to {+1, 0, -1} bucket
                if outcome_f > 0.5:
                    outcome_idx = 0   # win
                elif outcome_f < -0.5:
                    outcome_idx = 2   # loss
                else:
                    outcome_idx = 1   # draw
                overall_outcome[outcome_idx] += 1

                # Update tallies
                for j, on in enumerate(plane_on):
                    if on:
                        on_counts[j] += 1
                        outcome_on[j, outcome_idx] += 1
                    else:
                        outcome_off[j, outcome_idx] += 1

                # Mask-bits-set histogram (planes 20..29 → indices 2..11 in REP_PLANES)
                n_mask_on = int(plane_on[2:].sum())
                mask_bits_histogram[n_mask_on] += 1

            # ---- Report ----
            print()
            print("=== rep-plane occupancy in replay buffer ===")
            print(f"  ({n:,} positions sampled)")
            print(f"  {'plane':>5}  {'label':<10}  {'count_on':>12}  {'frac_on':>10}")
            for j, k in enumerate(REP_PLANES):
                frac = on_counts[j] / n
                print(f"  {k:>5}  {PLANE_LABELS[k]:<10}  {on_counts[j]:>12,}  {frac:>10.6f}")

            print()
            print("=== mask-bits-on per position (planes 20..29) ===")
            for nb in range(11):
                cnt = mask_bits_histogram[nb]
                if cnt > 0:
                    print(f"  {nb:>2} bits on: {cnt:>10,} positions ({100*cnt/n:6.3f}%)")

            print()
            print("=== overall outcome distribution (sampled) ===")
            total_o = max(overall_outcome.sum(), 1)
            print(f"  win  : {overall_outcome[0]:>10,}  ({100*overall_outcome[0]/total_o:6.3f}%)")
            print(f"  draw : {overall_outcome[1]:>10,}  ({100*overall_outcome[1]/total_o:6.3f}%)")
            print(f"  loss : {overall_outcome[2]:>10,}  ({100*overall_outcome[2]/total_o:6.3f}%)")
            print(f"  mean outcome (= P(win) - P(loss)): "
                  f"{(overall_outcome[0] - overall_outcome[2]) / total_o:+.4f}")

            print()
            print("=== conditional outcome distribution: P(outcome | plane on) vs P(outcome | plane off) ===")
            print(f"  {'plane':>5}  {'label':<10}  {'n_on':>9}  "
                  f"{'P(W|on)':>9} {'P(D|on)':>9} {'P(L|on)':>9} {'Δscalar_on':>11}  "
                  f"{'n_off':>10}  {'P(W|off)':>10} {'P(D|off)':>10} {'P(L|off)':>10} {'Δscalar_off':>13}")
            for j, k in enumerate(REP_PLANES):
                non = on_counts[j]
                noff = n - non
                w_on, d_on, l_on = outcome_on[j]
                w_off, d_off, l_off = outcome_off[j]
                if non > 0:
                    pW_on = w_on / non
                    pD_on = d_on / non
                    pL_on = l_on / non
                    sc_on = pW_on - pL_on
                else:
                    pW_on = pD_on = pL_on = sc_on = float("nan")
                if noff > 0:
                    pW_off = w_off / noff
                    pD_off = d_off / noff
                    pL_off = l_off / noff
                    sc_off = pW_off - pL_off
                else:
                    pW_off = pD_off = pL_off = sc_off = float("nan")
                print(f"  {k:>5}  {PLANE_LABELS[k]:<10}  {non:>9,}  "
                      f"{pW_on:>9.4f} {pD_on:>9.4f} {pL_on:>9.4f} {sc_on:>+11.4f}  "
                      f"{noff:>10,}  {pW_off:>10.4f} {pD_off:>10.4f} {pL_off:>10.4f} {sc_off:>+13.4f}")

            print()
            print("=== interpretation hint ===")
            print("  Compare Δscalar_on vs Δscalar_off and the overall scalar.")
            print("  If P(win) is much higher when a rep plane is ON than OFF,")
            print("  the network's 'set rep bit → predict win' behavior is mirroring")
            print("  a real training-data correlation. If they're similar, the network")
            print("  is generalizing wrongly from sparse data, not learning real signal.")

        finally:
            mm.close()


if __name__ == "__main__":
    main()
