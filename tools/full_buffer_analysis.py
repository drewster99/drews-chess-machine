#!/usr/bin/env python3
"""
Full-buffer analysis of replay_buffer.bin.

Streams the entire stored buffer (no sampling) and reports:

A. Rep-plane analysis (the focused part)
   - Per-plane firing rates (planes 18..29)
   - Co-occurrence matrix between rep planes
   - Outcome conditioned on rep-pattern combinations

B. Other supplementary analyses
   1. Outcome by side-to-move (sanity check on encoding sign)
   2. Game length distribution
   3. Ply-index distribution
   4. Material count distribution
   5. Halfmove-clock distribution
   6. En passant firing rate
   7. Castling-rights distribution
   8. State-hash uniqueness (buffer diversity)
   9. Per-piece-type density (which piece types and how many)
   10. Outcome × ply correlation (decisiveness vs game stage)
   11. Outcome × material correlation (decisiveness vs material)

Stream-processes in batches to keep peak memory low.
"""

from __future__ import annotations

import argparse
import mmap
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Format constants (match ReplayBuffer.swift v7)
INPUT_PLANES = 30
BOARD_CELLS = 64
PER_BOARD_FLOATS = INPUT_PLANES * BOARD_CELLS
PER_BOARD_BYTES = PER_BOARD_FLOATS * 4
HEADER_SIZE = 8 + 4 + 4 + 8 * 5
TRAILER_SIZE = 32

PIECE_LABELS = [
    "my pawn", "my knight", "my bishop", "my rook", "my queen", "my king",
    "their pawn", "their knight", "their bishop", "their rook", "their queen", "their king",
]
PLANE_LABELS = PIECE_LABELS + [
    "my castle KS", "my castle QS", "their castle KS", "their castle QS",
    "en passant", "halfmove clock",
    "rep>=1", "rep>=2",
    "mask t-1", "mask t-2", "mask t-3", "mask t-4", "mask t-5",
    "mask t-6", "mask t-7", "mask t-8", "mask t-9", "mask t-10",
]
assert len(PLANE_LABELS) == 30

REP_PLANE_INDICES = list(range(18, 30))
REP_PLANE_NAMES = [PLANE_LABELS[k] for k in REP_PLANE_INDICES]


def find_default_buffer() -> Path:
    root = Path.home() / "Library" / "Application Support" / "DrewsChessMachine" / "Sessions"
    candidates: list[tuple[float, Path]] = []
    for d in root.glob("*.dcmsession"):
        b = d / "replay_buffer.bin"
        if b.exists():
            candidates.append((b.stat().st_mtime, b))
    if not candidates:
        raise SystemExit(f"No replay_buffer.bin under {root}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_header(buf: mmap.mmap) -> dict:
    if bytes(buf[0:8]) != b"DCMRPBUF":
        raise SystemExit("bad magic")
    version, _pad = struct.unpack_from("<II", buf, 8)
    fpb, cap, stored, wi, total = struct.unpack_from("<qqqqq", buf, 16)
    if fpb != PER_BOARD_FLOATS:
        raise SystemExit(f"floatsPerBoard mismatch: file={fpb}, expected={PER_BOARD_FLOATS}")
    return dict(version=version, capacity=cap, stored=stored, write_idx=wi, total=total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="replay_buffer.bin or .dcmsession dir")
    ap.add_argument("--batch", type=int, default=2048, help="batch size for streaming (default 2048)")
    ap.add_argument("--max-positions", type=int, default=0, help="cap positions analyzed (0 = all)")
    args = ap.parse_args()

    if args.path:
        p = Path(args.path)
        if p.is_dir():
            p = p / "replay_buffer.bin"
        buffer_path = p
    else:
        buffer_path = find_default_buffer()
    print(f"reading: {buffer_path}")
    print(f"  size: {buffer_path.stat().st_size / 2**30:.2f} GiB")

    with open(buffer_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            hdr = parse_header(mm)
            stored = hdr["stored"]
            print(f"  capacity={hdr['capacity']:,}  stored={stored:,}  totalAdded={hdr['total']:,}")
            if args.max_positions > 0:
                stored = min(stored, args.max_positions)
                print(f"  limiting analysis to first {stored:,} positions")

            # Section offsets
            boards_off = HEADER_SIZE
            moves_off = boards_off + hdr["stored"] * PER_BOARD_BYTES
            outcomes_off = moves_off + hdr["stored"] * 4
            ply_off = outcomes_off + hdr["stored"] * 4
            gamelen_off = ply_off + hdr["stored"] * 2
            tau_off = gamelen_off + hdr["stored"] * 2
            hash_off = tau_off + hdr["stored"] * 4
            wgid_off = hash_off + hdr["stored"] * 8
            mat_off = wgid_off + hdr["stored"] * 4

            # ----- Accumulators -----
            # Per-plane stats (broadcast and sparse handled separately)
            piece_total_per_plane = np.zeros(12, dtype=np.int64)        # sum of cells across all positions
            sparse_any_per_plane = np.zeros(12, dtype=np.int64)         # how many positions had any of this piece
            ep_any = 0
            halfmove_hist = np.zeros(100, dtype=np.int64)               # bucket of round(value * 99)
            # Halfmove × outcome buckets
            hm_buckets_edges = [0, 5, 10, 20, 30, 50, 75, 100]
            hm_outcome = np.zeros((len(hm_buckets_edges) - 1, 3), dtype=np.int64)
            castling_count = np.zeros(4, dtype=np.int64)                # planes 12..15 "still have right"
            rep_any_count = np.zeros(12, dtype=np.int64)                # planes 18..29 firing count

            # 12-bit rep-pattern bitmask histogram (planes 18..29 → bits 0..11)
            rep_combo_counter: Counter[int] = Counter()
            # Per-combo outcome tallies: combo_bits -> [W, D, L]
            combo_outcomes: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))

            # Co-occurrence matrix (12x12 of rep planes), counts of positions where both planes fire
            rep_co_occurrence = np.zeros((12, 12), dtype=np.int64)

            # Outcome counts overall + by STM
            overall_outcome = np.zeros(3, dtype=np.int64)   # [W, D, L]
            outcome_by_stm = np.zeros((2, 3), dtype=np.int64)   # 0=white-to-move, 1=black-to-move

            # Game length / ply / material histograms
            gamelen_buckets_edges = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 1 << 16]
            gamelen_buckets = np.zeros(len(gamelen_buckets_edges) - 1, dtype=np.int64)
            gamelen_total = 0
            gamelen_sum = 0
            gamelen_max = 0

            ply_buckets_edges = [0, 10, 20, 40, 60, 80, 100, 150, 200, 500, 1 << 16]
            ply_buckets = np.zeros(len(ply_buckets_edges) - 1, dtype=np.int64)

            material_buckets_edges = [0, 4, 8, 12, 14, 16, 100]
            material_buckets = np.zeros(len(material_buckets_edges) - 1, dtype=np.int64)

            # Outcome × ply (use ply buckets above)
            ply_outcome = np.zeros((len(ply_buckets_edges) - 1, 3), dtype=np.int64)
            # Outcome × material
            material_outcome = np.zeros((len(material_buckets_edges) - 1, 3), dtype=np.int64)

            # State-hash uniqueness — sample to keep memory bounded (5M uint64s = 40 MB)
            unique_hashes: set[int] = set()
            hash_sample_cap = 10_000_000

            # ----- Stream the file in batches -----
            print(f"\n  streaming {stored:,} positions in batches of {args.batch:,} …")
            done = 0
            while done < stored:
                n = min(args.batch, stored - done)

                # Boards (n × 1920 floats)
                boards_bytes = mm[boards_off + done * PER_BOARD_BYTES :
                                  boards_off + (done + n) * PER_BOARD_BYTES]
                boards = np.frombuffer(boards_bytes, dtype=np.float32).reshape(n, INPUT_PLANES, BOARD_CELLS)

                # Metadata
                outcomes = np.frombuffer(mm[outcomes_off + done * 4 : outcomes_off + (done + n) * 4],
                                         dtype=np.float32)
                plies = np.frombuffer(mm[ply_off + done * 2 : ply_off + (done + n) * 2],
                                      dtype=np.uint16)
                gamelens = np.frombuffer(mm[gamelen_off + done * 2 : gamelen_off + (done + n) * 2],
                                         dtype=np.uint16)
                hashes = np.frombuffer(mm[hash_off + done * 8 : hash_off + (done + n) * 8],
                                       dtype=np.uint64)
                materials = np.frombuffer(mm[mat_off + done : mat_off + done + n],
                                          dtype=np.uint8)

                # ----- Per-piece-type density (planes 0..11, sparse) -----
                for k in range(12):
                    plane = boards[:, k, :]
                    piece_total_per_plane[k] += int(plane.sum())
                    sparse_any_per_plane[k] += int((plane.max(axis=1) > 0.5).sum())

                # ----- Castling planes 12..15 (broadcast: read cell 0) -----
                for k in range(4):
                    castling_count[k] += int((boards[:, 12 + k, 0] > 0.5).sum())

                # ----- EP plane 16 (sparse) -----
                ep_any += int((boards[:, 16, :].max(axis=1) > 0.5).sum())

                # ----- Halfmove plane 17 (broadcast normalized 0..1) -----
                hm_vals = boards[:, 17, 0]
                hm_buckets = np.clip((hm_vals * 99 + 0.5).astype(np.int32), 0, 99)
                for bucket in hm_buckets:
                    halfmove_hist[bucket] += 1
                # Halfmove × outcome (vectorized)
                # outcome_idx is computed a few lines below — we compute it early here
                # to enable this cross-tab without restructuring the order.
                outcome_idx_for_hm = np.where(outcomes > 0.5, 0,
                                              np.where(outcomes < -0.5, 2, 1))
                for b in range(len(hm_buckets_edges) - 1):
                    lo = hm_buckets_edges[b]
                    hi = hm_buckets_edges[b + 1]
                    in_bucket = (hm_buckets >= lo) & (hm_buckets < hi)
                    for o_i in range(3):
                        hm_outcome[b, o_i] += int(((outcome_idx_for_hm == o_i) & in_bucket).sum())

                # ----- Rep planes 18..29 (broadcast: cell 0) -----
                rep_bits = (boards[:, 18:30, 0] > 0.5)   # (n, 12) bool
                # per-plane firings
                for j in range(12):
                    rep_any_count[j] += int(rep_bits[:, j].sum())
                # combo bitmask per position
                weights = (1 << np.arange(12, dtype=np.uint16))
                combos = (rep_bits.astype(np.uint16) * weights).sum(axis=1)
                for c, oc in zip(combos.tolist(), outcomes.tolist()):
                    rep_combo_counter[c] += 1
                    if oc > 0.5:
                        combo_outcomes[c][0] += 1
                    elif oc < -0.5:
                        combo_outcomes[c][2] += 1
                    else:
                        combo_outcomes[c][1] += 1
                # co-occurrence (vectorized via matmul on the bool matrix → int)
                rb_i = rep_bits.astype(np.int64)
                rep_co_occurrence += rb_i.T @ rb_i   # (12,12)

                # ----- Outcomes overall + by STM -----
                stm = (plies & 1).astype(np.int64)   # 0=white-to-move, 1=black-to-move
                outcome_idx = np.where(outcomes > 0.5, 0,
                                       np.where(outcomes < -0.5, 2, 1))
                for o_i in range(3):
                    overall_outcome[o_i] += int((outcome_idx == o_i).sum())
                for s_i in (0, 1):
                    mask = (stm == s_i)
                    for o_i in range(3):
                        outcome_by_stm[s_i, o_i] += int(((outcome_idx == o_i) & mask).sum())

                # ----- Game length stats -----
                gl_int = gamelens.astype(np.int64)
                gamelen_sum += int(gl_int.sum())
                gamelen_total += n
                gamelen_max = max(gamelen_max, int(gl_int.max()))
                # bucket
                for b in range(len(gamelen_buckets)):
                    lo = gamelen_buckets_edges[b]
                    hi = gamelen_buckets_edges[b + 1]
                    gamelen_buckets[b] += int(((gl_int >= lo) & (gl_int < hi)).sum())

                # ----- Ply distribution + outcome × ply -----
                plies_i = plies.astype(np.int64)
                for b in range(len(ply_buckets)):
                    lo = ply_buckets_edges[b]
                    hi = ply_buckets_edges[b + 1]
                    in_bucket = (plies_i >= lo) & (plies_i < hi)
                    cnt = int(in_bucket.sum())
                    ply_buckets[b] += cnt
                    for o_i in range(3):
                        ply_outcome[b, o_i] += int(((outcome_idx == o_i) & in_bucket).sum())

                # ----- Material distribution + outcome × material -----
                mat_i = materials.astype(np.int64)
                for b in range(len(material_buckets)):
                    lo = material_buckets_edges[b]
                    hi = material_buckets_edges[b + 1]
                    in_bucket = (mat_i >= lo) & (mat_i < hi)
                    cnt = int(in_bucket.sum())
                    material_buckets[b] += cnt
                    for o_i in range(3):
                        material_outcome[b, o_i] += int(((outcome_idx == o_i) & in_bucket).sum())

                # ----- State hash uniqueness (sampled to cap memory) -----
                if len(unique_hashes) < hash_sample_cap:
                    for h in hashes.tolist():
                        unique_hashes.add(int(h))
                        if len(unique_hashes) >= hash_sample_cap:
                            break

                done += n
                if done % (args.batch * 50) == 0 or done == stored:
                    print(f"    progress: {done:>10,} / {stored:,}  ({100*done/stored:5.2f}%)")

            # ----- Report -----
            total_o = max(int(overall_outcome.sum()), 1)
            print()
            print("=" * 90)
            print(f"=== A. REP-PLANE ANALYSIS (full buffer of {stored:,} positions) ===")
            print("=" * 90)
            print()
            print(f"  {'plane':>5}  {'label':<14}  {'firings':>14}  {'frac':>10}")
            for j, k in enumerate(REP_PLANE_INDICES):
                cnt = int(rep_any_count[j])
                print(f"  {k:>5}  {PLANE_LABELS[k]:<14}  {cnt:>14,}  {cnt/stored:>10.6f}")

            print()
            print("--- co-occurrence: rows = condition plane, cols = also-firing plane, value = P(col fires | row fires) ---")
            header = "  " + "        " + "  ".join(f"{PLANE_LABELS[k][:6]:>6}" for k in REP_PLANE_INDICES)
            print(header)
            for j, k in enumerate(REP_PLANE_INDICES):
                row_total = max(int(rep_co_occurrence[j, j]), 1)   # diagonal = total firings of this plane
                cells = []
                for j2 in range(12):
                    cells.append(f"{rep_co_occurrence[j, j2] / row_total:>6.3f}")
                print(f"  {PLANE_LABELS[k]:<8}  " + "  ".join(cells))

            print()
            print("--- top rep-pattern combos (by frequency), with outcome distribution ---")
            sorted_combos = sorted(rep_combo_counter.items(), key=lambda kv: -kv[1])
            print(f"  {'combo (binary, 12 bits)':<32}  {'planes set':<28}  {'count':>14}  {'%total':>8}  "
                  f"{'P(W)':>8} {'P(D)':>8} {'P(L)':>8} {'Δscalar':>9}")
            for combo, cnt in sorted_combos[:20]:
                bits_str = format(combo, "012b")[::-1]   # bit 0 = plane 18
                planes_set = ", ".join(
                    PLANE_LABELS[18 + i] for i in range(12) if (combo >> i) & 1
                ) or "(none)"
                wdl = combo_outcomes[combo]
                tot = max(int(wdl.sum()), 1)
                pW = wdl[0] / tot
                pD = wdl[1] / tot
                pL = wdl[2] / tot
                print(f"  {bits_str:<32}  {planes_set:<28}  {cnt:>14,}  {100*cnt/stored:>7.4f}%  "
                      f"{pW:>8.4f} {pD:>8.4f} {pL:>8.4f} {pW - pL:>+9.4f}")

            print()
            print("=" * 90)
            print("=== B. SUPPLEMENTARY ANALYSES ===")
            print("=" * 90)

            print()
            print("--- 1. Overall outcome distribution + by side-to-move ---")
            print(f"  overall: W={overall_outcome[0]:>10,} ({100*overall_outcome[0]/total_o:6.2f}%)  "
                  f"D={overall_outcome[1]:>10,} ({100*overall_outcome[1]/total_o:6.2f}%)  "
                  f"L={overall_outcome[2]:>10,} ({100*overall_outcome[2]/total_o:6.2f}%)  "
                  f"Δscalar={overall_outcome[0]/total_o - overall_outcome[2]/total_o:+.4f}")
            for s_i, label in enumerate(("white-to-move", "black-to-move")):
                row = outcome_by_stm[s_i]
                tot = max(int(row.sum()), 1)
                print(f"  {label}: W={row[0]:>10,} ({100*row[0]/tot:6.2f}%)  "
                      f"D={row[1]:>10,} ({100*row[1]/tot:6.2f}%)  "
                      f"L={row[2]:>10,} ({100*row[2]/tot:6.2f}%)  "
                      f"Δscalar={row[0]/tot - row[2]/tot:+.4f}")
            print("  (W = STM wins; sign-consistency check: white and black halves should look symmetric "
                  "since outcome is STM-relative)")

            print()
            print("--- 2. Game length distribution ---")
            avg = gamelen_sum / max(gamelen_total, 1)
            print(f"  mean game length: {avg:.2f} plies  (max observed: {gamelen_max})")
            for b in range(len(gamelen_buckets)):
                lo = gamelen_buckets_edges[b]
                hi = gamelen_buckets_edges[b + 1]
                cnt = int(gamelen_buckets[b])
                print(f"  [{lo:>4}, {hi:>5}) plies: {cnt:>12,} ({100*cnt/stored:6.3f}%)")

            print()
            print("--- 3. Ply-index distribution (position-level, not game-level) ---")
            for b in range(len(ply_buckets)):
                lo = ply_buckets_edges[b]
                hi = ply_buckets_edges[b + 1]
                cnt = int(ply_buckets[b])
                print(f"  ply [{lo:>4}, {hi:>5}): {cnt:>12,} ({100*cnt/stored:6.3f}%)")

            print()
            print("--- 4. Material count distribution (non-pawn pieces) ---")
            for b in range(len(material_buckets)):
                lo = material_buckets_edges[b]
                hi = material_buckets_edges[b + 1]
                cnt = int(material_buckets[b])
                print(f"  material [{lo:>3}, {hi:>4}): {cnt:>12,} ({100*cnt/stored:6.3f}%)")

            print()
            print("--- 5. Halfmove-clock distribution (plane 17 × 99) ---")
            # Print non-empty buckets
            for v in range(100):
                cnt = int(halfmove_hist[v])
                if cnt > 0:
                    if v % 10 == 0 or v == 99 or cnt > stored * 0.001:
                        print(f"  halfmove={v:>3}: {cnt:>12,} ({100*cnt/stored:6.3f}%)")
            high_50_plus = int(halfmove_hist[50:].sum())
            high_80_plus = int(halfmove_hist[80:].sum())
            print(f"  ≥50: {high_50_plus:>12,} ({100*high_50_plus/stored:6.3f}%)")
            print(f"  ≥80: {high_80_plus:>12,} ({100*high_80_plus/stored:6.3f}%)")

            print()
            print("--- 5b. Halfmove-clock × outcome (decisiveness vs proximity to 50-move-rule) ---")
            for b in range(len(hm_buckets_edges) - 1):
                lo = hm_buckets_edges[b]
                hi = hm_buckets_edges[b + 1]
                row = hm_outcome[b]
                tot = max(int(row.sum()), 1)
                print(f"  halfmove [{lo:>3}, {hi:>3}) (n={tot:>10,}, {100*tot/stored:>5.2f}% of buffer): "
                      f"W={row[0]/tot:.3f}  D={row[1]/tot:.3f}  L={row[2]/tot:.3f}  "
                      f"decisiveness={1 - row[1]/tot:.3f}")

            print()
            print("--- 6. En passant firing rate ---")
            print(f"  positions with EP target set: {ep_any:>12,} ({100*ep_any/stored:6.4f}%)")

            print()
            print("--- 7. Castling-rights distribution (broadcast plane 12..15) ---")
            for k in range(4):
                cnt = int(castling_count[k])
                print(f"  plane {12+k} ({PLANE_LABELS[12+k]}): {cnt:>12,} ({100*cnt/stored:6.3f}%)")

            print()
            print("--- 8. State-hash uniqueness (sampled up to {} hashes) ---"
                  .format(hash_sample_cap))
            uniq = len(unique_hashes)
            scanned = min(stored, hash_sample_cap)   # rough — actual samples can be less if cap reached
            print(f"  unique state hashes seen: {uniq:,} (out of up to {scanned:,} positions scanned)")
            if scanned > 0:
                print(f"  uniqueness ratio: {uniq/scanned:.4f}  (1.00 = all positions distinct)")

            print()
            print("--- 9. Per-piece-type density (avg pieces of each type per position) ---")
            for k in range(12):
                avg_per_pos = piece_total_per_plane[k] / stored
                any_frac = sparse_any_per_plane[k] / stored
                print(f"  plane {k:>2} {PLANE_LABELS[k]:<14}: avg={avg_per_pos:6.3f}  "
                      f"P(any present)={any_frac:6.3f}")

            print()
            print("--- 10. Outcome × ply (decisiveness vs game stage) ---")
            for b in range(len(ply_buckets)):
                lo = ply_buckets_edges[b]
                hi = ply_buckets_edges[b + 1]
                row = ply_outcome[b]
                tot = max(int(row.sum()), 1)
                print(f"  ply [{lo:>4}, {hi:>5}) (n={tot:>10,}): "
                      f"W={row[0]/tot:.3f}  D={row[1]/tot:.3f}  L={row[2]/tot:.3f}  "
                      f"decisiveness={1 - row[1]/tot:.3f}")

            print()
            print("--- 11. Outcome × material (decisiveness vs material) ---")
            for b in range(len(material_buckets)):
                lo = material_buckets_edges[b]
                hi = material_buckets_edges[b + 1]
                row = material_outcome[b]
                tot = max(int(row.sum()), 1)
                print(f"  material [{lo:>3}, {hi:>4}) (n={tot:>10,}): "
                      f"W={row[0]/tot:.3f}  D={row[1]/tot:.3f}  L={row[2]/tot:.3f}  "
                      f"decisiveness={1 - row[1]/tot:.3f}")

        finally:
            mm.close()


if __name__ == "__main__":
    main()
