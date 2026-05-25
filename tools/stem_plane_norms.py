#!/usr/bin/env python3
"""
Per-input-plane L2 norm of the stem conv weight.

Reads a .dcmmodel (or a .dcmsession dir; picks trainer.dcmmodel) and reports,
for each of the 30 input planes, how strongly the network is reading from
that plane in its very first layer (stem 3x3 conv, weight shape
[128, 30, 3, 3] OIHW).

A plane the network has learned to ignore will have a per-plane L2 norm
much smaller than the actively-used planes. The script is intended as a
quick diagnostic for the question: "did adding planes X..Y actually do
anything?"

Usage:
    tools/stem_plane_norms.py                                # auto-pick most recent
    tools/stem_plane_norms.py /path/to/file.dcmmodel
    tools/stem_plane_norms.py /path/to/session.dcmsession
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_dcmmodel import parse_dcmmodel, _resolve_to_model  # noqa: E402

# Match BoardEncoder.swift plane semantics. Always from STM perspective.
PLANE_LABELS = [
    "my pawn",          # 0
    "my knight",        # 1
    "my bishop",        # 2
    "my rook",          # 3
    "my queen",         # 4
    "my king",          # 5
    "their pawn",       # 6
    "their knight",     # 7
    "their bishop",     # 8
    "their rook",       # 9
    "their queen",      # 10
    "their king",       # 11
    "my castling KS",   # 12
    "my castling QS",   # 13
    "their castling KS",# 14
    "their castling QS",# 15
    "en passant",       # 16
    "halfmove clock",   # 17
    "rep >=1",          # 18
    "rep >=2",          # 19
    "rep mask t-1",     # 20
    "rep mask t-2",     # 21
    "rep mask t-3",     # 22
    "rep mask t-4",     # 23
    "rep mask t-5",     # 24
    "rep mask t-6",     # 25
    "rep mask t-7",     # 26
    "rep mask t-8",     # 27
    "rep mask t-9",     # 28
    "rep mask t-10",    # 29
]

CHANNELS_OUT = 128
INPUT_PLANES = 30
KH = 3
KW = 3
STEM_ELEM_COUNT = CHANNELS_OUT * INPUT_PLANES * KH * KW


def find_default_model() -> Path:
    root = Path.home() / "Library" / "Application Support" / "DrewsChessMachine"
    models_dir = root / "Models"
    sessions_dir = root / "Sessions"
    candidates: list[tuple[float, Path]] = []
    if models_dir.exists():
        for p in models_dir.glob("*.dcmmodel"):
            candidates.append((p.stat().st_mtime, p))
    if sessions_dir.exists():
        for d in sessions_dir.glob("*.dcmsession"):
            trainer = d / "trainer.dcmmodel"
            if trainer.exists():
                candidates.append((trainer.stat().st_mtime, trainer))
    if not candidates:
        raise SystemExit(f"No .dcmmodel found under {root}")
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help=".dcmmodel file or .dcmsession dir (auto-pick if omitted)")
    args = ap.parse_args()

    if args.path:
        model_path = _resolve_to_model(Path(args.path))
    else:
        model_path = find_default_model()
    print(f"reading: {model_path}")

    parsed = parse_dcmmodel(model_path)
    print(f"  modelID={parsed['model_id']}  numTensors={parsed['num_tensors']}")
    meta = parsed["metadata"]
    if "trainingStep" in meta:
        print(f"  trainingStep={meta['trainingStep']}")
    print()

    # Stem conv weight is tensor index 0, shape [128, 30, 3, 3] OIHW.
    if not parsed["tensors"]:
        raise SystemExit("no tensors found")
    idx0, t0 = parsed["tensors"][0]
    if idx0 != 0:
        raise SystemExit(f"first tensor has index {idx0}, expected 0")
    if len(t0) != STEM_ELEM_COUNT:
        raise SystemExit(
            f"stem tensor element count {len(t0)} != expected {STEM_ELEM_COUNT} "
            f"(network input planes may have changed; update INPUT_PLANES)"
        )

    W = t0.reshape(CHANNELS_OUT, INPUT_PLANES, KH, KW).astype(np.float64)

    # Per-input-plane L2 norm: sqrt(sum over [O, KH, KW] of W^2)
    plane_l2 = np.sqrt(np.sum(W ** 2, axis=(0, 2, 3)))
    plane_mean_abs = np.mean(np.abs(W), axis=(0, 2, 3))
    plane_max_abs = np.max(np.abs(W), axis=(0, 2, 3))

    # Consistency check: per-plane sum should equal total squared norm.
    total_l2_sq_from_planes = float(np.sum(plane_l2 ** 2))
    total_l2_sq = float(np.sum(W ** 2))
    diff = abs(total_l2_sq_from_planes - total_l2_sq)
    rel = diff / total_l2_sq if total_l2_sq > 0 else 0.0
    print(f"  consistency: sum(plane_L2^2)={total_l2_sq_from_planes:.6e} "
          f"total_L2^2={total_l2_sq:.6e} relDiff={rel:.2e}")
    if rel > 1e-10:
        print("  WARNING: per-plane decomposition does not match total. Math bug.", file=sys.stderr)

    # Headline summary
    board_state_mean = float(plane_l2[0:18].mean())
    rep_count_mean = float(plane_l2[18:20].mean())
    rep_mask_mean = float(plane_l2[20:30].mean())
    print()
    print("=== headline group means ===")
    print(f"  board-state planes  [ 0..17]:  mean L2 = {board_state_mean:.4f}")
    print(f"  repetition counts   [18..19]:  mean L2 = {rep_count_mean:.4f}  "
          f"(ratio vs board: {rep_count_mean / board_state_mean:.3f})")
    print(f"  temporal rep mask   [20..29]:  mean L2 = {rep_mask_mean:.4f}  "
          f"(ratio vs board: {rep_mask_mean / board_state_mean:.3f})")

    # Per-plane table
    print()
    print("=== per-input-plane stem weight stats ===")
    print(f"  {'idx':>3}  {'label':<20}  {'L2':>10}  {'meanAbs':>10}  {'maxAbs':>10}  {'L2 / boardMean':>16}")
    for k in range(INPUT_PLANES):
        ratio = plane_l2[k] / board_state_mean if board_state_mean > 0 else 0.0
        if k == 18:
            print("  " + "-" * 80)
        if k == 20:
            print("  " + "-" * 80)
        print(f"  {k:>3}  {PLANE_LABELS[k]:<20}  {plane_l2[k]:>10.4f}  "
              f"{plane_mean_abs[k]:>10.4f}  {plane_max_abs[k]:>10.4f}  {ratio:>16.3f}")

    # Per-(output-channel, input-plane) kernel-norm distribution.
    # For each input plane k, compute the 128 per-output-channel kernel L2
    # norms ||W[o, k, :, :]||_2. If the plane is unused, the distribution
    # will sit at the He-init / weight-decay equilibrium (tight cluster
    # near the floor). If specialized, expect a long tail — some output
    # channels strongly use this plane, most don't.
    kernel_l2 = np.sqrt(np.sum(W ** 2, axis=(2, 3)))  # shape [128, 30]
    print()
    print("=== per-input-plane kernel-norm distribution (over 128 output channels) ===")
    # He-init per-kernel-slice norm (9 weights, std = sqrt(2 / fan_in)):
    fan_in = INPUT_PLANES * KH * KW
    init_kernel_l2 = np.sqrt(2.0 / fan_in) * np.sqrt(KH * KW)
    # Weight-decay-only equilibrium after ~495k steps with lr=1e-3, wd=1e-4:
    wd_factor = np.exp(-494927 * 1e-3 * 1e-4)
    decay_floor = init_kernel_l2 * wd_factor
    print(f"  reference: He-init per-kernel-slice L2 = {init_kernel_l2:.4f}, "
          f"decay-only floor (lr=1e-3, wd=1e-4, 495k steps) = {decay_floor:.4f}")
    print()
    spec_thresh = 2.0 * decay_floor   # specialist = kernel-norm > 2× decay floor
    print(f"  {'idx':>3}  {'label':<20}  {'mean':>8}  {'std':>8}  {'p50':>8}  "
          f"{'p90':>8}  {'p99':>8}  {'max':>8}  {'#>2×floor':>10}  {'top10mean':>10}")
    for k in range(INPUT_PLANES):
        col = kernel_l2[:, k]
        sorted_col = np.sort(col)[::-1]
        p50 = float(np.percentile(col, 50))
        p90 = float(np.percentile(col, 90))
        p99 = float(np.percentile(col, 99))
        n_spec = int(np.sum(col > spec_thresh))
        top10mean = float(sorted_col[:10].mean())
        if k == 18:
            print("  " + "-" * 110)
        if k == 20:
            print("  " + "-" * 110)
        print(f"  {k:>3}  {PLANE_LABELS[k]:<20}  {col.mean():>8.4f}  {col.std():>8.4f}  "
              f"{p50:>8.4f}  {p90:>8.4f}  {p99:>8.4f}  {col.max():>8.4f}  "
              f"{n_spec:>10d}  {top10mean:>10.4f}")

    # Group summary: how many specialist channels each group has
    print()
    print("=== specialization summary ===")
    for label, lo, hi in [
        ("board state (0-17)", 0, 18),
        ("rep counts (18-19)", 18, 20),
        ("rep mask (20-29)", 20, 30),
    ]:
        block = kernel_l2[:, lo:hi]
        total_kernels = block.size
        n_spec = int(np.sum(block > spec_thresh))
        print(f"  {label:<22}: {n_spec:>5d} / {total_kernels:>5d} kernels "
              f"({100*n_spec/total_kernels:5.1f}%) exceed 2× decay floor; "
              f"max kernel norm = {block.max():.4f}")

    # Interpretation rubric
    print()
    print("=== interpretation ===")
    ratio_rep_mask = rep_mask_mean / board_state_mean
    if ratio_rep_mask > 0.5:
        verdict = ("rep-mask planes appear to be carrying real signal "
                   "(>0.5× board-state plane norm). Network is using them.")
    elif ratio_rep_mask > 0.1:
        verdict = ("rep-mask planes are AMBIGUOUS (0.1×–0.5× board norm). "
                   "Inconclusive — run a probe-positions test before deciding.")
    else:
        verdict = ("rep-mask planes appear effectively UNUSED (<0.1× board norm). "
                   "Strong evidence the network learned to ignore them.")
    print(f"  rep-mask vs board ratio = {ratio_rep_mask:.3f} → {verdict}")

    # Per-plane outlier callouts within 20..29
    mask_planes = plane_l2[20:30]
    mask_mean = mask_planes.mean()
    mask_std = mask_planes.std()
    if mask_std > 0:
        for k in range(20, 30):
            z = (plane_l2[k] - mask_mean) / mask_std
            if abs(z) >= 1.5:
                tag = "HIGH" if z > 0 else "LOW"
                print(f"  outlier within rep-mask: plane {k} ({PLANE_LABELS[k]})  "
                      f"L2={plane_l2[k]:.4f}  z={z:+.2f}σ  ({tag})")


if __name__ == "__main__":
    main()
