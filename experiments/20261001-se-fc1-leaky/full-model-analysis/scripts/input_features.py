#!/usr/bin/env python3
"""Element-level dead inputs that per-unit norms hide.

Two places in the network are not weight-shared across squares, or are only
partly reachable, so individual *input elements* can be dead while the unit
that reads them is alive:

1. value.fc1 reads the flattened value-conv features: input index
   `k * 64 + square` (value-conv channel k after BN + ReLU, at one square;
   flatten of NCHW). If ReLU(value.bn channel k) is 0 at that square for every
   training position, that input column of fc1 gets exactly zero gradient for
   all 128 hidden units -- a dead (channel, square) feature. Velocity layout
   [in, out] = [1024, 128] (verified in layout_check.md).
2. stem.conv reads the 30 input planes through a 7x7 kernel. A (plane, kernel
   offset) pair whose input square is never set in training gets zero gradient
   for every output channel (e.g. en passant only ever sits on one rank).

Only runs that save velocity (leaky-FC1, ReLU seed 2). Writes:
- results/value_fc1_dead_inputs.csv  -- one row per (run, step, channel, square)
  that is dead (all 128 velocity entries exactly 0), plus per-channel counts.
- results/stem_dead_kernel_offsets.csv -- one row per (run, step, plane, dy, dx)
  whose velocity is 0 for all 128 output channels.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fma_lib as L  # noqa: E402


def square_name(square):
    """Encoder frame: row 0 = the side to move's far rank (rank 8 for white), col 0 = file a."""
    row, col = divmod(square, 8)
    return f"{'abcdefgh'[col]}{8 - row}"


def main():
    value_rows, stem_rows, channel_counts = [], [], []
    for run in ("leaky", "relu_s2"):
        files, _, _ = L.discover(run)
        for step, path in files.items():
            if step == 0:
                continue
            ckpt = L.Checkpoint(path)
            # The value-conv features below are read as ReLU(value.bn).
            L.dcm_arch.require_relu(ckpt.metadata, ckpt.file, ("value_head_conv_activation",))
            if not ckpt.has_velocity:
                raise ValueError(f"{ckpt.file}: expected optimizer velocity")
            v = ckpt.velocity("value.fc1.weight")  # [128 out, 1024 in]
            dead_inputs = np.all(v == 0.0, axis=0)
            channels = ckpt["value.conv.weight"].shape[0]
            per_channel = dead_inputs.reshape(channels, 64)
            for k in range(channels):
                channel_counts.append({"run": run, "step": step, "model_id": ckpt.model_id, "channel": k,
                                       "dead_squares": int(per_channel[k].sum()),
                                       "squares": " ".join(square_name(s) for s in np.flatnonzero(per_channel[k]))})
                for s in np.flatnonzero(per_channel[k]):
                    value_rows.append({"run": run, "step": step, "model_id": ckpt.model_id, "channel": k,
                                       "square_index": int(s), "square": square_name(int(s))})
            sv = ckpt.velocity("stem.conv.weight")  # [128, 30, 7, 7]
            dead = np.all(sv == 0.0, axis=0)  # [30, 7, 7]
            for plane, dy, dx in zip(*np.nonzero(dead)):
                stem_rows.append({"run": run, "step": step, "model_id": ckpt.model_id, "plane": int(plane),
                                  "plane_label": L.PLANE_LABELS[plane], "kernel_dy": int(dy) - 3, "kernel_dx": int(dx) - 3})
            print(f"{run} {step}: value.fc1 dead inputs {int(dead_inputs.sum())}; stem dead (plane, offset) {int(dead.sum())}",
                  file=sys.stderr)
    with open(os.path.join(L.RESULTS_DIR, "value_fc1_dead_inputs.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "model_id", "channel", "square_index", "square"])
        writer.writeheader()
        writer.writerows(value_rows)
    with open(os.path.join(L.RESULTS_DIR, "value_fc1_dead_inputs_by_channel.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "model_id", "channel", "dead_squares", "squares"])
        writer.writeheader()
        writer.writerows(channel_counts)
    with open(os.path.join(L.RESULTS_DIR, "stem_dead_kernel_offsets.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "model_id", "plane", "plane_label", "kernel_dy", "kernel_dx"])
        writer.writeheader()
        writer.writerows(stem_rows)


if __name__ == "__main__":
    main()
