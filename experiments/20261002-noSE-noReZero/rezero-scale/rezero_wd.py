"""Per-block branch scale (effective ReZero alpha x ||conv2||) across runs, from the checkpoints.

Usage: rezero_wd.py [--output PATH]

Each block's ReZero cap comes from its checkpoint's own metadata (`rezero_alpha_cap`,
or `rezero_alpha_init` x 1.0 for files older than format v6), so runs with different
ReZero settings, zero-init included, are read correctly. Writes a new CSV and refuses
to replace an existing one; the default output is rezero_wd.csv next to this script.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "20261001-se-fc1-leaky", "full-model-analysis", "scripts"))
import fma_lib as L  # noqa: E402

L.RUNS["se_none"] = ("ReLU no SE seed 1", "20260929-test_SE_none")
L.RUNS["norezero"] = ("no SE no ReZero", "20261002-bench_v5s3_noSE_noReZero")
RUN_ORDER = ["leaky", "se_none", "norezero"]


def block_rows(run, step, checkpoint):
    rows = []
    for block in checkpoint.rezero_blocks:
        p = f"blocks.{block.index}"
        has_alpha = f"{p}.rezero_alpha" in checkpoint
        if has_alpha != block.use_rezero:
            raise ValueError(f"{checkpoint.file}: block {block.index} use_rezero={block.use_rezero} "
                             f"but the file {'has' if has_alpha else 'lacks'} {p}.rezero_alpha")
        w1 = np.linalg.norm(checkpoint[f"{p}.conv1.weight"])
        w2 = np.linalg.norm(checkpoint[f"{p}.conv2.weight"])
        raw = checkpoint[f"{p}.rezero_alpha"][0] if has_alpha else None
        effective = block.effective(raw) if has_alpha else 1.0
        rows.append((run, step, block.index, raw, effective, w1, w2, effective * w2,
                     checkpoint[f"{p}.res_ln.weight"].mean()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=os.path.join(HERE, "rezero_wd.csv"))
    output = parser.parse_args().output
    if os.path.lexists(output):
        raise SystemExit(f"{output} already exists; pass --output with a new path (nothing is overwritten)")

    learning_rates = {}
    for line in open(os.path.expanduser("~/Library/Logs/DrewsChessMachine/dcm_log_20261001-151822.txt"), errors="replace"):
        m = re.search(r"\[REPLAY\] step=(\d+) .* lr=([0-9.e-]+)", line)
        if m and int(m.group(1)) % 1000 == 0:
            learning_rates[int(m.group(1))] = float(m.group(2))

    rows = []
    for run in RUN_ORDER:
        steps, _, _ = L.discover(run)
        for step, path in steps.items():
            rows.extend(block_rows(run, step, L.Checkpoint(path)))

    with open(output, "x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "block", "alpha_raw", "alpha_eff", "conv1_norm", "conv2_norm",
                         "eff_x_conv2", "ln_gamma_mean"])
        writer.writerows(rows)

    print("lr", {k: learning_rates[k] for k in sorted(learning_rates) if k % 2000 == 0})
    for run in RUN_ORDER:
        print(f"\n{run}: step | per block: raw a / eff / ||conv1|| / ||conv2|| / eff*||conv2|| / LN gamma")
        for step in sorted({r[1] for r in rows if r[0] == run}):
            if step % 2000 and step != 1000:
                continue
            cells = []
            for r in rows:
                if r[0] == run and r[1] == step:
                    raw = "-" if r[3] is None else f"{r[3]:.3f}"
                    cells.append(f"{raw}/{r[4]:.3f}/{r[5]:.1f}/{r[6]:.1f}/{r[7]:.2f}/{r[8]:.3f}")
            print(f"{step:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
