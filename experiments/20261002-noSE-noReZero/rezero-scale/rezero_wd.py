import sys, re, os, numpy as np
sys.path.insert(0, "/Users/andrew/cursor/drews-chess-machine/experiments/20261001-se-fc1-leaky/full-model-analysis/scripts")
import fma_lib as L
L.RUNS["se_none"] = ("ReLU no SE seed 1", "20260929-test_SE_none")
L.RUNS["norezero"] = ("no SE no ReZero", "20261002-bench_v5s3_noSE_noReZero")
lr = {}
for line in open(os.path.expanduser("~/Library/Logs/DrewsChessMachine/dcm_log_20261001-151822.txt"), errors="replace"):
    m = re.search(r"\[REPLAY\] step=(\d+) .* lr=([0-9.e-]+)", line)
    if m and int(m.group(1)) % 1000 == 0: lr[int(m.group(1))] = float(m.group(2))
rows = []
for run in ["leaky", "se_none", "norezero"]:
    steps, _, _ = L.discover(run)
    for step, path in steps.items():
        c = L.Checkpoint(path)
        for b in range(3):
            p = f"blocks.{b}"
            w1 = np.linalg.norm(c[f"{p}.conv1.weight"]); w2 = np.linalg.norm(c[f"{p}.conv2.weight"])
            a = c[f"{p}.rezero_alpha"][0] if f"{p}.rezero_alpha" in c else None
            eff = L.rezero_effective(a, 0.4472136) if a is not None else 1.0
            rows.append((run, step, b, a, eff, w1, w2, eff * w2, c[f"{p}.res_ln.weight"].mean()))
import csv
with open(os.path.join(os.path.dirname(__file__), "rezero_wd.csv"), "w") as f:
    w = csv.writer(f); w.writerow(["run","step","block","alpha_raw","alpha_eff","conv1_norm","conv2_norm","eff_x_conv2","ln_gamma_mean"]); w.writerows(rows)
print("lr", {k: lr[k] for k in sorted(lr) if k % 2000 == 0})
for run in ["leaky", "se_none", "norezero"]:
    print(f"\n{run}: step | per block: raw a / eff / ||conv1|| / ||conv2|| / eff*||conv2|| / LN gamma")
    for step in sorted({r[1] for r in rows if r[0] == run}):
        if step % 2000 and step not in (1000,): continue
        cells = []
        for r in rows:
            if r[0] == run and r[1] == step:
                a = "-" if r[3] is None else f"{r[3]:.3f}"
                cells.append(f"{a}/{r[4]:.3f}/{r[5]:.1f}/{r[6]:.1f}/{r[7]:.2f}/{r[8]:.3f}")
        print(f"{step:>6} | " + " | ".join(cells))
