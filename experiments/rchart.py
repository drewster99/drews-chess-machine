"""The R1–R16 summary chart: pElo at 33k, 32k, 31k, 30k, 21k and 7k for every compared run,
read from the experiments' own table scripts (one source for every number)."""
import subprocess, sys

REPO = "/Users/andrew/cursor/drews-chess-machine"
TABLES = ["experiments/20261001-se-fc1-leaky/table.py",
          "experiments/20261002-label-smoothing-C/table.py",
          "experiments/20261002-noSE-noReZero/table.py"]
STEPS = [33000, 32000, 31000, 30000, 21000, 7000]
RUNNING = {"pElo fatty", "pElo fatconv"}
# (label, column header in the table scripts, note)
RUNS = [
    ("R1", "SE scale+bias, seed 1 (label-smoothing baseline, ε 0.1/0.013)", "pElo baseline ε 0.1 / 0.013", ""),
    ("R2", "SE scale+bias, seed 2 (baseline seed 2)", "pElo baseline seed 2", ""),
    ("R3", "SE attenuate-only", "pElo ReLU attenuate-only", ""),
    ("R4", "SE scale+bias, leaky ReLU in SE FC1", "pElo leaky FC1 scale+bias", ""),
    ("R5", "no SE + ReZero, seed 1", "pElo no SE + ReZero s1", ""),
    ("R6", "no SE + ReZero, seed 2", "pElo no SE + ReZero s2", ""),
    ("R7", "no SE, no ReZero, seed 1", "pElo no SE, no ReZero s1", ""),
    ("R8", "no SE, no ReZero, seed 2", "pElo no SE, no ReZero s2", ""),
    ("R9", "zero-init ReZero (no SE)", "pElo zero-init ReZero", "same start weights as R5 (only ReZero α init/cap changed)"),
    ("R10", "C: SE scale+bias, policy ε 0.03, seed 1", "pElo C policy ε 0.03", "same start weights as R1"),
    ("R11", "C: SE scale+bias, policy ε 0.03, seed 2", "pElo C seed 2", "same start weights as R2; stopped by the owner at 31,906"),
    ("R12", "D: SE scale+bias, value ε 0", "pElo D value ε 0", "same start weights as R1"),
    ("R13", "fatty: no SE, no ReZero, 1 block × 7×7 @216", "pElo fatty", "same budget as R7/R8"),
    ("R14", "skinny: no SE, no ReZero, 22 blocks × 7×7 @48", "pElo skinny", "same budget as R7/R8; stopped by the owner at 1,605"),
    ("R15", "slim-neck fatty: fatty with a 3×3 stem, 1 block × 7×7 @224", "pElo slim-neck fatty", "same budget as R7/R8"),
    ("R16", "fatconv: 15×15 stem, 1 block × 15×15 @98", "pElo fatconv", "same budget as R7/R8"),
]

def num(v):
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None

cols = {}
for script in TABLES:
    out = subprocess.run(["python3", script], capture_output=True, text=True, cwd=REPO, check=True).stdout.splitlines()
    hdr = [h.strip() for h in out[0].strip("|").split("|")]
    for line in out[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        step = int(cells[0].replace(",", ""))
        for h, v in zip(hdr, cells):
            if h.startswith("pElo") and num(v) is not None:
                cols.setdefault(h, {})[step] = num(v)

import json, os
for col, rel in (("pElo fatty", "experiments/20261003-fatty-1x7x7-216/probes.jsonl"),
                 ("pElo skinny", "experiments/20261003-skinny-22x7x7-48/probes.jsonl"),
                 ("pElo slim-neck fatty", "experiments/20261003-fatty224-3x3stem/probes.jsonl"),
                 ("pElo fatconv", "experiments/20261004-fatconv-1x15x15-98/probes.jsonl")):
    cols[col] = {}
    path = os.path.join(REPO, rel)
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("pElo") is not None:
                    cols[col][int(rec["step"])] = float(rec["pElo"])
missing = [h for _, _, h, _ in RUNS if h not in cols]
if missing:
    sys.exit(f"columns not found in the table scripts: {missing}")

best = {s: max((cols[h][s] for _, _, h, _ in RUNS if s in cols[h]), default=None) for s in STEPS}
print("| Max step | Run | " + " | ".join(f"{s // 1000}k" for s in STEPS) + " |")
print("|---:|---|" + "---:|" * len(STEPS))
for label, name, h, note in RUNS:
    top = max(cols[h]) if cols[h] else 0
    if top:
        maxcell = f"{top:,}" + (" (running)" if h in RUNNING else "")
    else:
        maxcell = "running, no probe yet" if h in RUNNING else "queued"
    vals = []
    for s in STEPS:
        v = cols[h].get(s)
        vals.append("" if v is None else (f"**{v:.1f}**" if v == best[s] else f"{v:.1f}"))
    runcell = f"**{label}** {name}" + (f" — *{note}*" if note else "")
    print(f"| {maxcell} | {runcell} | " + " | ".join(vals) + " |")
