#!/usr/bin/env python3
"""Per-1000-step table: R-elitecyc-unlik vs R-elitecyc (bf16), lichess-wide first, then lichess-large.

unlik lichess-wide comes from each checkpoint's embedded dcm_test_set_results; bf16 lichess-wide from the
training-tables page (its 1000-step files were thinned to 10k); lichess-large from the probe jsonl files.
Usage: unlik_table.py [from_step] [to_step]
"""
import glob, json, os, re, sys
sys.path.insert(0, "/Users/andrew/cursor/drews-chess-machine/scripts"); import dcm_arch
X = "/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr"
M = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models/")
LO = int(sys.argv[1]) if len(sys.argv) > 1 else 385000
HI = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9


FORK = "/Users/andrew/cursor/drews-chess-machine/experiments/20261123-unlikelihood-fork"
# Session logs of the fork's segments, in order (stdout is block-buffered).
FORK_LOGS = ["dcm_log_20261122-235822.txt", "dcm_log_20261124-181200.txt", "dcm_log_20261126-155001.txt"]
BIN = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2701-e5e79ab2.app/Contents/MacOS/DrewsChessMachine")
LARGE = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/OpsScripts/lichess_probes_large.json")

def log_lines(logs):
    out = {}
    for log in logs:
        for l in open(os.path.expanduser(f"~/Library/Logs/DrewsChessMachine/{log}"), errors="replace"):
            if "[REPLAY] step=" in l and (m := re.search(r"trainerStep=(\d+)", l)) and int(m.group(1)) % 1000 == 0:
                out[int(m.group(1))] = dict(re.findall(r"(\w+)=([-\d.e+]+)", l))
    return out

def large_probe(s):
    """lichess-large probe of the fork's step-s checkpoint, run once and kept in large_probe.jsonl."""
    import subprocess
    path = f"{FORK}/large_probe.jsonl"
    rows = {}
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l); rows[int(re.search(r"step(\d+)\.safetensors", d["model"]).group(1))] = d
    f = f"{M}20261123-zlra-elitecyc-unlik-latest-step{s}.safetensors"
    if s not in rows and os.path.exists(f):
        tmp = f"{FORK}/.probe_{s}.jsonl"
        subprocess.run([BIN, "--probe-model", f, "--probe-set-file", LARGE, "--probe-out", tmp], check=True, capture_output=True)
        line = open(tmp).readline(); os.remove(tmp)
        with open(path, "a") as out: out.write(line if line.endswith("\n") else line + "\n")
        rows[s] = json.loads(line)
    return rows.get(s)

def step_lines(pat):
    out = {}
    for p in sorted(glob.glob(f"{X}/{pat}"), key=lambda p: int(re.search(r"seg(\d+)", p).group(1))):
        for l in open(p, errors="replace"):
            if l.startswith("[REPLAY] step=") and (m := re.search(r"trainerStep=(\d+)", l)) and int(m.group(1)) % 1000 == 0:
                out[int(m.group(1))] = dict(re.findall(r"(\w+)=([-\d.e+]+)", l))
    return out

def bf16_wide():
    # R-elitecyc's own key in the step tables' record (wide pElo/NLL per checkpoint, kept after thinning).
    # Never parse the rendered page: its sections change as runs are added.
    rec = json.load(open(f"{X}/tables/wide_pelo_record.json"))["elitecyc"]
    return {int(k): (v[0], v[1]) for k, v in rec.items()}

def fp32_wide(s):
    fn = f"{M}20261123-zlra-elitecyc-unlik-latest-step{s}.safetensors"
    if not os.path.exists(fn): return None
    r = json.loads(dcm_arch.read_header(fn)[0]["__metadata__"]["dcm_test_set_results"])
    w = [t for t in r["sets"] if t["id"] == "lichess-wide"][0]
    return (w["pelo"], w["nll"])

def large(path):
    try: return json.loads(open(path).readline())
    except OSError: return None

F, B, BW = log_lines(FORK_LOGS), step_lines("train-elitecyc-seg*.stdout"), bf16_wide()
BL = {int(re.search(r"step(\d+)", (d := json.loads(l))["model"]).group(1)): d
      for l in open("/Users/andrew/cursor/drews-chess-machine/experiments/20261108-elitecyc-heldout/large_probe.jsonl")}
v = lambda x, f: format(x, f) if x is not None else ""
cols = ["step", "wide pElo unlik", "wide pElo cyc", "Δ wide", "wide NLL unlik", "wide NLL cyc",
        "large pElo unlik", "large pElo cyc", "Δ large", "large NLL unlik", "large top-1", "large top-5",
        "loss", "Δ loss", "pLoss", "vLoss", "pEnt", "pD", "gNorm", "LR", "ms/step unlik", "ms/step cyc", "buffer avg plies/game"]
print("| " + " | ".join(cols) + " |"); print("|" + "---:|" * len(cols))
for s in sorted(k for k in F if LO <= k <= HI):
    f, b, fw, bw = F[s], B.get(s), fp32_wide(s), BW.get(s)
    p, q = large_probe(s), BL.get(s)
    row = [f"{s:,}", v(fw and fw[0], ".1f"), v(bw and bw[0], ".1f"), v(fw[0] - bw[0] if fw and bw else None, "+.1f"),
           v(fw and fw[1], ".3f"), v(bw and bw[1], ".3f"),
           v(p and p["pElo"], ".1f"), v(q and q["pElo"], ".1f"), v(p["pElo"] - q["pElo"] if p and q else None, "+.1f"),
           v(p and p["nll"], ".4f"), v(p and p["argmaxCorrect"], ","), v(p and p["top5Correct"], ","),
           f["loss"], v(float(f["loss"]) - float(b["loss"]) if b else None, "+.4f"), f["pLoss"], f["vLoss"], f["pEnt"], f["pD"],
           f["gNorm"], f["lr"], f"{float(f['ms']):.0f}", v(b and float(b["ms"]), ".0f"), f"{int(f['plies']) / int(f['games']):.1f}"]
    print("| " + " | ".join(row) + " |")
