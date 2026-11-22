#!/usr/bin/env python3
"""Per-1000-step table for the three label-smoothing arms: lichess-wide (embedded in each checkpoint),
lichess-large (probed here once per checkpoint into large_probe_<arm>.jsonl) and the step-line metrics.
Usage: report.py [from_step] [to_step] [--no-probe]"""
import glob, json, os, re, subprocess, sys
X = os.path.dirname(os.path.abspath(__file__))
M = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
BIN = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2687-d8afa524.app/Contents/MacOS/DrewsChessMachine")
LARGE = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/OpsScripts/lichess_probes_large.json")
sys.path.insert(0, "/Users/andrew/cursor/drews-chess-machine/scripts"); import dcm_arch
ARMS = ["A", "B", "C"]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
LO, HI = (int(args[0]) if args else 1000), (int(args[1]) if len(args) > 1 else 10**9)
PROBE = "--no-probe" not in sys.argv

def ckpts(arm):
    out = {}
    for f in glob.glob(f"{M}/20261122-lsmode-{arm}-*step*.safetensors"):
        s = int(re.search(r"step(\d+)\.safetensors$", f).group(1))
        if s % 1000 == 0: out[s] = f
    return out

def wide(f):
    r = json.loads(dcm_arch.read_header(f)[0]["__metadata__"].get("dcm_test_set_results", "null") or "null")
    w = [t for t in (r or {}).get("sets", []) if t.get("id") == "lichess-wide"]
    return (w[0]["pelo"], w[0]["nll"]) if w and w[0].get("pelo") is not None else (None, None)

def large(arm, s, f):
    path = f"{X}/large_probe_{arm}.jsonl"
    rows = {}
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l); rows[int(re.search(r"step(\d+)\.safetensors", d["model"]).group(1))] = d
    if s not in rows and PROBE:
        tmp = f"{X}/.probe_{arm}_{s}.jsonl"
        subprocess.run([BIN, "--probe-model", f, "--probe-set-file", LARGE, "--probe-out", tmp],
                       check=True, capture_output=True)
        line = open(tmp).readline(); os.remove(tmp)
        with open(path, "a") as out: out.write(line if line.endswith("\n") else line + "\n")
        rows[s] = json.loads(line)
    return rows.get(s)

def step_lines(arm):
    out = {}
    for l in open(f"{X}/train-{arm}.stdout", errors="replace"):
        if l.startswith("[REPLAY] step=") and (m := re.search(r"trainerStep=(\d+)", l)) and int(m.group(1)) % 1000 == 0:
            out[int(m.group(1))] = dict(re.findall(r"(\w+)=([-\d.e+]+)", l))
    return out

C = {a: ckpts(a) for a in ARMS}; S = {a: step_lines(a) for a in ARMS}
steps = sorted(set().union(*C.values())); steps = [s for s in steps if LO <= s <= HI]
v = lambda x, f: format(x, f) if x is not None else ""
head = ["step"] + [f"wide pElo {a}" for a in ARMS] + [f"large pElo {a}" for a in ARMS] + [f"large NLL {a}" for a in ARMS] \
       + [f"pLoss {a}" for a in ARMS] + [f"pEnt {a}" for a in ARMS] + [f"playedP {a}" for a in ARMS] + ["vLoss A", "ms/step A"]
print("| " + " | ".join(head) + " |"); print("|" + "---:|" * len(head))
for s in steps:
    W = {a: wide(C[a][s]) if s in C[a] else (None, None) for a in ARMS}
    L = {a: large(a, s, C[a][s]) if s in C[a] else None for a in ARMS}
    row = [f"{s:,}"] + [v(W[a][0], ".1f") for a in ARMS] + [v(L[a] and L[a]["pElo"], ".1f") for a in ARMS] \
          + [v(L[a] and L[a]["nll"], ".4f") for a in ARMS] \
          + [S[a].get(s, {}).get(k, "") for k in ("pLoss", "pEnt", "playedP") for a in ARMS] \
          + [S["A"].get(s, {}).get("vLoss", ""), S["A"].get(s, {}).get("ms", "")]
    print("| " + " | ".join(row) + " |")
