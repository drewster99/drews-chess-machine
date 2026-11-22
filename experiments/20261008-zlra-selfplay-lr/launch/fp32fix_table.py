#!/usr/bin/env python3
"""Per-1000-step table: R-elitecyc-fp32fix vs R-elitecyc (bf16), lichess-wide first, then lichess-large.

fp32fix lichess-wide comes from each checkpoint's embedded dcm_test_set_results; bf16 lichess-wide from the
training-tables page (its 1000-step files were thinned to 10k); lichess-large from the probe jsonl files.
Usage: fp32fix_table.py [from_step] [to_step]
"""
import glob, json, os, re, sys
sys.path.insert(0, "/Users/andrew/cursor/drews-chess-machine/scripts"); import dcm_arch
X = "/Users/andrew/cursor/drews-chess-machine/experiments/20261008-zlra-selfplay-lr"
M = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models/")
LO = int(sys.argv[1]) if len(sys.argv) > 1 else 420000
HI = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

def step_lines(pat):
    out = {}
    for p in sorted(glob.glob(f"{X}/{pat}"), key=lambda p: int(re.search(r"seg(\d+)", p).group(1))):
        for l in open(p, errors="replace"):
            if l.startswith("[REPLAY] step=") and (m := re.search(r"trainerStep=(\d+)", l)) and int(m.group(1)) % 1000 == 0:
                out[int(m.group(1))] = dict(re.findall(r"(\w+)=([-\d.e+]+)", l))
    return out

def bf16_wide():
    h = open(f"{X}/tables/training_tables.html").read()
    sec = h[h.index("R-elitecyc <span"):h.index("R-elitecyc-anneal <span")]
    out = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", sec, re.S):
        c = [re.sub("<[^>]+>", "", x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if c and re.fullmatch(r"[\d,]+", c[0]) and c[2]:
            out[int(c[0].replace(",", ""))] = (float(c[2]), float(c[3]))
    return out

def fp32_wide(s):
    fn = f"{M}20261114-zlra-elitecyc-fp32fix-latest-step{s}.safetensors"
    if not os.path.exists(fn): return None
    r = json.loads(dcm_arch.read_header(fn)[0]["__metadata__"]["dcm_test_set_results"])
    w = [t for t in r["sets"] if t["id"] == "lichess-wide"][0]
    return (w["pelo"], w["nll"])

def large(path):
    try: return json.loads(open(path).readline())
    except OSError: return None

F, B, BW = step_lines("train-elitecyc-fp32fix-seg*.stdout"), step_lines("train-elitecyc-seg*.stdout"), bf16_wide()
BL = {int(re.search(r"step(\d+)", (d := json.loads(l))["model"]).group(1)): d
      for l in open("/Users/andrew/cursor/drews-chess-machine/experiments/20261108-elitecyc-heldout/large_probe.jsonl")}
v = lambda x, f: format(x, f) if x is not None else ""
cols = ["step", "wide pElo fp32", "wide pElo bf16", "Δ wide", "wide NLL fp32", "wide NLL bf16",
        "large pElo fp32", "large pElo bf16", "Δ large", "large NLL fp32", "large top-1", "large top-5",
        "loss", "Δ loss", "pLoss", "vLoss", "pEnt", "pD", "gNorm", "LR", "ms/step fp32", "ms/step bf16", "buffer avg plies/game"]
print("| " + " | ".join(cols) + " |"); print("|" + "---:|" * len(cols))
for s in sorted(k for k in F if LO <= k <= HI):
    f, b, fw, bw = F[s], B.get(s), fp32_wide(s), BW.get(s)
    p, q = large(f"{X}/analysis-fp32/fp32fix_large_probe_step{s}.jsonl"), BL.get(s)
    row = [f"{s:,}", v(fw and fw[0], ".1f"), v(bw and bw[0], ".0f"), v(fw[0] - bw[0] if fw and bw else None, "+.1f"),
           v(fw and fw[1], ".3f"), v(bw and bw[1], ".3f"),
           v(p and p["pElo"], ".1f"), v(q and q["pElo"], ".1f"), v(p["pElo"] - q["pElo"] if p and q else None, "+.1f"),
           v(p and p["nll"], ".4f"), v(p and p["argmaxCorrect"], ","), v(p and p["top5Correct"], ","),
           f["loss"], v(float(f["loss"]) - float(b["loss"]) if b else None, "+.4f"), f["pLoss"], f["vLoss"], f["pEnt"], f["pD"],
           f["gNorm"], f["lr"], f"{float(f['ms']):.0f}", v(b and float(b["ms"]), ".0f"), f"{int(f['plies']) / int(f['games']):.1f}"]
    print("| " + " | ".join(row) + " |")
