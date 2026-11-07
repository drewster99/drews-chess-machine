#!/usr/bin/env python3
"""Per-50k-trainer-step means of R-elitecyc step-line metrics over every segment's stdout (the anneal branch excluded)."""
import sys,re,glob,collections
b=collections.defaultdict(list)
for f in sorted(glob.glob("../train-elitecyc*.stdout")):
    if "anneal" in f: continue
    for l in open(f,errors="replace"):
        if "[REPLAY] step=" not in l or "pEnt=--" in l: continue
        d=dict(re.findall(r"(\w+)=([-\d.e]+)",l))
        if "trainerStep" in d and "pLoss" in d: b[int(d["trainerStep"])//50000].append(d)
print("| trainer steps | lines | pLoss | vLoss | pEnt | pD | gNorm |"); print("|---|---:|---:|---:|---:|---:|---:|")
for k in sorted(b):
    r=b[k]; m=lambda f: sum(float(x[f]) for x in r)/len(r)
    print(f"| {k*50}k-{k*50+50}k | {len(r)} | {m('pLoss'):.3f} | {m('vLoss'):.3f} | {m('pEnt'):.3f} | {m('pD'):.3f} | {m('gNorm'):.3f} |")
