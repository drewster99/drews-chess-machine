#!/usr/bin/env python3
"""Compares the plateau checkpoints' probe, per-position and numerics outputs in this folder."""
import json, glob, statistics as st, collections
TAGS = [("250k","20261011-zlra-elitecyc-latest-step250000"),("291k","20261011-zlra-elitecyc-latest-step291000"),
        ("385k","20261011-zlra-elitecyc-latest-step384964"),("anneal30k","20261025-zlra-elitecyc-anneal-latest-step30000")]
probe = {t:{json.loads(l)["set"]:json.loads(l) for l in open(f+".probe.jsonl")} for t,f in TAGS}
pos = {t:{(d["set"],d["puzzleId"]):d for d in map(json.loads,open(f+".positions.jsonl"))} for t,f in TAGS}
num = {}
for t,f in TAGS:
    line=[l for l in open(f+".numerics.stdout") if l.startswith('{"event":"numerics"')][-1]; num[t]=json.loads(line)
def row(name, vals): print(f"| {name} | " + " | ".join(vals) + " |")
print("| metric | " + " | ".join(t for t,_ in TAGS) + " |"); print("|---|" + "---:|"*len(TAGS))
for s in ("wide","200"):
    row(f"{s} pElo",[f'{probe[t][s]["pElo"]:.1f}' for t,_ in TAGS])
    row(f"{s} NLL",[f'{probe[t][s]["nll"]:.4f}' for t,_ in TAGS])
    row(f"{s} top-1",[f'{probe[t][s]["argmaxCorrect"]}/{probe[t][s]["n"]}' for t,_ in TAGS])
    row(f"{s} top-5",[f'{probe[t][s]["top5Correct"]}/{probe[t][s]["n"]}' for t,_ in TAGS])
    row(f"{s} avg prob of right move",[f'{probe[t][s]["avgProb"]:.4f}' for t,_ in TAGS])
    row(f"{s} avg rank",[f'{probe[t][s]["avgRank"]:.3f}' for t,_ in TAGS])
W=lambda t:[d for (s,_),d in pos[t].items() if s=="wide"]
row("wide mean entropy (nats)",[f'{st.mean(d["entropyNats"] for d in W(t)):.3f}' for t,_ in TAGS])
row("wide mean top-1 prob",[f'{st.mean(d["top1Prob"] for d in W(t)):.4f}' for t,_ in TAGS])
row("wide mean illegal mass",[f'{st.mean(d["illegalMass"] for d in W(t)):.2e}' for t,_ in TAGS])
row("policy logit |max| (peak)",[f'{probe[t]["wide"]["policy_logit_abs_max"]:.1f} ({probe[t]["wide"]["policy_logit_abs_max_peak"]:.1f})' for t,_ in TAGS])
row("value mean W/D/L",[f'{st.mean(d["valueWin"] for d in W(t)):.3f}/{st.mean(d["valueDraw"] for d in W(t)):.3f}/{st.mean(d["valueLoss"] for d in W(t)):.3f}' for t,_ in TAGS])
for k in ["layer_health_dead_channels","layer_health_mostly_off_channels","layer_health_always_on_channels","layer_health_running_variance_max_over_median","layer_health_running_variance_worst_site","layer_health_se_zero_velocity_units","layer_health_value_fc1_zero_velocity_units","layer_health_non_finite_values"]:
    row(k.replace("layer_health_",""),[str(round(num[t].get(k),1) if isinstance(num[t].get(k),float) else num[t].get(k)) for t,_ in TAGS])
print()
# themes (wide)
themes=sorted(probe["291k"]["wide"]["themes"])
print("| wide theme | " + " | ".join(t for t,_ in TAGS) + " | 291k→385k | 385k→anneal |"); print("|---|" + "---:|"*(len(TAGS)+2))
for th in themes:
    c={t:tuple(map(int,probe[t]["wide"]["themes"][th].split("/"))) for t,_ in TAGS}
    print(f"| {th.replace('lichess','')} | " + " | ".join(f"{c[t][0]}/{c[t][1]}" for t,_ in TAGS) + f" | {c['385k'][0]-c['291k'][0]:+d} | {c['anneal30k'][0]-c['385k'][0]:+d} |")
print()
# rating buckets
print("| wide rating | n | " + " | ".join(f"{t} top-1 / NLL" for t,_ in TAGS) + " |"); print("|---|---:|" + "---:|"*len(TAGS))
b=lambda r: f"{(r//400)*400}-{(r//400)*400+399}"
buckets=collections.defaultdict(list)
for d in W("291k"): buckets[b(d["rating"])].append(d["puzzleId"])
for k in sorted(buckets, key=lambda x:int(x.split('-')[0])):
    ids=buckets[k]
    cells=[]
    for t,_ in TAGS:
        ds=[pos[t][("wide",i)] for i in ids]
        cells.append(f'{sum(d["expectedRank"]==1 for d in ds)/len(ds)*100:.1f}% / {st.mean(d["nll"] for d in ds):.3f}')
    print(f"| {k} | {len(ids)} | " + " | ".join(cells) + " |")
print()
# churn
def top1(t,i): return pos[t][("wide",i)]["expectedRank"]==1
ids=[d["puzzleId"] for d in W("291k")]
for a,c in [("250k","291k"),("291k","385k"),("385k","anneal30k"),("291k","anneal30k")]:
    g=sum(top1(c,i) and not top1(a,i) for i in ids); l=sum(top1(a,i) and not top1(c,i) for i in ids)
    dn=[pos[c][("wide",i)]["nll"]-pos[a][("wide",i)]["nll"] for i in ids]
    print(f"- {a} → {c}: top-1 gained {g}, lost {l} (net {g-l:+d}); per-puzzle NLL change mean {st.mean(dn):+.4f}, mean |change| {st.mean(abs(x) for x in dn):.4f}")
