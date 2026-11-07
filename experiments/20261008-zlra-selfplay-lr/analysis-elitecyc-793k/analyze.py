#!/usr/bin/env python3
"""Compares R-elitecyc checkpoints 250k..793k: probes, per-position policy/value, numerics audit, BN sites.
Older checkpoints' outputs come from ../analysis-elitecyc-plateau (same commands, run.sh there)."""
import json, statistics as st, re, collections, os
P="../analysis-elitecyc-plateau/"
TAGS=[("250k",P+"20261011-zlra-elitecyc-latest-step250000"),("291k",P+"20261011-zlra-elitecyc-latest-step291000"),
      ("385k",P+"20261011-zlra-elitecyc-latest-step384964"),("500k","20261011-zlra-elitecyc-latest-step500000"),
      ("600k","20261011-zlra-elitecyc-latest-step600000"),("700k","20261011-zlra-elitecyc-latest-step700000"),
      ("793k","20261011-zlra-elitecyc-latest-step793000")]
probe={t:{json.loads(l)["set"]:json.loads(l) for l in open(f+".probe.jsonl")} for t,f in TAGS}
pos={t:{(d["set"],d["puzzleId"]):d for d in map(json.loads,open(f+".positions.jsonl"))} for t,f in TAGS}
num={}; txt={}
for t,f in TAGS:
    L=open(f+".numerics.stdout").read().splitlines(); txt[t]=L
    num[t]=json.loads([l for l in L if l.startswith('{"event":"numerics"')][-1])
def row(n,v): print(f"| {n} | "+" | ".join(v)+" |")
def hdr(first): print(f"| {first} | "+" | ".join(t for t,_ in TAGS)+" |"); print("|---|"+"---:|"*len(TAGS))
hdr("metric")
for s in ("wide","200"):
    row(f"{s} pElo",[f'{probe[t][s]["pElo"]:.1f}' for t,_ in TAGS])
    row(f"{s} NLL",[f'{probe[t][s]["nll"]:.4f}' for t,_ in TAGS])
    row(f"{s} top-1",[f'{probe[t][s]["argmaxCorrect"]}' for t,_ in TAGS])
    row(f"{s} top-5",[f'{probe[t][s]["top5Correct"]}' for t,_ in TAGS])
W=lambda t:[d for (s,_),d in pos[t].items() if s=="wide"]
row("wide avg prob right move",[f'{probe[t]["wide"]["avgProb"]:.4f}' for t,_ in TAGS])
row("wide mean policy entropy",[f'{st.mean(d["entropyNats"] for d in W(t)):.3f}' for t,_ in TAGS])
row("wide mean top-1 prob",[f'{st.mean(d["top1Prob"] for d in W(t)):.4f}' for t,_ in TAGS])
row("wide mean illegal mass",[f'{st.mean(d["illegalMass"] for d in W(t)):.1e}' for t,_ in TAGS])
row("policy logit abs max (peak)",[f'{probe[t]["wide"]["policy_logit_abs_max"]:.1f} ({probe[t]["wide"]["policy_logit_abs_max_peak"]:.1f})' for t,_ in TAGS])
row("puzzle value W/D/L (side to move)",[f'{st.mean(d["valueWin"] for d in W(t)):.3f}/{st.mean(d["valueDraw"] for d in W(t)):.3f}/{st.mean(d["valueLoss"] for d in W(t)):.3f}' for t,_ in TAGS])
row("numerics overall",[num[t]["overall"] for t,_ in TAGS])
row("policy offset x init",[f'{num[t]["policy_offset_ratio_to_init"]:.3f}' for t,_ in TAGS])
row("value offset x init",[f'{num[t]["value_offset_ratio_to_init"]:.4f}' for t,_ in TAGS])
row("bf16 policy KL vs fp32",[f'{num[t]["policy_kl_bf16_fp32_from_pre_bn"]:.1e}' for t,_ in TAGS])
row("bf16 value CE delta",[f'{num[t]["value_ce_delta_bf16_fp32_from_pre_bn"]:+.1e}' for t,_ in TAGS])
row("bf16 policy top-2 ties",[str(num[t]["policy_top2_ties_bf16_fp32_from_pre_bn"]) for t,_ in TAGS])
row("rv max/median (site)",[f'{num[t]["layer_health_running_variance_max_over_median"]:.0f}' for t,_ in TAGS])
def grab(t,pat):
    for l in txt[t]:
        m=re.search(pat,l)
        if m: return m.group(1)
    return "-"
row("worst BN |mean|/std",[grab(t,r"worst \|mean\|/std ([\d.]+ at \w+)") for t,_ in TAGS])
row("rv>=10x channels",[grab(t,r"rvOver10xMedian=(\d+)") for t,_ in TAGS])
row("value fc1 low-velocity units",[grab(t,r"value.fc1.weight .*low (\d+)") for t,_ in TAGS])
row("value fc1 median unit velocity",[grab(t,r"median unit norm ([\d.e-]+)") for t,_ in TAGS])
row("value fp32 CE (audit positions)",[grab(t,r"^value fp32: .*CE ([\d.]+)") for t,_ in TAGS])
row("value start-position W/D/L",[grab(t,r"^value fp32: .*start W/D/L ([\d./]+)") for t,_ in TAGS])
row("policy legal spread median",[grab(t,r"^policy fp32: .*legal spread median ([\d.]+)") for t,_ in TAGS])
row("degraded findings",[grab(t,r"findings: \d+ BAD, (\d+) degraded") for t,_ in TAGS])
print()
# per-site BN table, rv max/median
sites=["stem.bn","blocks.0.bn1","blocks.0.bn2","blocks.1.bn1","blocks.1.bn2","blocks.2.bn1","blocks.2.bn2","tower_final_bn","policy.pre_bn","value.bn"]
hdr("BN site rv max/median (rv>=10x ch)")
for s in sites:
    cells=[]
    for t,_ in TAGS:
        l=[x for x in txt[t] if x.strip().startswith(s+" ") and "[" in x]
        if not l: cells.append("-"); continue
        m=re.findall(r"([\d.]+) \[\d+\]\s+(\d+)\s*$",l[0]); cells.append(f"{m[0][0]} ({m[0][1]})" if m else "?")
    row(s,cells)
hdr("BN site min/median pass-through P")
for s in sites[1:]:
    cells=[]
    for t,_ in TAGS:
        l=[x for x in txt[t] if x.strip().startswith(s+" ") and "[" not in x and re.search(r"\d\.\d{6}",x)]
        cells.append("/".join(l[0].split()[-2:]) if l else "-")
    row(s,cells)
print()
themes=sorted(probe["793k"]["wide"]["themes"])
hdr("wide theme (correct)")
for th in themes:
    row(th.replace("lichess",""),[probe[t]["wide"]["themes"][th].split("/")[0] for t,_ in TAGS])
print()
b=lambda r:(r//400)*400
ids=[d["puzzleId"] for d in W("793k")]
bk=collections.defaultdict(list)
for i in ids: bk[b(pos["793k"][("wide",i)]["rating"])].append(i)
hdr("wide rating bucket: top-1 % / NLL")
for k in sorted(bk):
    row(f"{k}-{k+399} (n={len(bk[k])})",[f'{sum(pos[t][("wide",i)]["expectedRank"]==1 for i in bk[k])/len(bk[k])*100:.1f} / {st.mean(pos[t][("wide",i)]["nll"] for i in bk[k]):.3f}' for t,_ in TAGS])
print()
top1=lambda t,i:pos[t][("wide",i)]["expectedRank"]==1
for a,c in [("385k","500k"),("500k","600k"),("600k","700k"),("700k","793k"),("385k","793k")]:
    g=sum(top1(c,i) and not top1(a,i) for i in ids); l=sum(top1(a,i) and not top1(c,i) for i in ids)
    dn=[pos[c][("wide",i)]["nll"]-pos[a][("wide",i)]["nll"] for i in ids]
    print(f"- {a} -> {c}: top-1 gained {g}, lost {l} (net {g-l:+d}); per-puzzle NLL change mean {st.mean(dn):+.4f}, mean |change| {st.mean(abs(x) for x in dn):.4f}")
