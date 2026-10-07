import json,os,re,sys
sys.path.insert(0,"/Users/andrew/cursor/drews-chess-machine/experiments/20261005-lr-schedule-ab")
import bn_liveness as bl
E="/Users/andrew/cursor/drews-chess-machine/experiments/20261005-lr-schedule-ab/"
def load(fs):
    d={}
    for f,off in fs:
        if os.path.exists(E+f):
            for l in open(E+f):
                if l.strip(): r=json.loads(l); d[r["training_step"]+off]=r["pElo"]
    return d
A=load([("probes-A.jsonl",0),("probes-A-seg1.jsonl",36000)]); B=load([("probes-B.jsonl",0),("probes-B-seg1.jsonl",36000)])
K=load([("probes-Bleaky.jsonl",0)]); KA=load([("probes-Bleakyall.jsonl",0)]); KS=load([("probes-Bsilu.jsonl",0)]); KC=load([("probes-Bsilu-clip1-seg1.jsonl",18000)]); KT=load([("probes-Bsilu-ctl15-seg1.jsonl",18000)]); K2=load([("probes-Bsilu-clip2-seg1.jsonl",18000)]); K5=load([("probes-Bsilu-clip5-seg1.jsonl",18000)])
LR={}
for f in ["dcm_log_20261005-013235.txt","dcm_log_20261005-171541.txt"]:
    for l in open("/Users/andrew/Library/Logs/DrewsChessMachine/"+f):
        m=re.search(r"\[REPLAY\] step=\d+ .*?lr=([0-9.e+-]+) .*trainerStep=(\d+)",l)
        if m: LR[int(m.group(2))]=float(m.group(1))
RUNS=[("A (ReLU, const 0.01)",A),("B (ReLU)",B),("B-leaky (value head)",K),("B-leakyall",KA),("B-silu",KS),("B-silu clip 1.0 (from 18k)",KC),("B-silu control cap 15 (from 18k)",KT),("B-silu clip 2.0 (from 18k)",K2),("B-silu clip 5.0 (from 18k)",K5)]
CPS={n:bl.checkpoints(n) for n,_ in RUNS}
cache={}
def parked(n,s):
    if s not in CPS[n]: return ""
    key=(n,s)
    if key not in cache: cache[key]=bl.analyze(CPS[n][s])["policy.pre_bn"]["parked"]
    return str(cache[key])
f=lambda d,s: f"{d[s]:.1f}" if s in d else ""
print("| step | B's LR | A (ReLU, constant 0.01) | B (ReLU, cycle) | B leaky value head | B leaky everywhere | B SiLU tower | B SiLU, clip 1.0 from 18k | B SiLU, cap 15 control from 18k | B SiLU, clip 2.0 from 18k | B SiLU, clip 5.0 from 18k | policy pre-BN parked (of 128): A | B | B leaky value | B leaky all | B SiLU | B SiLU clip 1.0 | B SiLU control | clip 2.0 | clip 5.0 |")
print("|"+"---:|"*20)
for s in range(1000,40001,1000):
    pk=[parked(n,s) for n,_ in RUNS]
    print(f"| {s:,} | {'%.3g'%LR[s] if s in LR else ''} | {f(A,s)} | {f(B,s)} | {f(K,s)} | {f(KA,s)} | {f(KS,s)} | {f(KC,s)} | {f(KT,s)} | {f(K2,s)} | {f(K5,s)} | "+" | ".join(pk)+" |")
