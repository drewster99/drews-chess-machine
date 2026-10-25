# Markdown table: each 10k LR-cycle peak (1k, 11k, ...) for the three live runs, ±1,000 steps.
import re,statistics,os
L=os.path.expanduser("~/Library/Logs/DrewsChessMachine/")
kv=lambda l,k:(m.group(1) if (m:=re.search(r"(?<![\w.])"+k+r"=([^\s)]+)",l)) else None)
def num(x):
    m=re.match(r"[+-]?[0-9.]+(e[+-]?\d+)?",x or ""); return float(m.group(0)) if m else None
runs={"B-siluall":["dcm_log_20261008-125718.txt","dcm_log_20261009-163612.txt","dcm_log_20261009-213842.txt","dcm_log_20261010-082147.txt","dcm_log_20261010-111314.txt","dcm_log_20261010-211711.txt"],
      "R-replay":["dcm_log_20261008-094622.txt","dcm_log_20261010-122003.txt","dcm_log_20261010-211714.txt"],
      "R-fixedlr":["dcm_log_20261008-094623.txt","dcm_log_20261010-113351.txt","dcm_log_20261010-224411.txt"]}
data={}
for name,logs in runs.items():
    rows=[];clips=[];alarms=[]
    for f in logs:
        for l in open(L+f,errors="replace"):
            if "[REPLAY] step=" in l:
                ts=kv(l,"trainerStep"); s=int(ts) if ts else int(kv(l,"step"))
                if kv(l,"lr"): rows.append((s,num(kv(l,"lr")),num(kv(l,"gNormMax")),num(kv(l,"gNorm")),num(kv(l,"loss"))))
            elif "[STATS] elapsed=" in l and kv(l,"steps"):
                p,v=num(kv(l,"pLoss")),num(kv(l,"vLoss"))
                if p is None or v is None: continue
                rows.append((int(kv(l,"steps")),num(kv(l,"lr")),num(kv(l,"gNormMax")),num(kv(l,"gNorm")),p+v))
            elif "[GRAD-CLIP] trainerStep" in l and "applied=true" in l:
                clips.append((int(kv(l,"trainerStep")),float(kv(l,"preNorm"))))
            elif re.search(r"\[ALARM\] health (raise|escalate|worsen) rule=bn_running_variance",l):
                alarms.append(int(kv(l,"trainerStep")))
    data[name]=(rows,clips,alarms)
top=max(max(r[0] for r in data[n][0]) for n in runs)
print("| peak step | run | max LR | max pre-clip gNorm | median gNorm | clipped steps | BN-variance alarms | loss before | loss max |")
print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for p in range(1000,top+1,10000):
    for name in runs:
        rows,clips,alarms=data[name]
        w=[r for r in rows if p-1000<=r[0]<=p+1000]
        if not w:
            print(f"| {p:,} | {name} |  |  |  |  |  |  |  |"); continue
        lr=max(r[1] for r in w if r[1] is not None)
        g=[r[2] for r in w if r[2] is not None]; t=[r[3] for r in w if r[3] is not None]
        c=[x for x in clips if p-1000<=x[0]<=p+1000]; a=[x for x in alarms if p-1000<=x<=p+1000]
        before=[r for r in rows if r[0]<=p-1000]; lb=before[-1][4] if before else None
        print(f"| {p:,} | {name} | {'0.01 fixed' if name=='R-fixedlr' else f'{lr:.3f}'} | {f'{max(g):.2f}' if g else ''} | {f'{statistics.median(t):.2f}' if t else ''} | {len(c)} | {len(a)} | {f'{lb:.3f}' if lb is not None else ''} | {max(r[4] for r in w):.3f} |")
