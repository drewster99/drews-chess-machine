import re,json,glob,os,sys
sys.path.insert(0,"/Users/andrew/cursor/drews-chess-machine/scripts"); import dcm_arch
L=os.path.expanduser("~/Library/Logs/DrewsChessMachine/"); M=os.path.expanduser("~/Library/Application Support/DrewsChessMachine/")
kv=lambda l,k:(m.group(1) if (m:=re.search(r"(?<![\w.])"+re.escape(k)+r"=([^\s)]+)",l)) else None)
def f(x,d=3):
    try: return f"{float(x):.{d}f}"
    except: return ""
def wide(fn):
    r=json.loads(dcm_arch.read_header(fn)[0]["__metadata__"]["dcm_test_set_results"])
    w=[s for s in r["sets"] if s["id"]=="lichess-wide"][0]; return w["pelo"],w["nll"]
def overwrite(out,step):
    # A pass that goes back to `step` overwrites what earlier passes logged from `step` on: keep only the latest.
    while out and out[-1][0]>=step: out.pop()
def rvs(lines):
    out=[]
    for l in lines:
        if "[LAYER-HEALTH] live" in l:
            s=int(kv(l,"trainerStep")); overwrite(out,s); out.append((s,kv(l,"rvMaxOverMedian").split("@")[0]))
    return out
def near(rv,s): return min(rv,key=lambda t:abs(t[0]-s))[1] if rv else ""
def dated(log,lines):
    # Prefix each line's HH:MM with its date: the log name gives the start day, a clock wrap adds a day.
    import datetime
    d=datetime.date(int(log[8:12]),int(log[12:14]),int(log[14:16])); last=""; out=[]
    for l in lines:
        t=l[:8]
        if re.match(r"\d\d:\d\d:\d\d",t):
            if last and t<last: d+=datetime.timedelta(days=1)
            last=t
        out.append(f"{d:%m-%d} {l[:5]}")
    return out
def replay(logs,glob_):
    if isinstance(logs,str): logs=[logs]
    lines=[]; stamps=[]; segstart=[]
    for log in logs:
        ls=open(L+log,errors="replace").read().splitlines(); segstart.append(len(lines)); lines+=ls; stamps+=dated(log,ls)
    rv=rvs(lines); pe={}
    for fn in glob.glob(M+glob_):
        pe[int(re.search(r"step(\d+)",fn).group(1))]=wide(fn)
    rows={}; seg={}
    for i,l in enumerate(lines):
        if "[REPLAY] step=" not in l: continue
        ts=kv(l,"trainerStep"); s=int(ts) if ts else int(kv(l,"step"))
        for k in [k for k in rows if k>s]: del rows[k]; seg.pop(k,None)
        if s%1000==0 or int(kv(l,"step"))==1:
            rows[s]=(stamps[i],l); seg[s]=sum(1 for b in segstart if b<=i)-1
    print("| step | time | wide pElo | NLL | pLoss | vLoss | pEnt | pD | gNorm | LR | mom | ms/step | rv max/med | buffer avg plies/game |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    pg=pp=None
    for s in sorted(rows):
        st,l=rows[s]; g,p=int(kv(l,"games")),int(kv(l,"plies"))
        if pg is not None and g<pg: pg=pp=None   # a resumed segment restarts its feed counters
        apg=f"{(p-pp)/(g-pg):.1f}" if pg is not None and g>pg else ""
        pg,pp=g,p
        e=pe.get(s); 
        print(f"| {s:,} | {st} | {f(e[0],0) if e else ''} | {f(e[1]) if e else ''} | {f(kv(l,'pLoss'))} | {f(kv(l,'vLoss'))} | {f(kv(l,'pEnt'))} | {kv(l,'pD')} | {f(kv(l,'gNorm'))} | {kv(l,'lr')} | {kv(l,'mom')} | {f(kv(l,'ms'),0)} | {near(rv,s)} | {apg} |")
def gui(logs,runid):
    if isinstance(logs,str): logs=[logs]
    lines=[]; stamps=[]
    for log in logs:
        ls=open(L+log,errors="replace").read().splitlines(); lines+=ls; stamps+=dated(log,ls)
    rv=rvs(lines); ticks=[]; stats=[]; promos=[]
    for idx,l in enumerate(lines):
        if "[TACTICAL-LICHESS] tick set=wide" in l:
            ts=int(kv(l,"step")); overwrite(ticks,ts); ticks.append((ts,float(kv(l,"pElo")),float(kv(l,"NLL"))))
        elif "[STATS]" in l and kv(l,"steps") and "ratio=(" in l: stats.append((int(kv(l,"steps")),(idx,l)))
        elif "Verdict: PROMOTED" in l: promos.append(idx)
    # last crossing of each 1000 (trainer rewinds on promotion)
    rows={}
    for i,(s,l) in enumerate(stats):
        for r in [r for r in rows if r>s]: del rows[r]
        k=s//1000
        if i and stats[i-1][0]//1000<k: rows[k*1000]=l
    def promos_by(i): return sum(1 for p in promos if p<i)
    print("| step | time | wide pElo (±500) | NLL | pLoss | vLoss | pEnt | pD | gNorm | ratio | self-play M pos/hr | train M pos/hr | buffer avg plies/game | new-game draw % | promotions | rv max/med |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in sorted(rows):
        li,l=rows[s]; t=[x for x in ticks if abs(x[0]-s)<=500]
        pe=f"{sum(x[1] for x in t)/len(t):.0f}" if t else ""; nl=f"{sum(x[2] for x in t)/len(t):.3f}" if t else ""
        sp=kv(l,"spRate"); tr=kv(l,"trainRate")
        comp=re.search(r"comp=\(games=\d+ lenG=([0-9.]+) lenP=[0-9.]+ W=[0-9.]+ D=([0-9.]+)",l)
        print(f"| {s:,} | {stamps[li]} | {pe} | {nl} | {f(kv(l,'pLoss'))} | {f(kv(l,'vLoss'))} | {f(kv(l,'pEnt'))} | {kv(l,'pD')} | {f(kv(l,'gNorm'))} | {kv(l,'cur')} | {f(int(sp.split('/')[0])/1e6,1) if sp else ''} | {f(int(tr.split('/')[0])/1e6,1) if tr else ''} | {comp.group(1) if comp else ''} | {f(float(comp.group(2))*100,1) if comp else ''} | {promos_by(li)} | {near(rv,s)} |")
which=sys.argv[1]
if which=="replay": replay(["dcm_log_20261008-094622.txt","dcm_log_20261010-122003.txt","dcm_log_20261010-211714.txt"],"Models/20261008-zlra-replay-step*.safetensors")
if which=="silu": replay(["dcm_log_20261008-125718.txt","dcm_log_20261009-163612.txt","dcm_log_20261009-213842.txt","dcm_log_20261010-082147.txt","dcm_log_20261010-111314.txt","dcm_log_20261010-211711.txt","dcm_log_20261014-214445.txt"],"Models/20261008-lrBsiluall-relcap-replay-step*.safetensors")
if which=="elite": replay("dcm_log_20261011-130546.txt","Models/20261011-zlra-elite-latest-step*.safetensors")
if which=="elitecyc": replay(["dcm_log_20261011-213940.txt","dcm_log_20261012-153015.txt","dcm_log_20261014-201112.txt","dcm_log_20261015-131635.txt","dcm_log_20261015-230809.txt","dcm_log_20261017-010300.txt","dcm_log_20261017-182142-2.txt","dcm_log_20261017-201253.txt","dcm_log_20261017-235728.txt","dcm_log_20261018-142539-2.txt","dcm_log_20261018-190000-2.txt","dcm_log_20261019-230139.txt","dcm_log_20261023-072629.txt","dcm_log_20261024-224808-2.txt","dcm_log_20261026-000416.txt","dcm_log_20261026-133117.txt","dcm_log_20261026-165556.txt","dcm_log_20261029-230847-2.txt"],"Models/20261011-zlra-elitecyc-latest-step*.safetensors")
if which=="anneal": replay("dcm_log_20261025-142424.txt","Models/20261025-zlra-elitecyc-anneal-latest-step*.safetensors")
if which=="elitefresh": replay(["dcm_log_20261012-153151.txt","dcm_log_20261014-201112-2.txt","dcm_log_20261017-145133.txt","dcm_log_20261017-201253-2.txt","dcm_log_20261017-235728-2.txt","dcm_log_20261018-142539.txt","dcm_log_20261018-190000.txt","dcm_log_20261019-230141.txt","dcm_log_20261023-072631.txt","dcm_log_20261024-224808-3.txt","dcm_log_20261028-213703.txt","dcm_log_20261029-230847.txt"],"Models/20261012-zlra-elitefresh-latest-step*.safetensors")
if which=="fixed": gui(["dcm_log_20261008-094623.txt","dcm_log_20261010-113351.txt","dcm_log_20261010-224411.txt","dcm_log_20261012-161833.txt","dcm_log_20261014-140421.txt","dcm_log_20261015-102200.txt","dcm_log_20261024-023842.txt"],"A2599216")
