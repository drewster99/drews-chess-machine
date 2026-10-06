import pickle,json,numpy as np
from fwd3 import *
P=pickle.load(open('posset.pkl','rb')); L=[p for p in P if p['ours']]
X=np.stack([p['x'] for p in L]).astype(np.float64)
md,T=load('/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20260629-mini1b-Coxw-resume-replay-step277000.safetensors')
arch=norm_arch_md(md)
res={}
res['f64']=forward_batched(prep(T,'f64'),arch,X,'f64',md=md)
res['bf16']=forward_batched(prep(T,'bf16'),arch,X,'bf16',md=md)
o=dict(res['f64']); o['pl']=bf16(o['pl']); res['f64->bf16out']=o
for name,ob in res.items():
    errs=[];ties_ok=0;top1=0
    for k,p in enumerate(L):
        lg=ob['pl'][k][p['legal']].astype(np.float32); e=np.exp(lg-lg.max()); pr=e/e.sum()
        emu={p['legal_uci'][j]:pr[j] for j in range(len(pr))}
        tm=p['obs']['topMoves']
        errs.append(max(abs(emu[t['uci']]-t['probability']) for t in tm))
        top1+= p['legal_uci'][int(np.argmax(pr))]==tm[0]['uci'] or pr.max()==emu[tm[0]['uci']]
    print(f"{name:14} topMoves max-abs-err median {np.median(errs):.2e} p90 {np.percentile(errs,90):.2e}  top1 agree {top1}/{len(L)}")
# observed ties reproduce?
for k,p in enumerate(L):
    tm=p['obs']['topMoves']
    if tm[0]['probability']==tm[1]['probability']:
        for name,ob in res.items():
            lg=ob['pl'][k][p['legal']]; d={p['legal_uci'][j]:lg[j] for j in range(len(lg))}
            print(p['game'],p['ply'],name,[(t['uci'],round(float(d[t['uci']]),4)) for t in tm[:3]])
