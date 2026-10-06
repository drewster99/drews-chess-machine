import pickle,numpy as np,sys
from fwd3 import *
P=pickle.load(open('posset.pkl','rb')); L=[p for p in P if p['ours']]
X=np.stack([p['x'] for p in L]).astype(np.float64)
md,T=load('/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20260629-mini1b-Coxw-resume-replay-step277000.safetensors')
arch=norm_arch_md(md)
def outround(ob,q=bf16):
    ob=dict(ob); ob['pl']=q(ob['pl']); ob['vl']=q(ob['vl']); e=np.exp(ob['vl']-ob['vl'].max(1,keepdims=True)); ob['vp']=q(e/e.sum(1,keepdims=True)); return ob
V={}
V['f64 + bf16 out']=outround(forward_batched(prep(T,'f64'),arch,X,'f64',md=md))
V['f32 all + bf16 out']=outround(forward_batched(prep(T,'f32'),arch,X,'f32',md=md))
V['bf16 weights, f64 acts, bf16 out']=outround(forward_batched(prep(T,'bf16'),arch,X,'f64',md=md))
V['bf16 weights+conv outs, bf16 out']=outround(forward_batched(prep(T,'bf16'),arch,X,'bf16:mm',md=md))
for name,ob in V.items():
    errs=[];wm=0;werr=[];ties=0;pexact=0
    for k,p in enumerate(L):
        o=p['obs']; obs=np.array([o['win'],o['draw'],o['loss']],dtype=np.float32)
        wm+=np.array_equal(ob['vp'][k].astype(np.float32),obs); werr.append(np.abs(ob['vp'][k]-obs).max())
        lg=ob['pl'][k][p['legal']].astype(np.float32); e=np.exp(lg-lg.max()); pr=e/e.sum()
        emu={p['legal_uci'][j]:pr[j] for j in range(len(pr))}
        tm=o['topMoves']; errs.append(max(abs(emu[t['uci']]-t['probability']) for t in tm))
        if tm[0]['probability']==tm[1]['probability']: ties+= emu[tm[0]['uci']]==emu[tm[1]['uci']]
    print(f"{name:36} WDL exact {wm}/{len(L)} | policy top5 maxerr median {np.median(errs):.2e} p90 {np.percentile(errs,90):.2e} exact<1e-6 {sum(e<1e-6 for e in errs)}/{len(L)} | ties reproduced {ties}/6")
