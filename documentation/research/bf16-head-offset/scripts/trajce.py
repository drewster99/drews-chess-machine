import pickle,json,sys,numpy as np
from fwd3 import *
P=[p for p in pickle.load(open('posset.pkl','rb')) if p['src']=='corpus']
X=np.stack([p['x'] for p in P]).astype(np.float64); lab=np.array([p['label'] for p in P])
r=json.load(open('scan2.json'))
def pick(mid,step):
    c=sorted([x for x in r if x['model_id']==mid and x['step']==str(step)],key=lambda x:x['path']); return c[0]['path'] if c else None
for spec in sys.argv[1:]:
    mid,step=spec.split(':'); p=pick(mid,step)
    if not p: print('missing',spec); continue
    md,T=load(p); assert md['model_id']==mid and md['training_step']==step
    arch=norm_arch(md['architecture']); o=forward_batched(prep(T,'bf16'),arch,X,'f64')
    vl=o['vl']; e=np.exp(vl-vl.max(1,keepdims=True)); p64=e/e.sum(1,keepdims=True)
    vq=bf16(vl); e=np.exp(vq-vq.max(1,keepdims=True)); pq=bf16(e/e.sum(1,keepdims=True))
    ce64=-np.log(p64[np.arange(len(P)),lab]).mean(); ceq=-np.log(np.maximum(pq[np.arange(len(P)),lab],1e-30)).mean()
    pce=[];pceq=[]
    for i,pp in enumerate(P):
        a=o['pl'][i][pp['legal']]; t=pp['legal'].tolist().index(pp['target'])
        for arr,out in ((a,pce),(bf16(a),pceq)):
            z=arr-arr.max(); out.append(-(z[t]-np.log(np.exp(z).sum())))
    print(f"{mid[-4:]} {step:>7} | value CE fp64 {ce64:.4f} bf16-out {ceq:.4f} | shared logit med {np.median(vl.mean(1)):8.1f} | ties {np.mean([len(set(x))<3 for x in vq]):.3f} | policy CE(legal) fp64 {np.mean(pce):.4f} bf16-out {np.mean(pceq):.4f}",flush=True)
