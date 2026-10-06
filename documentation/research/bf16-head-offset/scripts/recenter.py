import json,numpy as np
from fwd import load,game_states,encode,forward
from archnorm import norm_arch_md
G='/Users/andrew/Library/Application Support/DrewsChessMachine/LichessBot/Games/2026/09/'
X=[]
for g in ['20260928-163650-Qvlv85tG.json','20260928-162907-ZQRPySw4.json']:
    d=json.load(open(G+g)); ucis=[m['uciAsGiven'] for m in d['moves']]
    for st,rep,mask in game_states(ucis)[:-1]: X.append(encode(st,rep,mask))
P='/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20260629-mini1b-Coxw-resume-replay-step277000.safetensors'
md,T=load(P); arch=norm_arch_md(md)
W=T['value.wdl_fc2.weight']; b=T['value.wdl_fc2.bias']; m=W.mean(0)
print('rows cosine:',[round(float(W[i]@W[j]/np.linalg.norm(W[i])/np.linalg.norm(W[j])),5) for i,j in ((0,1),(0,2),(1,2))])
T2=dict(T); T2['value.wdl_fc2.weight']=W-m; T2['value.wdl_fc2.bias']=b-b.mean()
kl1=[];kl2=[];mx=[];distinct=set()
for x in X:
    lg,p,acts=forward(T,arch,x,'f64',return_all=True,md=md); f1=acts['fc1']
    _,pb=forward(T,arch,x,'bf16',md=md); lg2,pb2=forward(T2,arch,x,'bf16',md=md)
    kl1.append(np.sum(p*np.log(p/np.maximum(pb,1e-30)))); kl2.append(np.sum(p*np.log(p/np.maximum(pb2,1e-30)))); mx.append(np.abs(lg2).max()); distinct.add(tuple(pb2))
    cm=m@f1+b.mean()
print(f"positions {len(X)} | original bf16 KL mean {np.mean(kl1):.4f} max {np.max(kl1):.3f} | recentered bf16 KL mean {np.mean(kl2):.6f} max {np.max(kl2):.4f} | recentered max|logit| {max(mx):.2f} | distinct triples {len(distinct)}")
