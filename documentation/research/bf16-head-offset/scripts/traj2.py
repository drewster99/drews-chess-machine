import json,sys,struct,numpy as np
from fwd3 import bf16
r=json.load(open('scan2.json'))
def tens(p,names):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n; out={}
        for k in names:
            a,b=h[k]['data_offsets']; f.seek(base+a); out[k]=np.frombuffer(f.read(b-a),dtype='<f4').reshape(h[k]['shape']).astype(np.float64)
    return h['__metadata__'],out
mids=sys.argv[1].split(','); every=int(sys.argv[2]) if len(sys.argv)>2 else 1
for mid in mids:
    c=sorted([x for x in r if x['model_id']==mid and x['step']],key=lambda x:(int(x['step']),x['path']))
    seen=set(); c2=[]
    for x in c:
        if x['step'] in seen: continue
        seen.add(x['step']); c2.append(x)
    c2=c2[::every]+([c2[-1]] if c2 and c2[-1] not in c2[::every] else [])
    for x in c2:
        md,t=tens(x['path'],['value.wdl_fc2.weight','value.wdl_fc2.bias','policy.conv.bias','policy.conv.weight'])
        W=t['value.wdl_fc2.weight'];b=t['value.wdl_fc2.bias'];m=W.mean(0)
        pb=t['policy.conv.bias']; PW=t['policy.conv.weight'].reshape(76,-1); pm=PW.mean(0)
        print(f"{mid[-4:]} step {int(md['training_step']):>8} | v.fc2 bias {np.array2string(b,precision=3)} mean {b.mean():+.4f} | meanrow {np.linalg.norm(m):7.3f} resid {np.array2string(np.linalg.norm(W-m,axis=1),precision=3)} | pol bias mean {pb.mean():+.4f} std {pb.std():.3f} | pol meanrow {np.linalg.norm(pm):.3f} resid med {np.median(np.linalg.norm(PW-pm,axis=1)):.3f}")
