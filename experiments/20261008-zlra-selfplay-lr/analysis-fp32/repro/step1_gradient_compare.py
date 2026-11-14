import json,struct,sys
import numpy as np
def load(p):
    f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
    def get(k):
        m=h[k]; a,b=m['data_offsets']; f.seek(base+a); return np.frombuffer(f.read(b-a),dtype=np.float32).astype(np.float64)
    keys=[k for k in h if k.startswith('opt.')]
    return get,keys
S=sys.argv[1]
src,keys=load('20261011-zlra-elitecyc-latest-step384964.safetensors'); a,_=load(f'{S}/bf16-1step.safetensors'); b,_=load(f'{S}/fp32-1step.safetensors')
for k in keys:
    v0=src(k); g16=a(k)-0.9*v0; g32=b(k)-0.9*v0
    d=g32-g16
    c=(g16@g32)/np.linalg.norm(g16)/np.linalg.norm(g32)
    print('%-38s |g16|=%.4e |g32|=%.4e cos=%.4f |diff|/|g16|=%.3f'%(k[4:-9],np.linalg.norm(g16),np.linalg.norm(g32),c,np.linalg.norm(d)/np.linalg.norm(g16)))
