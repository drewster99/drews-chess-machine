import json,struct,sys
import numpy as np
def load(p):
    f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
    def get(k):
        m=h[k]; a,b=m['data_offsets']; f.seek(base+a); buf=f.read(b-a)
        assert m['dtype']=='F32', m['dtype']
        return np.frombuffer(buf,dtype=np.float32).reshape(m['shape'])
    return h,get
names=['stem.conv','blocks.0.conv1','blocks.0.conv2','blocks.1.conv1','blocks.1.conv2','blocks.2.conv1','blocks.2.conv2','policy.pre_conv','value.conv']
for p in sys.argv[1:]:
    h,get=load(p)
    out=[]
    for nm in names:
        w=get(nm+'.weight').astype(np.float64); v=get('opt.'+nm+'.weight.velocity').astype(np.float64)
        w2=w.reshape(w.shape[0],-1); v2=v.reshape(w.shape[0],-1)
        cos=(w2*v2).sum(1)/np.linalg.norm(w2,axis=1)/np.linalg.norm(v2,axis=1)
        out.append('%s |w|=%.2f |v|=%.3f cos(v,w)ch mean=%+.3f maxabs=%.3f'%(nm,np.linalg.norm(w),np.linalg.norm(v),cos.mean(),np.abs(cos).max()))
    print('==',p.split('/')[-1]); print('\n'.join(out))
