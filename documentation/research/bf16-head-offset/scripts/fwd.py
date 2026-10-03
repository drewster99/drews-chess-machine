import numpy as np, json, struct, sys
def load(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); data=f.read()
    md=h.pop('__metadata__'); T={}
    for k,v in h.items():
        a,b=v['data_offsets']; assert v['dtype']=='F32'
        T[k]=np.frombuffer(data[a:b],dtype='<f4').reshape(v['shape']).astype(np.float64)
    return md,T
def bf16(x):
    a=np.asarray(x,dtype=np.float32).copy(); u=a.view(np.uint32)
    u2=((u+(((u>>16)&1)+0x7FFF))>>16)<<16
    return u2.astype(np.uint32).view(np.float32).astype(np.float64)
# ---------------- board -----------------
PT={'p':0,'n':1,'b':2,'r':3,'q':4,'k':5}
def start():
    b=[None]*64; back='rnbqkbnr'
    for c in range(8):
        b[c]=('b',back[c]); b[8+c]=('b','p'); b[48+c]=('w','p'); b[56+c]=('w',back[c])
    return dict(b=b,stm='w',cr=set('KQkq'),ep=None,hm=0)
def sq(s): return (8-int(s[1]))*8+(ord(s[0])-97)
def apply(st,u):
    b=st['b'][:]; f=sq(u[:2]); t=sq(u[2:4]); pc=b[f]; assert pc and pc[0]==st['stm'],(u,pc)
    cap=b[t] is not None; ep=None; cr=set(st['cr'])
    if pc[1]=='p' and st['ep'] is not None and t==st['ep'] and not cap:
        b[t+8 if pc[0]=='w' else t-8]=None; cap=True
    if pc[1]=='k' and abs((t%8)-(f%8))==2:
        if t%8==6: b[f+1]=b[f+3]; b[f+3]=None
        else: b[f-1]=b[f-4]; b[f-4]=None
    b[t]=pc; b[f]=None
    if len(u)==5: b[t]=(pc[0],u[4])
    if pc[1]=='p' and abs(t-f)==16: ep=(t+f)//2
    if pc[1]=='k': cr-= set('KQ') if pc[0]=='w' else set('kq')
    for s,r in ((63,'K'),(56,'Q'),(7,'k'),(0,'q')):
        if f==s or t==s: cr.discard(r)
    hm=0 if (pc[1]=='p' or cap) else st['hm']+1
    return dict(b=b,stm='b' if st['stm']=='w' else 'w',cr=cr,ep=ep,hm=hm)
def key(st): return (tuple(st['b']),st['stm'],frozenset(st['cr']),st['ep'])
def game_states(ucis):
    st=start(); counts={key(st):1}; window=[]; out=[(st,0,0)]
    for u in ucis:
        pk=key(st); st=apply(st,u)
        if st['hm']==0: counts={}; window=[]
        else:
            window.insert(0,pk); window=window[:10]
        k=key(st); counts[k]=counts.get(k,0)+1
        mask=sum(1<<i for i,w in enumerate(window) if w==k)
        out.append((st,min(counts[k]-1,2),mask))
    return out
def encode(st,rep,mask):
    x=np.zeros((30,8,8)); me=st['stm']; flip= me=='b'
    for r in range(8):
        sr=7-r if flip else r
        for c in range(8):
            p=st['b'][sr*8+c]
            if p: x[(0 if p[0]==me else 6)+PT[p[1]],r,c]=1
    cr=st['cr']
    mk,mq,ok,oq=(('k' in cr),('q' in cr),('K' in cr),('Q' in cr)) if flip else (('K' in cr),('Q' in cr),('k' in cr),('q' in cr))
    for i,v in enumerate((mk,mq,ok,oq)):
        if v: x[12+i]=1
    if st['ep'] is not None:
        er=st['ep']//8; er=7-er if flip else er; x[16,er,st['ep']%8]=1
    x[17]=min(st['hm'],99)/99.0
    x[18]=1.0 if rep>=1 else 0; x[19]=1.0 if rep>=2 else 0
    for i in range(10):
        if (mask>>i)&1: x[20+i]=1
    return x
# ---------------- net -----------------
def conv(x,w,q):
    # x [C,8,8], w [O,C,k,k], same padding
    O,C,k,_=w.shape; p=(k-1)//2
    xp=np.zeros((C,8+2*p,8+2*p)); xp[:,p:p+8,p:p+8]=x
    cols=np.empty((C,k,k,8,8))
    for i in range(k):
        for j in range(k): cols[:,i,j]=xp[:,i:i+8,j:j+8]
    y=np.tensordot(w,cols,axes=([1,2,3],[0,1,2]))
    return q(y)
def bn(x,T,n,q):
    g=T[n+'.weight'];b=T[n+'.bias'];m=T[n+'.running_mean'];v=T[n+'.running_var']
    return q((x-m[:,None,None])/np.sqrt(v[:,None,None]+1e-5)*g[:,None,None]+b[:,None,None])
def forward(T,arch,x,precision='f64',return_all=False):
    q=(lambda a:a) if precision=='f64' else bf16
    W=(lambda a:a) if precision=='f64' else bf16   # weights stored bf16 in-graph
    TT={k:W(v) for k,v in T.items() if not k.startswith('opt.')}
    x=q(x); acts={}
    h=conv(x,TT['stem.conv.weight'],q); h=bn(h,TT,'stem.bn',q)
    assert len(arch['block_groups'])==1 and arch['block_groups'][0]['activation_style']=='pre'
    g=arch['block_groups'][0]
    for i in range(g['count']):
        pre=f'blocks.{i}.'
        y=bn(h,TT,pre+'bn1',q); y=np.maximum(y,0)
        y=conv(y,TT[pre+'conv1.weight'],q); y=bn(y,TT,pre+'bn2',q); y=np.maximum(y,0)
        y=conv(y,TT[pre+'conv2.weight'],q)
        assert g['se_style']=='none'
        if g['use_rezero']:
            C=g['rezero_alpha_cap']; a=C*np.tanh(TT[pre+'rezero_alpha'][0]/C); y=q(y*q(a))
        h=q(h+y)
        if g.get('output_norm')=='layer_norm':
            mu=h.mean(0,keepdims=True); var=h.var(0,keepdims=True)
            h=q((h-mu)/np.sqrt(var+1e-5)*TT[pre+'res_ln.weight'][:,None,None]+TT[pre+'res_ln.bias'][:,None,None])
        acts[f'block{i}']=h
    h=bn(h,TT,'tower_final_bn',q); h=np.maximum(h,0); acts['tower']=h
    v=conv(h,TT['value.conv.weight'],q); v=bn(v,TT,'value.bn',q); v=np.maximum(v,0); acts['vconv']=v
    f=v.reshape(-1)
    f1=q(TT['value.fc1.weight']@f+TT['value.fc1.bias']); f1=np.maximum(f1,0); acts['fc1']=f1
    lg=q(TT['value.wdl_fc2.weight']@f1+TT['value.wdl_fc2.bias'])
    e=np.exp(lg-lg.max()); p=e/e.sum()
    if precision!='f64': p=bf16(p)
    return (lg,p,acts) if return_all else (lg,p)
