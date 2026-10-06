"""Batched faithful forward pass of ChessNetwork (all block/head styles used by
saved models), with per-op rounding emulation for bf16 / fp16 / fp32 compute."""
import numpy as np, json, struct, math
from archnorm import norm_arch_md, require_architecture_of, site_activations_md
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
Q={'f64':lambda a:a,'bf16':bf16,'f16':lambda a:np.asarray(a,dtype=np.float16).astype(np.float64),'f32':lambda a:np.asarray(a,dtype=np.float32).astype(np.float64)}
DT={'bfloat16':'bf16','float16':'f16','float32':'f32'}
_erf=np.vectorize(math.erf)
def act(x,fn,q):
    if fn=='relu': return np.maximum(x,0)
    if fn=='silu': return q(x*q(1/(1+np.exp(-x))))
    if fn=='gelu': return q(0.5*x*(1+_erf(x/math.sqrt(2))))
    # ActivationFunction.leakyReLUNegativeSlope in the Swift source.
    if fn=='leaky_relu': return np.where(x>=0,x,q(0.01*x))
    raise ValueError(fn)
def conv(x,w,q,qm=None):
    N,C=x.shape[:2]; O,_,k,_=w.shape; p=(k-1)//2
    xp=np.zeros((N,C,8+2*p,8+2*p)); xp[:,:,p:p+8,p:p+8]=x
    y=np.zeros((N,8,8,O))
    for i in range(k):
        for j in range(k):
            y+=np.tensordot(xp[:,:,i:i+8,j:j+8],w[:,:,i,j],axes=([1],[1]))
    return (qm or q)(y.transpose(0,3,1,2))
def bn(x,T,n,q):
    g=T[n+'.weight'];b=T[n+'.bias'];m=T[n+'.running_mean'];v=T[n+'.running_var']
    s=lambda a:a[None,:,None,None]
    return q((x-s(m))/np.sqrt(s(v)+1e-5)*s(g)+s(b))
def ln(h,T,n,q):
    mu=q(h.mean(1,keepdims=True)); var=q(((h-mu)**2).mean(1,keepdims=True))
    return q((h-mu)/np.sqrt(var+1e-5)*T[n+'.weight'][None,:,None,None]+T[n+'.bias'][None,:,None,None])
def fc(x,W,b,q,qm=None):  # W stored [out,in]
    return q((qm or q)(x@W.T)+b)
def se(z,T,pre,g,q,qm=None):
    base=pre+('se_attenuate.' if g['se_style']=='attenuate_only' else 'se_scalebias.')
    s=q(z.mean((2,3)))
    # SE FC1 has its own activation (`se_activation`); norm_arch resolves it
    # for files that predate the field.
    s=act(fc(s,T[base+'fc1.weight'],T[base+'fc1.bias'],q,qm),g['se_activation'],q)
    s=fc(s,T[base+'fc2.weight'],T[base+'fc2.bias'],q,qm)
    C=z.shape[1]; sig=lambda a:q(1/(1+np.exp(-a)))
    if g['se_style']=='attenuate_only': return q(z*sig(s)[:,:,None,None])
    return q(q(z*sig(s[:,:C])[:,:,None,None])+s[:,C:][:,:,None,None])
def prep(T,precision):
    q=Q[precision.split(':')[0]]; return {k:q(v) for k,v in T.items() if not k.startswith('opt.')}
def forward(TT,arch,x,precision,*,md):
    """md: the file's safetensors __metadata__; `arch` must be norm_arch_md(md). The
    architecture-level site activations are read from md (site_activations_md)."""
    require_architecture_of(md,arch,'fwd3.forward')
    sites=site_activations_md(md)
    if ':' in precision:
        base,mode=precision.split(':'); qb=Q[base]; ident=lambda a:a
        return _forward(TT,arch,sites,x,ident,qb,qb)
    return _forward(TT,arch,sites,x,Q[precision],Q[precision],Q[precision])
def _forward(TT,arch,sites,x,q,qm,qout):
    """x [N,30,8,8]. Returns value logits [N,3], value probs [N,3], policy logits [N,4864],
    plus intermediates (fc1 hidden, policy pre-projection features)."""
    a=arch
    assert a.get('feature_skip_source','none')=='none' or not (a.get('feature_skip_to_policy_head') or a.get('feature_skip_to_value_head') or a.get('feature_skip_to_final_block')), 'feature skip unsupported'
    assert a['value_head_style']=='wdl_softmax'
    groups=a['block_groups']
    h=conv(q(x),TT['stem.conv.weight'],q,qm); h=bn(h,TT,'stem.bn',q)
    if groups[0]['activation_style']=='post': h=act(h,sites['stem_activation'],q)
    i=0; inC=h.shape[1]
    for g in groups:
        fn=g['activation_function']
        for _ in range(g['count']):
            pre=f'blocks.{i}.'
            if g['activation_style']=='pre':
                y=act(bn(h,TT,pre+'bn1',q),fn,q); a1=y
                y=conv(y,TT[pre+'conv1.weight'],q,qm); y=act(bn(y,TT,pre+'bn2',q),fn,q)
                z=conv(y,TT[pre+'conv2.weight'],q,qm)
            else:
                a1=h
                y=conv(h,TT[pre+'conv1.weight'],q,qm); y=act(bn(y,TT,pre+'bn1',q),fn,q)
                y=conv(y,TT[pre+'conv2.weight'],q,qm); z=bn(y,TT,pre+'bn2',q)
            if g['se_style']!='none': z=se(z,TT,pre,g,q,qm)
            if g['use_rezero']:
                C=g['rezero_alpha_cap']; al=C*math.tanh(float(TT[pre+'rezero_alpha'].reshape(-1)[0])/C); z=q(z*q(al))
            skip=h if inC==g['channels'] else conv(a1,TT[pre+'skip_proj.weight'],q,qm)
            h=q(skip+z)
            if g['skip_merge']=='activation_gated': h=act(h,fn,q)
            if g.get('output_norm')=='layer_norm': h=ln(h,TT,pre+'res_ln',q)
            inC=g['channels']; i+=1
    if groups[-1]['activation_style']=='pre':
        h=act(bn(h,TT,'tower_final_bn',q),sites['tower_end_activation'],q)
    N=x.shape[0]
    # policy
    ps=a['policy_head_style']
    if ps=='simple_conv':
        feat=h
        pl=qout(conv(h,TT['policy.conv.weight'],q,qm)+TT['policy.conv.bias'][None,:,None,None]).reshape(N,-1)
    else:
        feat=act(bn(conv(h,TT['policy.pre_conv.weight'],q,qm),TT,'policy.pre_bn',q),sites['policy_head_activation'],q)
        if ps=='intermediate_conv':
            pl=qout(conv(feat,TT['policy.conv.weight'],q,qm)+TT['policy.conv.bias'][None,:,None,None]).reshape(N,-1)
        else:
            pl=qout(fc(feat.reshape(N,-1),TT['policy.fc.weight'],TT['policy.fc.bias'],q,qm))
    # value
    v=act(bn(conv(h,TT['value.conv.weight'],q,qm),TT,'value.bn',q),sites['value_head_conv_activation'],q)
    f1=act(fc(v.reshape(N,-1),TT['value.fc1.weight'],TT['value.fc1.bias'],q,qm),sites['value_head_fc1_hidden_activation'],q)
    vl=qout(fc(f1,TT['value.wdl_fc2.weight'],TT['value.wdl_fc2.bias'],q,qm))
    e=np.exp(vl-vl.max(1,keepdims=True)); vp=qout(e/e.sum(1,keepdims=True))
    return dict(vl=vl,vp=vp,pl=pl,f1=f1,pfeat=feat)
def forward_batched(TT,arch,X,precision,bs=128,*,md):
    outs=[forward(TT,arch,X[i:i+bs],precision,md=md) for i in range(0,len(X),bs)]
    return {k:np.concatenate([o[k] for o in outs]) for k in outs[0]}
