import pickle,json,sys,os,glob,struct,time,numpy as np, traceback
from fwd3 import *
P=pickle.load(open('posset.pkl','rb'))
X=np.stack([p['x'] for p in P]).astype(np.float64)
corpus=np.array([p['src']=='corpus' for p in P]); labels=np.array([p['label'] for p in P]); targets=[p['target'] for p in P]
legal=[p['legal'] for p in P]
def lsm(z): z=z-z.max(); return z-np.log(np.exp(z).sum())
def vsoft(vl,q): e=np.exp(vl-vl.max(1,keepdims=True)); return q(e/e.sum(1,keepdims=True))
def value_metrics(vl64,vlq,vpq,q):
    p64=vsoft(vl64,lambda a:a)
    kl=np.sum(p64*(np.log(p64)-np.log(np.maximum(vpq,1e-30))),1)
    ce64=-np.log(p64[np.arange(len(P)),labels]); ceq=-np.log(np.maximum(vpq[np.arange(len(P)),labels],1e-30))
    ties=np.array([len(set(r))<3 for r in vlq])
    return dict(kl_mean=float(kl.mean()),kl_p90=float(np.percentile(kl,90)),kl_max=float(kl.max()),
        ce64=float(ce64[corpus].mean()),ceq=float(ceq[corpus].mean()),tie_frac=float(ties.mean()),
        argmax_change=float(np.mean(p64.argmax(1)!=vpq.argmax(1))),
        pD64_med=float(np.median(p64[:,1])), maxabs_dp=float(np.abs(p64-vpq).max()))
def policy_metrics(pl64,plq):
    kl=[];top1=[];tie2=[];tie5=[];ce64=[];ceq=[];tv=[]
    for i in range(len(P)):
        li=legal[i]; a=pl64[i][li]; b=plq[i][li].astype(np.float32).astype(np.float64)
        la=lsm(a); lb=lsm(b); pa=np.exp(la); pb=np.exp(lb)
        kl.append(float(np.sum(pa*(la-lb)))); tv.append(0.5*np.abs(pa-pb).sum())
        ob=np.sort(b)[::-1]; ia=int(np.argmax(a))
        top1.append(b[ia]<b.max())   # f64 best move no longer (co-)best in bf16
        tie2.append(len(li)>1 and ob[0]==ob[1]); tie5.append(len(set(ob[:5]))<min(5,len(li)))
        t=li.tolist().index(targets[i]); ce64.append(-la[t]); ceq.append(-lb[t])
    ce64=np.array(ce64);ceq=np.array(ceq)
    return dict(kl_mean=float(np.mean(kl)),kl_p90=float(np.percentile(kl,90)),kl_max=float(np.max(kl)),tv_mean=float(np.mean(tv)),
        top1_lost=float(np.mean(top1)),tie_top2=float(np.mean(tie2)),tie_top5=float(np.mean(tie5)),
        ce64=float(ce64[corpus].mean()),ceq=float(ceq[corpus].mean()))
def policy_mag(pl64,feat,TT,arch):
    g=pl64.mean(1); lm=np.array([pl64[i][legal[i]].mean() for i in range(len(P))]); ls=np.array([pl64[i][legal[i]].std() for i in range(len(P))])
    lmax=np.array([pl64[i][legal[i]].max() for i in range(len(P))]); amax=np.abs(pl64).max(1)
    d=dict(glob_mean_med=float(np.median(g)),glob_mean_absmean=float(np.abs(g).mean()),legal_mean_med=float(np.median(lm)),legal_std_med=float(np.median(ls)),
           legal_max_med=float(np.median(lmax)),abs_max_med=float(np.median(amax)),abs_mean=float(np.abs(pl64).mean()),
           glob_mean_std_across_pos=float(g.std()))
    if arch['policy_head_style']!='fc_bottleneck':
        L=pl64.reshape(len(P),76,64); sq=L.mean(1)   # per-square channel-mean
        d['sqcomp_std_within_pos_med']=float(np.median(sq.std(1)))
        d['resid_std_within_pos_med']=float(np.median((L-sq[:,None,:]).std((1,2))))
    return d
def struct_value(T):
    W=T['value.wdl_fc2.weight']; b=T['value.wdl_fc2.bias']; m=W.mean(0); R=W-m
    cos=[float(W[i]@W[j]/np.linalg.norm(W[i])/np.linalg.norm(W[j])) for i,j in ((0,1),(0,2),(1,2))]
    return dict(mean_row_norm=float(np.linalg.norm(m)),resid_norms=[float(x) for x in np.linalg.norm(R,axis=1)],row_cos=cos,bias=[float(x) for x in b],bias_mean=float(b.mean()),
                fc1_bias_mean=float(T['value.fc1.bias'].mean()))
def struct_policy(T,arch):
    if arch['policy_head_style']=='fc_bottleneck': W=T['policy.fc.weight']; b=T['policy.fc.bias']
    else: W=T['policy.conv.weight'].reshape(76,-1); b=T['policy.conv.bias']
    m=W.mean(0); R=W-m
    return dict(mean_row_norm=float(np.linalg.norm(m)),resid_norm_med=float(np.median(np.linalg.norm(R,axis=1))),resid_norm_max=float(np.linalg.norm(R,axis=1).max()),
                row_cos_mean_med=float(np.median(W@m/np.linalg.norm(W,axis=1)/np.linalg.norm(m))),bias_mean=float(b.mean()),bias_std=float(b.std()),bias_absmax=float(np.abs(b).max()))
def analyze(path,tag):
    md,T=load(path); arch=norm_arch(md['architecture']); nat=DT[arch['compute_data_type']]
    bfexact=float(np.mean(np.concatenate([(bf16(v)==v).ravel() for k,v in T.items() if not k.startswith('opt.')])))
    r=dict(tag=tag,path=path,model_id=md['model_id'],step=md.get('training_step'),native=nat,bf16_exact_weight_frac=bfexact,
           policy_style=arch['policy_head_style'],value_style=arch['value_head_style'])
    r['vstruct']=struct_value(T); r['pstruct']=struct_policy(T,arch)
    TT=prep(T,'f64'); t=time.time()
    o64=forward_batched(TT,arch,X,'f64')
    r['value_logit']=dict(mean_of_class_mean=float(o64['vl'].mean()),absmean=float(np.abs(o64['vl']).mean()),maxabs=float(np.abs(o64['vl']).max()),
        spread_med=float(np.median(o64['vl'].max(1)-o64['vl'].min(1))),shared_med=float(np.median(o64['vl'].mean(1))))
    r['policy_mag']=policy_mag(o64['pl'],o64['pfeat'],TT,arch)
    # emulations: 'out' = exact internals w/ weights rounded to compute dtype, outputs rounded (best fit to observed bot outputs);
    # 'perop' = every op rounded (pessimistic). For fp32 models also report the hypothetical bf16.
    emus={}
    for dt in sorted({nat,'bf16'}):
        q=Q[dt]; Tq=prep(T,dt)
        oo=forward_batched(Tq,arch,X,'f64'); vlq=q(oo['vl']); emus[f'{dt}-out']=(vlq,vsoft(vlq,q),q(oo['pl']),q)
        if dt=='bf16':
            po=forward_batched(Tq,arch,X,'bf16'); emus['bf16-perop']=(po['vl'],po['vp'],po['pl'],q)
        # value recentered (exact): subtract mean row and mean bias of fc2, then output-round
        W=T['value.wdl_fc2.weight']; b=T['value.wdl_fc2.bias']
        vlr=q(q(oo['f1']@(q(W-W.mean(0))).T)+q(b-b.mean())) if False else q(oo['vl']-oo['vl'].mean(1,keepdims=True)*0 - (oo['f1']@W.mean(0)+b.mean())[:,None])
        emus[f'{dt}-out-vrecentered']=(vlr,vsoft(vlr,q),None,q)
        # policy: exact global recentering = subtract mean bias; oracle = subtract per-position legal mean before rounding
        emus[f'{dt}-out-pbiascentered']=(None,None,q(oo['pl']-T['policy.conv.bias' if arch['policy_head_style']!='fc_bottleneck' else 'policy.fc.bias'].mean()),q)
        lm=np.array([oo['pl'][i][legal[i]].mean() for i in range(len(P))])
        emus[f'{dt}-out-poracle']=(None,None,q(oo['pl']-lm[:,None]),q)
    r['emu']={}
    for k,(vlq,vpq,plq,q) in emus.items():
        e={}
        if vlq is not None: e['value']=value_metrics(o64['vl'],vlq,vpq,q)
        if plq is not None: e['policy']=policy_metrics(o64['pl'],plq)
        r['emu'][k]=e
    r['secs']=time.time()-t
    return r
if __name__=='__main__':
    jobs=json.load(open(sys.argv[1])); out=sys.argv[2]
    done={}
    if os.path.exists(out):
        for l in open(out): d=json.loads(l); done[d['path']]=1
    with open(out,'a') as f:
        for tag,path in jobs:
            if path in done: continue
            try: r=analyze(path,tag)
            except Exception as ex: r=dict(tag=tag,path=path,error=repr(ex),tb=traceback.format_exc()[-600:])
            f.write(json.dumps(r)+'\n'); f.flush(); print(tag,r.get('model_id'),r.get('secs',r.get('error')),flush=True)
