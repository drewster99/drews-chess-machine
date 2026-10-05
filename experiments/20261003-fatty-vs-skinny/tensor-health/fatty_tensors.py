import json, struct, os, math, sys
import numpy as np
M=os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
def load(f):
    b=open(f,'rb'); n=struct.unpack('<Q',b.read(8))[0]; h=json.loads(b.read(n)); md=h.pop('__metadata__'); base=8+n
    raw=open(f,'rb').read()
    out={}
    for k,v in h.items():
        s,e=v['data_offsets']; out[k]=np.frombuffer(raw[base+s:base+e],dtype=np.float32).reshape(v['shape'])
    return md,out
Phi=lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
for label,fn in [("fatty","20261003-fatty216-b2275-replay-step33000"),("slim-neck","20261003-fatty224s3-b2275-replay-step33000")]:
    md,t=load(f"{M}/{fn}.safetensors")
    print(f"\n=== {label}  model_id={md['model_id']}  step={md.get('training_step')}")
    print(f"{'tensor':32s} {'shape':18s} {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s} {'|max|':>8s} {'~0 frac':>8s}")
    for k in sorted(t):
        a=t[k]
        print(f"{k:32s} {str(list(a.shape)):18s} {a.mean():9.4f} {a.std():8.4f} {a.min():9.4f} {a.max():9.4f} {np.abs(a).max():8.4f} {(np.abs(a)<1e-6).mean():8.4f}")
    print("-- conv/FC output-unit weight norms (dead unit = norm ~0)")
    for k in sorted(t):
        a=t[k]
        if k.endswith('weight') and a.ndim in (2,4) and 'bn' not in k and 'ln' not in k:
            norms=np.sqrt((a.reshape(a.shape[0],-1)**2).sum(1))
            print(f"  {k:30s} units {a.shape[0]:4d}  min {norms.min():.4f}  median {np.median(norms):.4f}  max {norms.max():.4f}  min/median {norms.min()/np.median(norms):.3f}")
    print("-- BN sites feeding ReLU: P(channel active) = Phi(beta/|gamma|)")
    for site in ["blocks.0.bn1","blocks.0.bn2","tower_final_bn","policy.pre_bn","value.bn"]:
        g=t[site+".weight"]; b=t[site+".bias"]; r=b/np.abs(g)
        p=np.array([Phi(x) for x in r])
        print(f"  {site:16s} ch {len(g):4d}  |gamma| min {np.abs(g).min():.3f} med {np.median(np.abs(g)):.3f} max {np.abs(g).max():.3f}  P(active) min {p.min():.3f} med {np.median(p):.3f} max {p.max():.3f}  (<5%: {(p<0.05).sum()}, >95%: {(p>0.95).sum()})")

print("\n\n######## zero-velocity patterns")
for label,fn in [("fatty","20261003-fatty216-b2275-replay-step33000"),("slim-neck","20261003-fatty224s3-b2275-replay-step33000")]:
    md,t=load(f"{M}/{fn}.safetensors")
    v=t["opt.stem.conv.weight.velocity"].reshape(t["stem.conv.weight"].shape)
    w=t["stem.conv.weight"]
    zp=[(p, (v[:,p]==0).mean(), np.sqrt((w[:,p]**2).sum())) for p in range(v.shape[1])]
    print(f"\n{label} stem: input planes with exact-zero velocity fraction (and weight norm):")
    print("  "+"  ".join(f"p{p}:{z:.2f}/{n:.2f}" for p,z,n in zp))
    # Optimizer velocity is stored flat in the trainer's native [in, out] layout (only weights are
    # written transposed to [out, in]); LayerHealth.hiddenUnitVelocityHealth reads it the same way.
    vf=t["opt.value.fc1.weight.velocity"].reshape(1024,128)  # [in, out]
    inzero=(vf==0).all(1); print(f"{label} value.fc1: input features whose velocity row is all zero: {inzero.sum()} of 1024 ->", end=" ")
    idx=np.where(inzero)[0]; ch=idx//64; sq=idx%64
    print("by value.conv channel:", {int(c):int((ch==c).sum()) for c in np.unique(ch)}, " squares(rank,file) sample:", [(int(s//8),int(s%8)) for s in sq[:12]])
    # Conv velocity is stored in the weight's own OIHW order (the stem check above relies on it).
    pv=t["opt.policy.conv.weight.velocity"].reshape(76,128)
    rz=(pv==0).all(1); cz=(pv==0).all(0)
    print(f"{label} policy.conv: output channels all-zero velocity: {np.where(rz)[0].tolist()}  input channels all-zero: {np.where(cz)[0].tolist()}  total zero entries {(pv==0).sum()}")
    print(f"{label} policy.conv.bias velocity zeros at channels: {np.where(t['opt.policy.conv.bias.velocity']==0)[0].tolist()}")
