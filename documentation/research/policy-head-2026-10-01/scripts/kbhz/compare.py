from load import *
import numpy as np
fcshape={"se_attenuate.fc1.weight":(128,32),"se_attenuate.fc2.weight":(32,128),"value.fc1.weight":(64,64),"value.wdl_fc2.weight":(64,3)}
def legacy_to_st(n,x):
    for k,s in fcshape.items():
        if n.endswith(k): return x.reshape(s).T.ravel()
    return x.ravel()
m,a=read_dcm('Sessions/last-before-big-changeup-20260524-175426-20260514-2-Ko63-manual.dcmsession/trainer.dcmmodel')
m2,b=read_st('Sessions/20260611-212501-20260514-2-Ko63-manual.dcmsession/trainer.safetensors')
lr=1e-3  # base lr during 494927->532369 (batch 4096 => sqrt scale 1)
print(f"{'tensor':40s} {'n':>7s} {'||W||@495k':>10s} {'||W||@532k':>10s} {'relchg':>8s} {'cos':>8s} {'|v|/|W|@495k':>12s} {'|v|/|W|@532k':>12s} {'rms':>8s}")
rows=[]
for n in block_names():
    x=legacy_to_st(n,a[n]); y=b[n].ravel()
    vx=a.get("opt."+n+".velocity"); vy=b.get("opt."+n+".velocity")
    vx=legacy_to_st(n,vx) if vx is not None else None
    nx,ny=np.linalg.norm(x),np.linalg.norm(y)
    rel=np.linalg.norm(y-x)/nx; cos=np.dot(x,y)/nx/ny
    r1=np.linalg.norm(vx)/nx; r2=np.linalg.norm(vy.ravel())/ny
    rows.append((n,x.size,nx,ny,rel,cos,r1,r2,ny/np.sqrt(x.size)))
    if n.endswith("weight") and ("conv" in n or "fc" in n) or n.startswith("policy"):
        print(f"{n:40s} {x.size:7d} {nx:10.4f} {ny:10.4f} {rel:8.4f} {cos:8.5f} {r1:12.3e} {r2:12.3e} {ny/np.sqrt(x.size):8.4f}")
