import numpy as np, itertools, math
def bf16(x):
    a=np.asarray(x,dtype=np.float32).copy(); u=a.view(np.uint32)
    r=((u>>16)&1)+0x7FFF; u2=((u+r)>>16)<<16
    return u2.astype(np.uint32).view(np.float32)
obs=[(0.496094,0.009094,0.496094),(0.964844,0.017700,0.017700),(1.000000,0.000336,0.000336),(0.333984,0.333984,0.333984),(0.980469,0.000330,0.017944),(0.496094,0.496094,0.009094)]
print("bf16 representability (exact bf16 value nearest to printed, rel err):")
for t in obs:
    for v in t:
        b=float(bf16(v)); print(f"  {v:.6f} -> bf16 {b:.9f}  |diff|={abs(b-v):.2e}")
# candidate logit offsets in multiples of 4
def sm_f32(z): z=np.array(z,dtype=np.float64); e=np.exp(z-z.max()); return e/e.sum()
def sm_bf16(z):
    # all-bf16 graph: z bf16; subtract max, exp -> bf16, sum -> bf16, divide -> bf16
    z=bf16(np.array(z,np.float32)); m=z.max(); e=bf16(np.exp(bf16(z-m))); s=bf16(e.sum()); return bf16(e/s)
print()
for t in obs:
    best=[]
    for d in itertools.product(range(0,13,1),repeat=3):
        if min(d)!=0: continue
        z=[-x for x in d]
        for name,f in (("f32->bf16",lambda z: bf16(sm_f32(z).astype(np.float32))),("allbf16",sm_bf16)):
            p=f(z); err=max(abs(float(p[i])-t[i]) for i in range(3))
            if err<2e-6: best.append((name,tuple(-x for x in d),[f"{float(x):.9f}" for x in p]))
    print(t, "matches:", best[:6])
