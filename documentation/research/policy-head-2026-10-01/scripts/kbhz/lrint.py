import re,sys,glob,os
L=os.path.expanduser("~/Library/Logs/DrewsChessMachine/")
def run(tag, files, final):
    m={}
    for f in files:
        for line in open(L+f,errors='replace'):
            if '[STATS]' not in line or tag not in line: continue
            s=re.search(r' steps=(\d+)',line); lr=re.search(r' lr=([0-9.e+-]+)',line)
            mu=re.search(r' μ=([0-9.]+)',line); d=re.search(r'decay=([0-9.e+-]+)',line)
            if not (s and lr and mu and d): continue
            wu=re.search(r'warmup\((\d+)/(\d+)\)',line)
            l=float(lr.group(1))*(int(wu.group(1))/int(wu.group(2)) if wu else 1)
            m[int(s.group(1))]=(l,float(mu.group(1)),float(d.group(1)))
    ks=sorted(m); eff=0; dec=0; raw=0; seg={}
    for i,k in enumerate(ks):
        if k>final: break
        nxt=min(ks[i+1] if i+1<len(ks) else final, final)
        n=nxt-k; l,mu,d=m[k]
        raw+=l*n; eff+=l/(1-mu)*n; dec+=l*d*n
        key=(l if not 0<l<1.5e-4 else 1.5e-4,mu,d); seg[key]=seg.get(key,0)+n
    print(tag,"final",final,"sum lr=%.1f  sum lr/(1-mu)=%.1f  sum lr*decay=%.4f (decay shrink factor e^-x=%.3f)"%(raw,eff,dec,2.718281828**-dec))
    for k,v in sorted(seg.items(),key=lambda kv:-kv[1])[:8]: print("   lr=%.2e mu=%.2f decay=%.0e steps=%d"%(k[0],k[1],k[2],v))
kb=sorted(os.path.basename(p) for p in glob.glob(L+"dcm_log_2026051*.txt")+glob.glob(L+"dcm_log_2026052*.txt")+glob.glob(L+"dcm_log_20260610-*.txt"))
run("KbHZ", [f for f in kb if 'KbHZ' in open(L+f,errors='replace').read(200000) or True], 532369)
run("sMe9", [f for f in kb if f>='dcm_log_20260524-2051' and f<'dcm_log_20260529-154054.txt'], 373416)
