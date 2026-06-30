import re, json, os, struct, glob, io, base64, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
DOC="/Users/andrew/Documents/drews-chess-machine/documentation/v5-layernorm-output.md"
LS=os.path.join(HERE,"loop_state.txt"); M=os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
OUT="/Users/andrew/Documents/drews-chess-machine/documentation/dashboards/dcm_dashboard.html"
# v5 lineage = one continuous run from random init; warm-start param changes at these cumulative steps:
OFF={'wd1e-4':0,'wd5e-4':45441,'m0.93':60901}   # wd1e-4 stopped @45441, wd5e-4 @15460 -> 60901
TRANS=[(45441,'→ wd5e-4 (wd 1e-4→5e-4)'),(60901,'→ m0.93 (wd 2.5e-4, μ0.93)')]
def ff(x):
    x=x.strip(); return None if x in ('','—','-') else float(x)
sub={}  # sub-run -> rows (own step)
for line in open(DOC):
    m=re.match(r'\|\s*(\*\*)?(wd1e-4|wd5e-4|m0\.93)(\*\*)?\s*\|(.*)\|\s*$', line)
    if not m: continue
    name=m.group(2); c=[x.strip().replace('**','') for x in m.group(4).split('|')]
    if len(c)<10: continue
    pl=c[9].split('/')
    sub.setdefault(name,[]).append(dict(step=int(float(c[0])),pElo=ff(c[1]),nll=ff(c[2]),pLoss=ff(c[3]),vLoss=ff(c[4]),
        legalMass=ff(c[5]),bn1Mean=ff(c[6]),gNorm=ff(c[7]),sae2=ff(c[8]),
        plogit_mean=ff(pl[0]),plogit_peak=(ff(pl[1]) if len(pl)>1 else None)))
# combined v5
v5=[]
for s in ['wd1e-4','wd5e-4','m0.93']:
    for r in sub.get(s,[]): v5.append({**r,'step':r['step']+OFF[s]})
v5.sort(key=lambda r:r['step'])
runs={'v5':v5}
def kv(line,key):
    m=re.search(rf'{re.escape(key)}=([0-9.]+)',line); return float(m.group(1)) if m else None
for line in open(LS):
    m=re.match(r'#\s*(MINI|COXW|YKKK|T97X)\s+(\d+)\b[^:\n]*:',line)
    if not m: continue
    name={'MINI':'mini2b','COXW':'coxw','YKKK':'ykkk','T97X':'t97x'}[m.group(1)]; pl=re.search(r'pLogitAbsMax=([0-9.]+)/([0-9.]+)',line)
    runs.setdefault(name,[]).append(dict(step=int(m.group(2)),pElo=kv(line,'pElo'),nll=kv(line,'nll'),pLoss=kv(line,'pLoss'),
        vLoss=kv(line,'vLoss'),legalMass=kv(line,'lm'),bn1Mean=kv(line,'bn1Mean'),gNorm=kv(line,'gNorm'),sae2=kv(line,'Σαeff²'),
        plogit_mean=float(pl.group(1)) if pl else None,plogit_peak=float(pl.group(2)) if pl else None))
for k in runs: runs[k].sort(key=lambda r:r['step'])
caps={'v5':1.0,'mini2b':0.5,'coxw':1.0,'ykkk':1.0,'t97x':1.0}  # t97x has no ReZero → no Σαeff² data; cap unused
# --- incremental stable-rank cache (keyed by sub-run / fresh-run) ---
pat={'wd1e-4':"20260628-v5_5block_7x7_lnout-step*-frozen.safetensors",'wd5e-4':"20260628-v5_5block_7x7_lnout-wd5e4-step*-frozen.safetensors",
 'm0.93':"20260628-v5_5block_7x7_lnout-wd2.5e4-m93-step*-frozen.safetensors",'mini2b':"20260629-mini2b-3MIV-step*-frozen.safetensors",'coxw':"20260629-mini1b-Coxw-step*-frozen.safetensors",'ykkk':"20260630-mini-YkKk-step*-frozen.safetensors",'t97x':"20260630-T97X-step*-frozen.safetensors"}
cf=os.path.join(HERE,"util_cache.json"); cache=json.load(open(cf)) if os.path.exists(cf) else {}
def srfrac(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n)); base=8+n; srs=[]
        for k,mm in hdr.items():
            if not(k.startswith('blocks.') and k.endswith(('conv1.weight','conv2.weight'))): continue
            s,e=mm['data_offsets']; f.seek(base+s); a=np.frombuffer(f.read(e-s),dtype='<f4').reshape(mm['shape'])
            W=a.reshape(a.shape[0],-1); sv=np.linalg.svd(W,compute_uv=False); s2=sv**2; srs.append(float(s2.sum()/s2[0])/min(W.shape))
        return float(np.mean(srs)) if srs else None
subutil={}; new=0
for run,p in pat.items():
    cache.setdefault(run,{}); pts=[]
    for fp in glob.glob(os.path.join(M,p)):
        mm=re.search(r'step(\d+)(?:-FINAL)?-frozen',fp)
        if not mm: continue
        st=mm.group(1)
        if st not in cache[run]: cache[run][st]=srfrac(fp); new+=1
        pts.append((int(st),cache[run][st]))
    subutil[run]=sorted(pts)
json.dump(cache,open(cf,'w'))
util={'v5':sorted([(s+OFF[sr],v) for sr in ['wd1e-4','wd5e-4','m0.93'] for (s,v) in subutil.get(sr,[])]),
      'mini2b':subutil.get('mini2b',[]),'coxw':subutil.get('coxw',[]),'ykkk':subutil.get('ykkk',[]),'t97x':subutil.get('t97x',[])}
# --- charts ---
ORDER=['v5','mini2b','coxw','ykkk','t97x']
COLOR={'v5':'#1f77b4','mini2b':'#ff7f0e','coxw':'#2ca02c','ykkk':'#d62728','t97x':'#9467bd'}
LAB={'v5':'v5 — 8.45M, 5-block (wd1e-4→wd5e-4→m0.93)','mini2b':'mini2b — 2.25M, 2-block','coxw':'coxw — 1.86M, 1-block','ykkk':'ykkk — 0.24M, 2-block (SE)','t97x':'t97x — 1.70M, 2-block 9×9 (SE, GELU, no-ReZero)'}
def png(fig):
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=125,bbox_inches="tight"); plt.close(fig); return base64.b64encode(b.getvalue()).decode()
def vlines(ax):
    for x,_ in TRANS: ax.axvline(x,color='#1f77b4',ls=':',lw=0.8,alpha=0.35)
def chart(key,title,ylab,logy=False,tf=None):
    fig,ax=plt.subplots(figsize=(9,4.2))
    for r in ORDER:
        xs=[];ys=[]
        for row in runs.get(r,[]):
            v=row.get(key)
            if v is None: continue
            ys.append(tf(r,v) if tf else v); xs.append(row['step'])
        if xs: ax.plot(xs,ys,'-o',ms=2.4,lw=1.0,color=COLOR[r],label=LAB[r])
    vlines(ax)
    if logy: ax.set_yscale('log')
    ax.set_xlabel("SGD steps since random init"); ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.22); ax.legend(fontsize=7.5)
    return png(fig)
def chart_util():
    fig,ax=plt.subplots(figsize=(9,4.2))
    for r in ORDER:
        pts=util.get(r,[])
        if pts: ax.plot([p[0] for p in pts],[p[1] for p in pts],'-o',ms=2.4,lw=1.0,color=COLOR[r],label=LAB[r])
    vlines(ax)
    ax.set_xlabel("SGD steps since random init"); ax.set_ylabel("mean conv stable-rank / 128"); ax.set_title("Block utilization — mean conv stable-rank fraction"); ax.grid(alpha=.22); ax.legend(fontsize=7.5)
    return png(fig)
charts=[("pElo",chart('pElo',"pElo vs step","pElo")),("nll",chart('nll',"nll vs step","nll")),
 ("pLoss",chart('pLoss',"pLoss vs step","pLoss")),("vLoss",chart('vLoss',"vLoss vs step","vLoss")),
 ("legalMass",chart('legalMass',"legalMass vs step","legalMass")),("gNorm",chart('gNorm',"gNorm vs step","gNorm")),
 ("bn1Mean",chart('bn1Mean',"bn1Mean vs step (log y)","bn1Mean",logy=True)),
 ("pLogit",chart('plogit_mean',"pLogitAbsMax mean vs step","pLogitAbsMax mean")),
 ("sae2",chart('sae2',"Σαeff² / cap vs step","Σαeff²/cap",tf=lambda r,v:v/caps[r])),
 ("util",chart_util())]
def table(r):
    rows=runs.get(r,[])
    def c(v,fmt="{:.3f}"): return "" if v is None else fmt.format(v)
    def pc(x):
        if x['plogit_mean'] is None: return ""
        return "{:.2f}/{}".format(x['plogit_mean'],("{:.3f}".format(x['plogit_peak']) if x['plogit_peak'] is not None else "—"))
    th="<tr><th>step</th><th>pElo</th><th>nll</th><th>pLoss</th><th>vLoss</th><th>legalMass</th><th>bn1Mean</th><th>gNorm</th><th>Σαeff²</th><th>pLogit μ/peak</th></tr>"
    body="".join("<tr>"+"".join("<td>%s</td>"%v for v in [x['step'],c(x['pElo'],"{:.1f}"),c(x['nll']),c(x['pLoss'],"{:.2f}"),c(x['vLoss'],"{:.2f}"),c(x['legalMass'],"{:.4f}"),c(x['bn1Mean'],"{:.3f}"),c(x['gNorm'],"{:.3f}"),c(x['sae2'],"{:.4f}"),pc(x)])+"</tr>" for x in rows)
    peak=max((x['pElo'] for x in rows if x['pElo']),default=0)
    return f"<details><summary><b>{LAB[r]}</b> — {len(rows)} marks, peak pElo {peak:.0f}</summary><table>{th}{body}</table></details>"
ch="".join(f"<h2>{i+1} · {n}</h2><img src='data:image/png;base64,{b}'>" for i,(n,b) in enumerate(charts))
tb="".join(table(r) for r in ORDER)
_active='ykkk' if runs.get('ykkk') else 'coxw'
cox=runs[_active][-1] if runs.get(_active) else {}
html=f"""<!doctype html><html><head><meta charset=utf-8><title>DCM sweep dashboard</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;max-width:1040px;color:#1a1a1a}}
h1{{font-size:21px}}h2{{font-size:14px;color:#444;margin-top:26px}}img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
table{{border-collapse:collapse;font-size:11px;margin:8px 0}}td,th{{border:1px solid #ccc;padding:2px 7px;text-align:right}}
summary{{cursor:pointer;font-size:13px;margin-top:10px}}.cap{{color:#666;font-size:12px}}</style></head><body>
<h1>DCM — capacity / architecture sweep dashboard</h1>
<p class=cap>Auto-refreshed {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · {_active} @ step {cox.get('step','?')} (pElo {cox.get('pElo','?')}). Reload to update.
x = SGD steps since random init. <b>v5</b> is the full lineage as ONE continuous run (wd1e-4→wd5e-4→m0.93; warm-start param changes at the dotted blue lines, ~45.4k and ~60.9k). <b>mini2b</b>, <b>coxw</b> &amp; <b>ykkk</b> are separate fresh-init runs. bn1Mean log-y; Σαeff² ÷ each run's cap; bn1Mean/Σαeff² not cross-comparable in absolute terms (differing block counts/caps).</p>
{ch}<h2>Data tables (click to expand)</h2>{tb}</body></html>"""
open(OUT,"w").write(html)
print(f"refreshed {OUT} ({len(html)//1024} KB); {new} new SVDs; v5 combined {len(v5)} marks (to cumstep {v5[-1]['step'] if v5 else 0})")
