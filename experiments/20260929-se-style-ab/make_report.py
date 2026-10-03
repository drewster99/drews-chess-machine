import csv,re,math,statistics as st,html
ROOT='/Users/andrew/cursor/drews-chess-machine'
ARMS=[('se_sb','scale+bias','sb'),('se_att','attenuate-only','att'),('se_none','none','none')]
PARAMS={'se_sb':'5,208,050','se_att':'5,195,378','se_none':'5,170,322'}
MID={'se_sb':'20260929-12-JZOe','se_att':'20260929-13-06yp','se_none':'20260929-18-D9is'}
D={}
for k,_,_ in ARMS:
    D[k]={int(float(r['cum_step'])):r for r in csv.DictReader(open(f'{ROOT}/documentation/dashboards/data/{k}.csv')) if r.get('pElo') and int(float(r['cum_step']))<=30000}
S=sorted(set.intersection(*[set(v) for v in D.values()]))
lr={}
for l in open('/Users/andrew/Library/Logs/DrewsChessMachine/dcm_log_20260929-150727.txt'):
    m=re.search(r'\[REPLAY\] step=(\d+) .* lr=([0-9.e-]+).* mom=([0-9.]+)',l)
    if m and int(m.group(1))%1000==0: lr[int(m.group(1))]=(float(m.group(2)),float(m.group(3)))
f=lambda k,s,c: float(D[k][s][c])
wins={k:sum(1 for s in S if max(ARMS,key=lambda a:f(a[0],s,'pElo'))[0]==k) for k,_,_ in ARMS}
nw={k:sum(1 for s in S if min(ARMS,key=lambda a:f(a[0],s,'nll'))[0]==k) for k,_,_ in ARMS}
nw7={k:sum(1 for s in S if s>=7000 and min(ARMS,key=lambda a:f(a[0],s,'nll'))[0]==k) for k,_,_ in ARMS}
n7=sum(1 for s in S if s>=7000)
def mean(k,c,a,b): return st.mean(f(k,s,c) for s in S if a<=s<=b)
def swing(k,a,b):
    xs=[s for s in S if a<=s<=b]; return st.mean(abs(f(k,xs[i],'pElo')-f(k,xs[i-1],'pElo')) for i in range(1,len(xs)))
peak={k:max(S,key=lambda s:f(k,s,'pElo')) for k,_,_ in ARMS}

# ---------- markdown ----------
md=[]
md.append("# SE style A/B/C — report at step 30,000\n")
md.append("Corpus-replay comparison of squeeze-and-excitation variants on an otherwise identical 3-block 7×7 @128 v5 net. All three arms reached step 30,000 (the second LR-cycle trough) on 2026-09-30. Experiment record: [README.md](README.md).\n")
md.append("## Result\n")
md.append(f"- **No SE is best at both LR troughs** on both pElo and nll. At the 30k trough: none {f('se_none',30000,'pElo'):.1f} / {f('se_none',30000,'nll'):.4f}, attenuate-only {f('se_att',30000,'pElo'):.1f} / {f('se_att',30000,'nll'):.4f}, scale+bias {f('se_sb',30000,'pElo'):.1f} / {f('se_sb',30000,'nll'):.4f}.")
md.append(f"- **nll is the steadier signal:** no SE had the lowest nll at {nw7['se_none']} of the {n7} marks from 7k to 30k ({nw['se_none']} of all {len(S)}).")
md.append(f"- **scale+bias is consistently last:** {f('se_none',11000,'pElo')-f('se_sb',11000,'pElo'):.0f} behind no SE at the 11k trough and {f('se_none',30000,'pElo')-f('se_sb',30000,'pElo'):.0f} behind at the 30k trough.")
md.append(f"- **attenuate-only is close to no SE at the troughs but noisier:** mean absolute pElo change per mark from 10k to 30k is {swing('se_att',10000,30000):.1f} (attenuate-only) vs {swing('se_none',10000,30000):.1f} (none) and {swing('se_sb',10000,30000):.1f} (scale+bias). It dropped 101 at 20k near the LR peak.")
md.append(f"- **At the LR peak (21k) the arms were level** (scale+bias {f('se_sb',21000,'pElo'):.1f}, attenuate-only {f('se_att',21000,'pElo'):.1f}, none {f('se_none',21000,'pElo'):.1f}), but around it the SE arms were unstable: scale+bias fell to {f('se_sb',19000,'pElo'):.0f} at 19k and attenuate-only to {f('se_att',20000,'pElo'):.0f} at 20k, while no SE stayed at or above {min(f('se_none',x,'pElo') for x in range(17000,24000,1000)):.0f} from 17k to 23k.")
md.append("- **Provisional conclusion:** on this net, SE adds parameters without improving play or fit; scale+bias SE is measurably worse. One seed per arm, so gaps under ~44 pElo are within the measured seed spread (6.4–43.7).\n")
md.append("## At the LR peak and troughs\n")
md.append("The troughs (LR ≈ 9e-4) are the fair comparison points: the weights have settled, so one probe reflects the arm's real level. At the peak (LR ≈ 0.087) each checkpoint lands wherever the last large steps pushed it, so single-mark readings there are mostly noise; the peak is still reported because how an arm behaves under high LR is a result in itself.\n")
md.append("| point | LR | scale+bias pElo / nll | attenuate-only pElo / nll | none pElo / nll |\n|---|---|---|---|---|")
for s in (11000,21000,30000):
    md.append(f"| {s//1000}k {'peak' if s==21000 else 'trough'} | {lr[s][0]:.3g} | "+" | ".join(f"{f(k,s,'pElo'):.1f} / {f(k,s,'nll'):.4f}" for k,_,_ in ARMS)+" |")
md.append("\n## Curves\n\n![pElo by step](chart-pelo-30k.svg)\n\n![nll by step](chart-nll-30k.svg)\n\nShaded bands: LR troughs (~11k, ~30k) and the cycle peak (~21k). Styled page: [report-30k.html](report-30k.html).")
md.append("\n## Summary by arm (steps 1k–30k)\n")
md.append("| | scale+bias | attenuate-only | none |\n|---|---|---|---|")
rows=[("params",lambda k:PARAMS[k]),("ModelID",lambda k:MID[k]),
("pElo @30k",lambda k:f"{f(k,30000,'pElo'):.1f}"),("nll @30k",lambda k:f"{f(k,30000,'nll'):.4f}"),
("peak pElo (step)",lambda k:f"{f(k,peak[k],'pElo'):.1f} ({peak[k]//1000}k)"),
("mean pElo 20k–30k",lambda k:f"{mean(k,'pElo',20000,30000):.1f}"),("mean nll 20k–30k",lambda k:f"{mean(k,'nll',20000,30000):.4f}"),
("marks with best pElo",lambda k:f"{wins[k]} of {len(S)}"),("marks with best nll",lambda k:f"{nw[k]} of {len(S)}"),
("mean |ΔpElo| per mark, 10k–30k",lambda k:f"{swing(k,10000,30000):.1f}"),
("bn1Mean @30k",lambda k:f"{f(k,30000,'bn1Mean'):.4f}"),("legalMass @30k",lambda k:f"{f(k,30000,'legalMass'):.4f}")]
for n,fn in rows: md.append(f"| {n} | "+" | ".join(fn(k) for k,_,_ in ARMS)+" |")
md.append("\n## pElo and nll at every mark\n")
md.append("| step | LR | momentum | scale+bias pElo | attenuate-only pElo | none pElo | scale+bias nll | attenuate-only nll | none nll |\n|---|---|---|---|---|---|---|---|---|")
for s in S:
    md.append(f"| {s} | {lr[s][0]:.3g} | {lr[s][1]:.3f} | "+" | ".join(f"{f(k,s,'pElo'):.1f}" for k,_,_ in ARMS)+" | "+" | ".join(f"{f(k,s,'nll'):.4f}" for k,_,_ in ARMS)+" |")
md.append("\n## Setup\n")
md.append("- **Architecture:** v5-style, basic30 input, 7×7 stem → 3×[7×7+7×7 @128, SE /4, ReLU pre-act, ReZero α 0.447 (tanh-capped), clean_add, LayerNorm out], policy intermediate_conv (128), value WDL (16ch → FC128), bf16. Only `se_style` differs between arms.")
md.append("- **Training:** corpus replay of `20260624-192615-w3aA5b` (Lichess 2026-05, 20.9M games), batch 4096, replay ratio 0.48, 500k buffer (250k prefill), weight decay 3e-4, warmup 1000, grad clip 15.")
md.append("- **Schedule:** decaying LR cycle, peak 1e-1→1e-4 and trough 1e-3→1e-6 over 1M steps, 20k-step period starting at the peak; momentum follows the cycle (0.85→0.90 low, 0.95 high). Troughs fell at ~11k and ~30k.")
md.append("- **Probes:** wide-set pElo / nll on each enumerated 1k-step checkpoint (`documentation/dashboards/data/se_{sb,att,none}.csv`).")
md.append("\n## Caveats\n")
md.append("- One random seed per arm; the audited seed spread for this family is 6.4–43.7 pElo.")
md.append("- Arms ran concurrently on one GPU; compare by step, not by time.")
md.append("- Runs are still going; these are results at step 30,000, not final.")
open(f'{ROOT}/experiments/20260929-se-style-ab/REPORT-30k.md','w').write("\n".join(md)+"\n")

# ---------- svg charts ----------
COL={'se_sb':'var(--sb)','se_att':'var(--att)','se_none':'var(--none)'}
def chart(col,ttl,lo,hi,step,fmt,top=False):
    W,H,L,R,T,B=760,300,56,16,16,36
    xs=lambda s:L+(s-1000)/(29000)*(W-L-R)
    ys=lambda v:T+(hi-v)/(hi-lo)*(H-T-B)
    o=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(ttl)}">']
    # LR trough/peak bands
    for a,b,lab in ((9500,12500,'trough'),(19500,22500,'peak'),(28500,30000,'trough')):
        o.append(f'<rect x="{xs(a):.1f}" y="{T}" width="{xs(b)-xs(a):.1f}" height="{H-T-B}" class="band{"-p" if lab=="peak" else ""}"/>')
        o.append(f'<text x="{(xs(a)+xs(b))/2:.1f}" y="{T+12}" class="bl">LR {lab}</text>')
    v=lo
    while v<=hi+1e-9:
        y=ys(v); o.append(f'<line x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{L-8}" y="{y+4:.1f}" class="yl">{fmt(v)}</text>'); v+=step
    for s in range(5000,30001,5000):
        o.append(f'<text x="{xs(s):.1f}" y="{H-12}" class="xl">{s//1000}k</text>')
    for k,_,cls in ARMS:
        pts=" ".join(f"{xs(s):.1f},{ys(f(k,s,col)):.1f}" for s in S)
        o.append(f'<polyline points="{pts}" class="ln {cls}"/>')
        s=S[-1]; o.append(f'<circle cx="{xs(s):.1f}" cy="{ys(f(k,s,col)):.1f}" r="3.5" class="dot {cls}"/>')
    lx=W-R-150
    ly=T+22 if top else H-B-70
    o.append(f'<rect x="{lx-8}" y="{ly}" width="150" height="62" rx="4" class="lg"/>')
    for i,(k,name,cls) in enumerate(ARMS[::-1]):
        y=ly+16+i*18
        o.append(f'<line x1="{lx}" x2="{lx+22}" y1="{y}" y2="{y}" class="ln {cls}"/><text x="{lx+30}" y="{y+4}" class="lt">{name}</text>')
    o.append('</svg>'); return "\n".join(o)
c1=chart('pElo','pElo by step',850,1500,100,lambda v:f"{v:.0f}")
c2=chart('nll','nll by step',2.20,3.20,0.2,lambda v:f"{v:.1f}",top=True)

def tbl_rows():
    out=[]
    for s in S:
        pe=[f(k,s,'pElo') for k,_,_ in ARMS]; nl=[f(k,s,'nll') for k,_,_ in ARMS]
        bp=pe.index(max(pe)); bn=nl.index(min(nl))
        tr=' class="tr"' if s in (11000,30000) else ''
        cells="".join(f'<td class="{"best" if i==bp else ""}">{v:.1f}</td>' for i,v in enumerate(pe))+"".join(f'<td class="{"best" if i==bn else ""}">{v:.4f}</td>' for i,v in enumerate(nl))
        out.append(f'<tr{tr}><th>{s//1000}k</th><td>{lr[s][0]:.3g}</td>{cells}</tr>')
    return "\n".join(out)
summ="\n".join(f'<tr><th>{html.escape(n)}</th>'+"".join(f'<td>{html.escape(fn(k))}</td>' for k,_,_ in ARMS)+'</tr>' for n,fn in rows)
trough="\n".join(f'<tr><th>{s//1000}k {"peak" if s==21000 else "trough"}</th><td>{lr[s][0]:.3g}</td>'+"".join(f'<td>{f(k,s,"pElo"):.1f} <span class="sub">/ {f(k,s,"nll"):.4f}</span></td>' for k,_,_ in ARMS)+'</tr>' for s in (11000,21000,30000))

HTML=f'''<title>SE Style A/B/C</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
:root{{--bg:#F5F6F4;--panel:#FFFFFF;--ink:#1C2126;--muted:#5E6770;--rule:#D9DDD8;--grid:#E6E9E5;
--sb:#B5532E;--att:#2F6F98;--none:#2F7D55;--band:rgba(47,111,152,.07);--bandp:rgba(181,83,46,.08);--hi:#EEF4EC;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14171A;--panel:#1B1F23;--ink:#E6E9EC;--muted:#9AA3AB;--rule:#2C3238;--grid:#262B30;
--sb:#E0845E;--att:#6FAEDB;--none:#6CC495;--band:rgba(111,174,219,.10);--bandp:rgba(224,132,94,.10);--hi:#1F2A22;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#14171A;--panel:#1B1F23;--ink:#E6E9EC;--muted:#9AA3AB;--rule:#2C3238;--grid:#262B30;
--sb:#E0845E;--att:#6FAEDB;--none:#6CC495;--band:rgba(111,174,219,.10);--bandp:rgba(224,132,94,.10);--hi:#1F2A22;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:15px/1.6 "IBM Plex Sans",system-ui,sans-serif;padding-inline:16px;padding-block:32px 64px}}
main{{max-width:900px;margin:0 auto;display:grid;gap:36px}}
h1{{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:clamp(28px,5vw,40px);line-height:1.15;margin:0;text-wrap:balance}}
h2{{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 12px;font-weight:600}}
p{{margin:0;max-width:68ch}} .lede{{color:var(--muted);font-size:16px}}
.eyebrow{{font:500 12px "IBM Plex Mono",monospace;letter-spacing:.06em;color:var(--muted)}}
.key{{display:flex;gap:18px;flex-wrap:wrap;font:500 13px "IBM Plex Mono",monospace}}
.key span::before{{content:"";display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle;background:currentColor}}
.k-sb{{color:var(--sb)}} .k-att{{color:var(--att)}} .k-none{{color:var(--none)}}
ul.find{{margin:0;padding-left:20px;display:grid;gap:8px;max-width:72ch}}
.panel{{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:16px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}}
th:first-child{{text-align:left}} thead th{{font-weight:600;color:var(--muted);font-size:12px}}
td{{font-family:"IBM Plex Mono",monospace}} .sub{{color:var(--muted)}}
td.best{{font-weight:600;color:var(--none)}} tr.tr{{background:var(--hi)}}
svg{{width:100%;height:auto;display:block}} .grid{{stroke:var(--grid)}} .yl,.xl,.bl{{fill:var(--muted);font:11px "IBM Plex Mono",monospace}}
.yl{{text-anchor:end}} .xl,.bl{{text-anchor:middle}} .band{{fill:var(--band)}} .band-p{{fill:var(--bandp)}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round}} .ln.sb{{stroke:var(--sb)}} .ln.att{{stroke:var(--att)}} .ln.none{{stroke:var(--none)}}
.dot.sb{{fill:var(--sb)}} .dot.att{{fill:var(--att)}} .dot.none{{fill:var(--none)}}
text.lt{{fill:var(--ink);font:12px "IBM Plex Mono",monospace}} .lg{{fill:var(--panel);stroke:var(--rule)}}
.charts{{display:grid;gap:20px}} .note{{font-size:13px;color:var(--muted)}}
</style>
<main>
<header style="display:grid;gap:10px">
<div class="eyebrow">DCM · corpus replay · 2026-09-30 · step 30,000 of 3 arms</div>
<h1>No SE beats both SE styles on this net</h1>
<p class="lede">Three copies of the same 3-block 7×7 @128 network, identical except for the squeeze-and-excitation block, trained on the same Lichess corpus with the same decaying LR cycle. All three have now passed the second LR trough.</p>
<div class="key"><span class="k-none">none</span><span class="k-att">attenuate-only</span><span class="k-sb">scale+bias</span></div>
</header>
<section><h2>Findings</h2><ul class="find">
<li><b>No SE is best at both LR troughs</b> on pElo and nll: {f('se_none',30000,'pElo'):.1f} / {f('se_none',30000,'nll'):.4f} at 30k, against attenuate-only {f('se_att',30000,'pElo'):.1f} / {f('se_att',30000,'nll'):.4f} and scale+bias {f('se_sb',30000,'pElo'):.1f} / {f('se_sb',30000,'nll'):.4f}.</li>
<li><b>nll is the steadier signal.</b> No SE had the lowest nll at {nw7['se_none']} of the {n7} marks from 7k to 30k.</li>
<li><b>Scale+bias is consistently last:</b> {f('se_none',11000,'pElo')-f('se_sb',11000,'pElo'):.0f} pElo behind no SE at the 11k trough and {f('se_none',30000,'pElo')-f('se_sb',30000,'pElo'):.0f} behind at 30k.</li>
<li><b>Attenuate-only is close at the troughs but noisier.</b> Its pElo moved {swing('se_att',10000,30000):.1f} per mark on average from 10k to 30k, against {swing('se_none',10000,30000):.1f} for no SE, and it fell 101 at 20k near the LR peak.</li>
<li><b>At the LR peak (21k) the arms were level</b> (scale+bias {f('se_sb',21000,'pElo'):.1f}, attenuate-only {f('se_att',21000,'pElo'):.1f}, none {f('se_none',21000,'pElo'):.1f}), but around it the SE arms were unstable: scale+bias fell to {f('se_sb',19000,'pElo'):.0f} at 19k and attenuate-only to {f('se_att',20000,'pElo'):.0f} at 20k, while no SE never dropped below {min(f('se_none',x,'pElo') for x in range(17000,24000,1000)):.0f}.</li>
<li><b>Provisional:</b> SE adds parameters here without improving play or fit, and scale+bias SE is measurably worse. One seed per arm, so gaps under ~44 pElo sit inside the measured seed spread.</li>
</ul></section>
<section><h2>At the LR peak and troughs</h2><p style="margin-bottom:12px">The troughs (LR ≈ 9e-4) are the fair comparison: weights have settled, so a single probe reflects the arm's level. Near the peak (LR ≈ 0.087, step 21k) each checkpoint lands wherever the last large steps pushed it. At the peak itself all three were level (7-point spread), but in the marks around it the SE arms swung 60–130 pElo while no SE stayed within about 35. That instability is the peak's result.</p><div class="panel scroll"><table>
<thead><tr><th>point</th><th>LR</th><th>scale+bias pElo / nll</th><th>attenuate-only</th><th>none</th></tr></thead><tbody>{trough}</tbody></table></div></section>
<section class="charts"><h2>Curves</h2>
<div class="panel"><div class="eyebrow" style="margin-bottom:6px">pElo (wide probe set), higher is better</div>{c1}</div>
<div class="panel"><div class="eyebrow" style="margin-bottom:6px">nll, lower is better</div>{c2}</div>
<p class="note">Shaded bands mark the LR troughs (≈1e-3 → 9e-4) and the cycle peak (≈0.087). The SE arms swing hardest around the peak.</p></section>
<section><h2>Summary by arm</h2><div class="panel scroll"><table>
<thead><tr><th></th><th>scale+bias</th><th>attenuate-only</th><th>none</th></tr></thead><tbody>{summ}</tbody></table></div></section>
<section><h2>Every 1k mark</h2><div class="panel scroll"><table>
<thead><tr><th>step</th><th>LR</th><th>pElo s+b</th><th>pElo att</th><th>pElo none</th><th>nll s+b</th><th>nll att</th><th>nll none</th></tr></thead><tbody>{tbl_rows()}</tbody></table></div>
<p class="note" style="margin-top:8px">Bold marks the best arm at each step; shaded rows are the LR troughs.</p></section>
<section><h2>Setup</h2><ul class="find">
<li><b>Net:</b> basic30 input, 7×7 stem → 3×[7×7+7×7 @128, SE /4, ReLU pre-act, ReZero α 0.447, clean add, LayerNorm out], policy intermediate_conv, value W/D/L, bf16. Only <code>se_style</code> differs.</li>
<li><b>Data:</b> corpus replay of Lichess 2026-05 (<code>w3aA5b</code>, 20.9M games), batch 4096, replay ratio 0.48, 500k buffer.</li>
<li><b>Schedule:</b> LR cycle peak 1e-1→1e-4, trough 1e-3→1e-6 over 1M steps, 20k-step period; momentum follows (0.85–0.95). Weight decay 3e-4, warmup 1000, clip 15.</li>
<li><b>Caveats:</b> one seed per arm; arms shared one GPU, so compare by step; runs continue past 30k.</li>
</ul></section>
</main>'''
EXP=f'{ROOT}/experiments/20260929-se-style-ab'
LIT={'var(--sb)':'#B5532E','var(--att)':'#2F6F98','var(--none)':'#2F7D55'}
CSS='<style>.grid{stroke:#E6E9E5}.yl,.xl,.bl{fill:#5E6770;font:11px monospace}.yl{text-anchor:end}.xl,.bl{text-anchor:middle}.band{fill:rgba(47,111,152,.07)}.band-p{fill:rgba(181,83,46,.08)}.ln{fill:none;stroke-width:2}.ln.sb{stroke:#B5532E}.ln.att{stroke:#2F6F98}.ln.none{stroke:#2F7D55}.dot.sb{fill:#B5532E}.dot.att{fill:#2F6F98}.dot.none{fill:#2F7D55}.lt{fill:#1C2126;font:12px monospace}.lg{fill:#fff;stroke:#D9DDD8}</style><rect width="100%" height="100%" fill="#fff"/>'
for name,c in (('chart-pelo-30k.svg',c1),('chart-nll-30k.svg',c2)):
    open(f'{EXP}/{name}','w').write(c.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ',1).replace('aria-label',CSS.join(['aria-label',''])[:0]+'aria-label',1).replace('">','">'+CSS,1))
open(f'{EXP}/report-30k.html','w').write('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+HTML)
print("ok",len(S))
