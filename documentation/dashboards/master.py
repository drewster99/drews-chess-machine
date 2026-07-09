#!/usr/bin/env python3
"""Combined interactive MASTER dashboard (main + elite runs) -> dcm_master.html.

Self-contained, dependency-free, Canvas-rendered. Features:
  - one summary table with a multi-line Corpus cell (label + plies + epochs),
    full corpus detail on hover;
  - a Corpora catalog listing every corpus (used or not);
  - toggleable chart series (click legend or row), persisted in localStorage;
  - by-step/by-time + smoothed/raw toggles (persisted);
  - dark-mode aware, and near-black series recolored to gray in dark mode;
  - smart hover: shows all checked series — the one whose line is nearest the
    cursor is highlighted, in-range ones normal, already-ended ones dimmed and
    reported at their final value.

Run:  python3 master.py
"""
import os, sys, json, csv, html, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))  # documentation/dashboards -> repo root
K_POS_PER_STEP = 8533   # batch/replay-ratio = 4096/0.48; used to derive epochs

# Curated corpus metadata: `label` (MUST equal the runs' `trainset`) + `source` +
# `note`. Everything that can DRIFT — whether the corpus still exists on disk, its
# game/ply counts, and completeness — is read LIVE from each corpus.json by
# _load_corpora() at render time, so a deleted corpus disappears and the counts
# never go stale against a hand-maintained list.
_CORPORA_DIR = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Corpora")
_CURATED_CORPORA = [
    {"label": "std-2026-05", "id": "20260624-192615-w3aA5b",
     "source": "lichess_db_standard_rated_2026-05.pgn",
     "note": "disk-full-truncated prefix (never sealed)"},
    {"label": "elite-2025-05_11", "id": "20260704-001142-lLOrLj",
     "source": "lichess_elite_2025-05_to_11.pgn",
     "note": "PARTIAL elite (7 months, 2025-05 → 11)"},
    {"label": "elite-2021_2025", "id": "20260704-215145-op24Gp",
     "source": "lichess_elite_2021-12_to_2025-11.pgn",
     "note": "FULL elite (4 years)"},
]


def _human(n):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}".rstrip("0").rstrip(".") + suf
    return str(n)


def _load_corpora():
    """Curated label/source/note overlaid with LIVE corpus.json: existence, per-
    source game/ply totals, and completeness (sealed). A corpus whose directory is
    gone is dropped (no phantom rows); a corpus.json that is present but unreadable
    is surfaced as an explicit ERROR row + a stderr warning (never silently
    defaulted) — but it does NOT abort the whole render, so one corrupt metadata
    file can't blind the operator to every training chart."""
    out = []
    for c in _CURATED_CORPORA:
        cj = os.path.join(_CORPORA_DIR, c["id"], "corpus.json")
        if not os.path.exists(cj):
            continue
        try:
            with open(cj) as fh:
                m = json.load(fh)
        except (ValueError, OSError) as e:
            sys.stderr.write(f"WARNING: corpus.json unreadable for {c['id']}: {e}\n")
            out.append({**c, "name": f"{c['id']} (corpus.json UNREADABLE)",
                        "games": "ERR", "plies": "ERR", "plies_n": 0.0,
                        "complete": False})
            continue
        srcs = m.get("sources", [])
        g = sum((s.get("gamesAdded") or 0) for s in srcs)
        p = sum((s.get("pliesAdded") or 0) for s in srcs)
        # "complete" = every source finished importing (per-source flag), NOT sealed —
        # the elite corpora are full imports left in the recording state.
        complete = bool(srcs) and all(s.get("complete") is True for s in srcs)
        out.append({**c, "name": m.get("name", c["id"]),
                    "games": _human(g), "plies": _human(p), "plies_n": float(p),
                    "complete": complete})
    return out


CORPORA = _load_corpora()
CORP_BY_LABEL = {c["label"]: c for c in CORPORA}

METRICS = [
    ("pElo", "pElo", False), ("nll", "NLL", False), ("loss", "loss (combined)", False),
    ("pLoss", "pLoss", False), ("vLoss", "vLoss", False), ("legalMass", "legalMass", False),
    ("gNorm", "gNorm", False), ("bn1Mean", "bn1Mean", True), ("sae2", "Σαeff²", False),
    ("pLogit_mean", "pLogitAbsMax mean", False),
]
NUMKEYS = [m[0] for m in METRICS]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_rows(data_dir, run):
    p = os.path.join(data_dir, f"{run}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def active_procs_text():
    """Concatenated args of live `--replay-corpus` processes, so the currently-
    training run can be flagged by matching its out-model basename (paths contain
    spaces — 'Application Support' — so we substring-match rather than split)."""
    try:
        ps = subprocess.check_output(["ps", "-Ao", "args="], text=True)
    except Exception:
        return ""
    return "\n".join(l for l in ps.splitlines() if "replay-corpus" in l)


def collect():
    main = json.load(open(os.path.join(HERE, "registry.json")))
    elite = json.load(open(os.path.join(HERE, "elite", "registry.json")))
    ACTIVE = active_procs_text()
    runs = []

    def build(key, cfg, data_dir):
        rows = read_rows(data_dir, key)
        s, t, m = [], [], {k: [] for k in NUMKEYS}
        for r in rows:
            cum = _f(r.get("cum_step"))
            if cum is None:
                continue
            el = _f(r.get("elapsed_train_sec"))
            s.append(int(cum))
            t.append(round(el / 3600.0, 4) if el is not None else None)
            for k in NUMKEYS:
                m[k].append(_f(r.get(k)))
        ts = cfg.get("trainset") or "?"
        corp = CORP_BY_LABEL.get(ts)
        peak = max([v for v in m["pElo"] if v is not None], default=None)
        final_cum = s[-1] if s else 0
        epochs = (final_cum * K_POS_PER_STEP / corp["plies_n"]) if corp and corp.get("plies_n") else None

        def last(key):
            for r in reversed(rows):
                if _f(r.get(key)) is not None:
                    return r.get(key)
            return "—"
        return {
            "k": key, "label": cfg.get("label", key), "color": cfg.get("color", "#888"),
            "type": cfg.get("type", "replay"),
            "trainset": ts, "corp": corp, "epochs": epochs,
            "active": bool(cfg.get("out_model")) and os.path.basename(cfg["out_model"]) in ACTIVE,
            "params": cfg.get("params", 0),
            "stem": cfg.get("arch_stem", cfg.get("arch_summary", "")),
            "blocks": cfg.get("arch_blocks", ""), "heads": cfg.get("arch_heads", ""),
            "n": len(s), "peak": peak, "steps": final_cum,
            "hrs": max([x for x in t if x is not None], default=0.0),
            "final_pElo": last("pElo"), "final_nll": last("nll"),
            "s": s, "t": t, "m": m,
        }

    for key, cfg in main["runs"].items():
        runs.append(build(key, cfg, os.path.join(HERE, "data")))
    for key in ("qeu8e",):
        if key in elite["runs"]:
            runs.append(build(key, elite["runs"][key], os.path.join(HERE, "elite", "data")))
    # self-play (non-replay) runs — same data/ dir + CSV schema, type=selfplay
    sp_path = os.path.join(HERE, "selfplay_registry.json")
    if os.path.exists(sp_path):
        sp = json.load(open(sp_path))
        for key, cfg in sp["runs"].items():
            runs.append(build(key, cfg, os.path.join(HERE, "data")))
    return runs


def render():
    runs = collect()
    data_json = json.dumps({
        "runs": runs, "metrics": [{"key": k, "title": ti, "logy": ly} for k, ti, ly in METRICS],
    }, separators=(",", ":"))

    def sw(c):
        return f"<span class=sw style='background:{c}'></span>"

    # ---- summary rows: Corpus cell = chip + plies·epochs, full detail on hover ----
    srows = ""
    for r in sorted(runs, key=lambda r: -(r["peak"] or 0)):
        peak = f"{r['peak']:.0f}" if r["peak"] else "—"
        c = r["corp"]
        if c:
            est = c["complete"] is False
            detail = (f"{c['name']}&#10;{c['games']} games · {c['plies']} plies"
                      f"&#10;{'PARTIAL — ' if est else 'complete · '}source: {c['source']}"
                      f"&#10;{c['note']}")
            epv = r["epochs"]
            pre = "~" if est else ""
            if epv is None:
                epline = ""
            elif epv >= 1.0:
                epline = f"<div class='ep ok' title='trained through at least one full pass of this corpus'>{pre}{epv:.1f} epochs · ≥1 full pass</div>"
            else:
                epline = f"<div class='ep part' title='never reached one full pass of this corpus'>{pre}{epv:.2f} epoch · partial pass</div>"
            corpcell = (f"<td class=corpus title=\"{detail}\">"
                        f"<span class=chip>{html.escape(r['trainset'])}</span>"
                        f"<div class=csub>{html.escape(c['plies'])} plies</div>"
                        f"{epline}</td>")
        elif r["type"] == "selfplay":
            corpcell = ("<td class=corpus><span class=chip>self-play</span>"
                        "<div class=csub>no corpus</div></td>")
        else:
            corpcell = "<td class=corpus><span class=chip>?</span></td>"
        tp = r["type"]
        typecell = f"<td><span class='tp {tp}'>{'self-play' if tp=='selfplay' else 'replay'}</span></td>"
        live = " <span class=live title='this run is training right now'>● running</span>" if r["active"] else ""
        srows += (
            f"<tr data-k='{r['k']}'{' class=liverow' if r['active'] else ''}>"
            f"<td class=run>{sw(r['color'])}{html.escape(r['k'])}{live}</td>"
            f"{typecell}"
            f"{corpcell}"
            f"<td class=arch>{html.escape(r['stem'])}</td>"
            f"<td class=arch>{html.escape(r['blocks'])}</td>"
            f"<td class=arch>{html.escape(r['heads'])}</td>"
            f"<td class=num>{r['params']:,}</td>"
            f"<td class=num>{r['steps']:,}</td>"
            f"<td class=num>{r['hrs']:.1f}</td>"
            f"<td class=num><b>{peak}</b></td>"
            f"<td class=num>{html.escape(str(r['final_pElo']))}</td>"
            f"<td class=num>{html.escape(str(r['final_nll']))}</td></tr>")

    # ---- corpora catalog ----
    corprows = ""
    for c in CORPORA:
        used = sorted(r["k"] for r in runs if r["trainset"] == c["label"])
        usedcell = ", ".join(used) if used else "<i style='color:var(--mut)'>never run</i>"
        comp = "yes" if c["complete"] else "<b>partial</b>"
        corprows += (
            f"<tr><td><span class=chip>{html.escape(c['label'])}</span></td>"
            f"<td class=arch>{html.escape(c['name'])}</td>"
            f"<td class=num>{html.escape(c['games'])}</td>"
            f"<td class=num>{html.escape(c['plies'])}</td>"
            f"<td>{comp}</td><td>{usedcell}</td>"
            f"<td class=arch>{html.escape(c['note'])}</td></tr>")

    tpl = (_HTML.replace("__DATA__", data_json).replace("__SROWS__", srows)
                .replace("__CORPROWS__", corprows))
    out = os.path.join(REPO_ROOT, "dcm_master.html")
    open(out, "w").write(tpl)
    print(f"wrote {out} ({len(tpl)//1024} KB, {len(runs)} runs)")


_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>DCM — master dashboard</title>
<style>
:root{--bg:#faf9f7;--ink:#1c1b19;--mut:#6b6862;--line:#e4e0d8;--card:#fff;--accent:#8c564b;--arch:#555}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:22px 22px 60px}
h1{font-size:20px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:12.5px;margin:0 0 18px}
h2{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin:26px 0 8px;font-weight:600}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:left;white-space:nowrap;vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
 position:sticky;top:0;background:var(--bg);cursor:pointer;user-select:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.arch{white-space:normal;max-width:230px;color:var(--arch);font-size:11px}
td.corpus{white-space:nowrap}
td.corpus .csub{color:var(--mut);font-size:10.5px;margin-top:3px;font-variant-numeric:tabular-nums}
td.corpus .ep{font-size:10.5px;margin-top:2px;font-weight:600;font-variant-numeric:tabular-nums;cursor:help}
td.corpus .ep.ok{color:#2e9d57}td.corpus .ep.part{color:#cf8a2c}
.live{color:#2e9d57;font-size:9.5px;font-weight:700;margin-left:6px;white-space:nowrap;vertical-align:1px}
tr.liverow{background:#2e9d5714}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
@media(prefers-reduced-motion:no-preference){.live{animation:pulse 1.6s ease-in-out infinite}}
.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.chip{background:#efece6;border:1px solid var(--line);border-radius:999px;padding:1px 8px;font-size:11px;color:#4a463f;cursor:help}
.tp{display:inline-block;border-radius:5px;padding:1px 7px;font-size:10.5px;font-weight:600;letter-spacing:.02em}
.tp.replay{background:#2f6fed1a;color:#2f6fed;border:1px solid #2f6fed44}
.tp.selfplay{background:#b4530e1a;color:#c8631a;border:1px solid #c8631a44}
tr.off{opacity:.4}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:6px 0 14px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card)}
.seg button{border:0;background:transparent;padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer;color:var(--mut)}
.seg button.on{background:var(--accent);color:#fff}
.mini{font-size:12px;color:var(--mut);background:var(--card);border:1px solid var(--line);
 border-radius:8px;padding:5px 10px;cursor:pointer}
.mini:hover{border-color:#c9c4ba}
.legend{display:flex;flex-wrap:wrap;gap:6px 8px;margin:4px 0 20px}
.lg{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);background:var(--card);
 border-radius:8px;padding:4px 10px;cursor:pointer;font-size:12.5px;user-select:none}
.lg .sw{width:12px;height:12px;margin:0}.lg.off{opacity:.42}.lg .ts{color:var(--mut);font-size:11px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px 6px}
.card h3{margin:0 0 2px;font-size:13px;font-weight:600}
.card canvas{width:100%;height:250px;display:block;touch-action:none;cursor:crosshair}
details{margin:8px 0}summary{cursor:pointer;color:var(--mut);font-size:12.5px;padding:4px 0}
.tip{position:fixed;pointer-events:none;background:#111014f2;color:#fff;border-radius:9px;padding:8px 10px;
 font-size:11.5px;line-height:1.55;z-index:50;display:none;box-shadow:0 6px 20px #0005;min-width:150px}
.tip .hd{color:#aaa;margin-bottom:4px;font-variant-numeric:tabular-nums}
.tip .r{display:flex;justify-content:space-between;gap:14px;align-items:center}
.tip .r.dim{opacity:.4}
.tip .r.hi{background:#ffffff1f;border-radius:5px;margin:1px -5px;padding:1px 5px;font-weight:700}
.tip .r .fin{color:#999;font-size:9.5px;font-weight:400;margin-left:4px}
.tip .r b{font-variant-numeric:tabular-nums}
@media (prefers-color-scheme:dark){:root{--bg:#171614;--ink:#eceae6;--mut:#9a958c;--line:#302e2a;--card:#201e1b;--arch:#c1bcb2}
 .chip{background:#2a2825;color:#d8d3ca}.seg button.on{color:#fff}}
</style></head><body><div class=wrap>
<h1>DCM — master dashboard</h1>
<p class=sub id=sub></p>

<h2>All runs</h2>
<div class=tblwrap><table id=summary><thead><tr>
<th data-s=k>run</th><th data-s=type>type</th><th data-s=trainset>corpus</th>
<th>input + stem</th><th>blocks</th><th>heads</th>
<th class=num data-s=params>params</th><th class=num data-s=steps>steps</th>
<th class=num data-s=hrs>hrs</th><th class=num data-s=peak>peak pElo</th>
<th class=num data-s=final_pElo>final pElo</th><th class=num data-s=final_nll>final NLL</th>
</tr></thead><tbody>__SROWS__</tbody></table></div>

<h2>Corpora</h2>
<div class=tblwrap><table id=corpora><thead><tr>
<th>label (on runs)</th><th>full name</th><th class=num>games</th><th class=num>plies</th>
<th>complete</th><th>used by</th><th>note</th>
</tr></thead><tbody>__CORPROWS__</tbody></table></div>

<h2>Charts</h2>
<div class=controls>
 <div class=seg id=typeseg><button data-tp=all>all</button><button data-tp=replay>replay</button><button data-tp=selfplay>self-play</button></div>
 <div class=seg id=xseg><button data-x=step>by step</button><button data-x=time>by time</button></div>
 <div class=seg id=emaseg><button data-e=1>smoothed</button><button data-e=0>raw</button></div>
 <button class=mini id=all>all</button><button class=mini id=none>none</button>
 <span id=tsquick></span>
</div>
<div class=legend id=legend></div>
<div class=grid id=grid></div>

<h2>Data tables</h2>
<div id=tables></div>
<div class=tip id=tip></div>
</div>
<script>
const DATA = __DATA__;
const RUNS = DATA.runs, METRICS = DATA.metrics;
const DARK = window.matchMedia && matchMedia("(prefers-color-scheme:dark)").matches;
// near-black is invisible on the dark card -> recolor to a medium gray (dark only)
RUNS.forEach(r=>{ if(DARK && (r.color||"").toLowerCase()=="#000000") r.color="#9aa0a6"; });
const byKey = Object.fromEntries(RUNS.map(r=>[r.k,r]));
const LS="dcm_master_v3";
const REPLAY_KEYS=RUNS.filter(r=>r.type!="selfplay").map(r=>r.k);
let st = Object.assign({checked:REPLAY_KEYS, x:"step", ema:1, tp:"replay"},
                       JSON.parse(localStorage.getItem(LS)||"{}"));
const checked = new Set(st.checked.filter(k=>byKey[k]));
if(!checked.size) REPLAY_KEYS.forEach(k=>checked.add(k));
function save(){localStorage.setItem(LS,JSON.stringify({checked:[...checked],x:st.x,ema:st.ema,tp:st.tp}));}

document.getElementById("sub").textContent =
 RUNS.length+" runs · "+new Set(RUNS.filter(r=>r.type!="selfplay").map(r=>r.trainset)).size+" corpora · "
 +"hover a chart to inspect (nearest line highlighted) · toggle series in the legend or table · saved across reloads";

/* fix table swatches for the dark recolor (they were server-rendered) */
if(DARK) document.querySelectorAll("#summary tbody tr").forEach(tr=>{
 const r=byKey[tr.dataset.k],s=tr.querySelector(".sw"); if(r&&s) s.style.background=r.color;});

/* ---- helpers ---- */
const DPR=Math.max(1,window.devicePixelRatio||1);
function niceStep(range,ticks){let raw=range/ticks,p=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/p;
 let n=f<1.5?1:f<3?2:f<7?5:10;return n*p;}
function ticksFor(lo,hi,n){if(!(hi>lo))return[lo];let stp=niceStep(hi-lo,n),a=[];
 for(let v=Math.ceil(lo/stp)*stp;v<=hi+1e-9;v+=stp)a.push(v);return a;}
function ema(ys,span){let a=2/(span+1),s=null,o=[];for(const y of ys){s=(s==null)?y:a*y+(1-a)*s;o.push(s);}return o;}
function fmtY(v){let a=Math.abs(v);if(a>=1000)return v.toFixed(0);if(a>=10)return v.toFixed(1);
 if(a>=1)return v.toFixed(2);if(a===0)return"0";return v.toFixed(3);}
function fmtX(v){return st.x=="time"?v.toFixed(1)+"h":(v>=1000?(v/1000).toFixed(0)+"k":v.toFixed(0));}
function rawSeries(run,key){const xa=st.x=="step"?run.s:run.t,ya=run.m[key],o=[];
 for(let i=0;i<xa.length;i++){let x=xa[i],y=ya[i];if(x==null||y==null)continue;o.push([x,y]);}return o;}
/* what actually gets drawn (ema line or raw) — used for both drawing and hover */
function drawnSeries(run,key){let pts=rawSeries(run,key);
 if(st.ema&&pts.length>3){let e=ema(pts.map(p=>p[1]),10);return pts.map((p,i)=>[p[0],e[i]]);}
 return pts;}

/* ---- chart ---- */
class Chart{
 constructor(metric){this.m=metric;
  const card=document.createElement("div");card.className="card";
  card.innerHTML="<h3>"+metric.title+(metric.logy?" (log y)":"")+"</h3>";
  this.cv=document.createElement("canvas");card.appendChild(this.cv);
  document.getElementById("grid").appendChild(card);
  this.cv.addEventListener("pointermove",e=>this.hover(e));
  this.cv.addEventListener("pointerleave",()=>{tip.style.display="none";this.draw();});}
 layout(){const r=this.cv.getBoundingClientRect();this.W=r.width;this.H=r.height;
  this.cv.width=r.width*DPR;this.cv.height=r.height*DPR;
  const c=this.cv.getContext("2d");c.setTransform(DPR,0,0,DPR,0,0);this.c=c;
  this.pad={l:52,r:10,t:8,b:22};}
 bounds(){let xs=[],ys=[];this.S=[];
  for(const k of checked){const run=byKey[k];let pts=drawnSeries(run,this.m.key);if(!pts.length)continue;
   this.S.push({run,pts});for(const[x,y]of pts){xs.push(x);
    if(this.m.logy){if(y>0)ys.push(Math.log10(y));}else ys.push(y);}}
  if(!xs.length){this.ok=false;return;}
  this.ok=true;this.x0=Math.min(...xs);this.x1=Math.max(...xs);
  this.y0=Math.min(...ys);this.y1=Math.max(...ys);
  if(this.x1==this.x0)this.x1=this.x0+1;let pd=(this.y1-this.y0)*0.06||1;this.y0-=pd;this.y1+=pd;}
 px(x){return this.pad.l+(x-this.x0)/(this.x1-this.x0)*(this.W-this.pad.l-this.pad.r);}
 py(yv){let y=this.m.logy?Math.log10(yv):yv;
  return this.pad.t+(1-(y-this.y0)/(this.y1-this.y0))*(this.H-this.pad.t-this.pad.b);}
 draw(hx,hi){this.layout();this.bounds();const c=this.c;c.clearRect(0,0,this.W,this.H);
  const css=getComputedStyle(document.body),line=css.getPropertyValue("--line").trim(),mut=css.getPropertyValue("--mut").trim();
  if(!this.ok){c.fillStyle=mut;c.font="12px sans-serif";c.fillText("no data for checked runs",this.pad.l,this.H/2);return;}
  c.font="10px -apple-system,sans-serif";c.textBaseline="middle";
  const yt=this.m.logy?this._logTicks():ticksFor(this.y0,this.y1,5);
  c.strokeStyle=line;c.fillStyle=mut;c.textAlign="right";
  for(const t of yt){let py=this.py(this.m.logy?Math.pow(10,t):t);if(py<this.pad.t-1||py>this.H-this.pad.b+1)continue;
   c.globalAlpha=.55;c.beginPath();c.moveTo(this.pad.l,py);c.lineTo(this.W-this.pad.r,py);c.stroke();
   c.globalAlpha=1;c.fillText(fmtY(this.m.logy?Math.pow(10,t):t),this.pad.l-6,py);}
  c.textAlign="center";c.textBaseline="top";
  for(const t of ticksFor(this.x0,this.x1,6))c.fillText(fmtX(t),this.px(t),this.H-this.pad.b+5);
  for(const s of this.S){const col=s.run.color,pts=s.pts;
   c.setLineDash(s.run.type=="selfplay"?[2,2]:[]);   // self-play series render tightly dotted
   let alpha=1,lw=st.ema?1.8:1.4;
   if(hi){if(hi.dim.has(s.run.k))alpha=.12;else if(s.run.k===hi.nearest){alpha=1;lw=2.9;}else alpha=.4;}
   if(!hi&&st.ema&&pts.length>3){c.globalAlpha=.2;c.strokeStyle=col;c.lineWidth=1;c.beginPath();
    pts.forEach((p,i)=>i?c.lineTo(this.px(p[0]),this.py(p[1])):c.moveTo(this.px(p[0]),this.py(p[1])));c.stroke();}
   c.globalAlpha=alpha;c.strokeStyle=col;c.lineWidth=lw;c.beginPath();
   pts.forEach((p,i)=>i?c.lineTo(this.px(p[0]),this.py(p[1])):c.moveTo(this.px(p[0]),this.py(p[1])));c.stroke();
   if(!st.ema&&!hi){c.fillStyle=col;pts.forEach(p=>{c.beginPath();c.arc(this.px(p[0]),this.py(p[1]),1.6,0,7);c.fill();});}
   c.globalAlpha=1;}
  if(hx!=null){c.strokeStyle=mut;c.globalAlpha=.7;c.lineWidth=1;c.beginPath();
   c.moveTo(hx,this.pad.t);c.lineTo(hx,this.H-this.pad.b);c.stroke();c.globalAlpha=1;}}
 _logTicks(){let a=[];for(let e=Math.floor(this.y0);e<=Math.ceil(this.y1);e++)a.push(e);return a;}
 hover(ev){const rct=this.cv.getBoundingClientRect(),mx=ev.clientX-rct.left,my=ev.clientY-rct.top;
  if(!this.ok||mx<this.pad.l||mx>this.W-this.pad.r){tip.style.display="none";this.draw();return;}
  const xv=this.x0+(mx-this.pad.l)/(this.W-this.pad.l-this.pad.r)*(this.x1-this.x0);
  const tol=(this.x1-this.x0)*0.004;
  let items=[],nearest=null,nd=1e18,dim=new Set();
  for(const s of this.S){const pts=s.pts,lastX=pts[pts.length-1][0],lastY=pts[pts.length-1][1];
   if(xv<=lastX+tol){let best=pts[0],bd=1e18;for(const p of pts){let d=Math.abs(p[0]-xv);if(d<bd){bd=d;best=p;}}
    let dd=Math.abs(my-this.py(best[1]));if(dd<nd){nd=dd;nearest=s.run.k;}
    items.push({run:s.run,val:best[1],ended:false});}
   else{items.push({run:s.run,val:lastY,ended:true});dim.add(s.run.k);}}
  items.sort((a,b)=>b.val-a.val);
  tip.innerHTML="<div class=hd>"+(st.x=="time"?"~"+xv.toFixed(1)+"h":"~cum "+Math.round(xv))+"</div>"+
   items.map(it=>{const on=it.run.k===nearest,cls=on?"hi":(it.ended?"dim":"");
    return "<div class='r "+cls+"'><span><span class=sw style='display:inline-block;background:"+it.run.color+
     ";border-radius:2px;margin-right:5px'></span>"+it.run.k+(it.ended?"<span class=fin>final</span>":"")+
     "</span><b>"+fmtY(it.val)+"</b></div>";}).join("");
  tip.style.display="block";
  let tx=ev.clientX+14,ty=ev.clientY+14;
  if(tx+tip.offsetWidth>innerWidth)tx=ev.clientX-tip.offsetWidth-14;
  if(ty+tip.offsetHeight>innerHeight)ty=ev.clientY-tip.offsetHeight-14;
  tip.style.left=tx+"px";tip.style.top=ty+"px";
  this.draw(mx,{nearest,dim});}
}
const tip=document.getElementById("tip");
const charts=METRICS.map(m=>new Chart(m));
function drawAll(){charts.forEach(c=>c.draw());}

/* ---- legend ---- */
const legend=document.getElementById("legend");
RUNS.forEach(r=>{const el=document.createElement("div");el.className="lg";el.dataset.k=r.k;
 el.innerHTML="<span class=sw style='background:"+r.color+"'></span>"+r.k+" <span class=ts>"+r.trainset+"</span>";
 el.onclick=()=>toggle(r.k);legend.appendChild(el);});
function syncUI(){
 document.querySelectorAll(".lg").forEach(e=>e.classList.toggle("off",!checked.has(e.dataset.k)));
 document.querySelectorAll("#summary tbody tr").forEach(e=>e.classList.toggle("off",!checked.has(e.dataset.k)));
 document.querySelectorAll("#xseg button").forEach(b=>b.classList.toggle("on",b.dataset.x==st.x));
 document.querySelectorAll("#emaseg button").forEach(b=>b.classList.toggle("on",(+b.dataset.e)==st.ema));
 document.querySelectorAll("#typeseg button").forEach(b=>b.classList.toggle("on",b.dataset.tp==st.tp));}
function toggle(k){checked.has(k)?checked.delete(k):checked.add(k);save();syncUI();drawAll();}
document.querySelectorAll("#summary tbody tr").forEach(tr=>tr.onclick=e=>{
 if(e.target.closest(".chip"))return; toggle(tr.dataset.k);});
document.getElementById("all").onclick=()=>{RUNS.forEach(r=>checked.add(r.k));save();syncUI();drawAll();};
document.getElementById("none").onclick=()=>{checked.clear();save();syncUI();drawAll();};
const tsq=document.getElementById("tsquick");
[...new Set(RUNS.map(r=>r.trainset))].forEach(ts=>{const b=document.createElement("button");
 b.className="mini";b.textContent="only "+ts;b.onclick=()=>{checked.clear();
  RUNS.filter(r=>r.trainset==ts).forEach(r=>checked.add(r.k));save();syncUI();drawAll();};tsq.appendChild(b);});
document.querySelectorAll("#xseg button").forEach(b=>b.onclick=()=>{st.x=b.dataset.x;save();syncUI();drawAll();});
document.querySelectorAll("#emaseg button").forEach(b=>b.onclick=()=>{st.ema=+b.dataset.e;save();syncUI();drawAll();});
function applyType(tp){st.tp=tp;checked.clear();
 RUNS.forEach(r=>{if(tp=="all"||r.type==tp)checked.add(r.k);});save();syncUI();drawAll();}
document.querySelectorAll("#typeseg button").forEach(b=>b.onclick=()=>applyType(b.dataset.tp));

/* ---- summary sort ---- */
document.querySelectorAll("#summary th[data-s]").forEach(th=>{th.onclick=e=>{e.stopPropagation();
 const k=th.dataset.s,tb=document.querySelector("#summary tbody"),num=th.classList.contains("num");
 const val=tr=>{let v=byKey[tr.dataset.k][k];return num?(parseFloat(v)||0):(""+v);};
 const asc=th._asc=!th._asc;
 [...tb.rows].sort((a,b)=>{let x=val(a),y=val(b);return (x<y?-1:x>y?1:0)*(asc?1:-1);}).forEach(r=>tb.appendChild(r));};});

/* ---- per-run data tables ---- */
const tcols=[["cum","s"],["h","t"]].concat(METRICS.map(m=>[m.title,m.key]));
const tw=document.getElementById("tables");
RUNS.forEach(r=>{if(!r.n)return;const d=document.createElement("details");
 let h="<summary><span class=sw style='background:"+r.color+";display:inline-block'></span> "+
  r.k+" — "+r.trainset+" · "+r.n+" marks · peak "+(r.peak?r.peak.toFixed(0):"—")+"</summary>";
 h+="<div class=tblwrap><table><thead><tr>"+tcols.map(c=>"<th class=num>"+c[0]+"</th>").join("")+"</tr></thead><tbody>";
 for(let i=0;i<r.n;i++){h+="<tr>"+tcols.map(c=>{let v=(c[1]=="s"||c[1]=="t")?r[c[1]][i]:r.m[c[1]][i];
  return "<td class=num>"+(v==null?"—":(typeof v=="number"?(Math.abs(v)>=100?v.toFixed(0):v.toFixed(3)):v))+"</td>";}).join("")+"</tr>";}
 h+="</tbody></table></div>";d.innerHTML=h;tw.appendChild(d);});

syncUI();drawAll();
let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(drawAll,120);});
</script></body></html>"""

if __name__ == "__main__":
    render()
