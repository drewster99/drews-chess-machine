#!/usr/bin/env python3
"""Regenerate v5-strength.html from the three probe files.

Supersedes build_html.py, which rebuilt the page by regex-mutating the previous
copy of itself and had no concept of run 3. This builds from the jsonl sources
only, so the page is reproducible from data rather than from its own last output.

Segment offsets (see HANDOFF.md section 3) — cum step = offset + raw step:

    run 1  new_ckpts.jsonl        offset       0
    run 2  new_ckpts_run2.jsonl   offset 268,506
    run 3  new_ckpts_run3.jsonl   offset 605,116

Run 3's offset is 268506 + 336610, the abort save the restart resumed from.
"""
import json
import os
import re

MON = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(MON, "v5-strength.html")

SEGMENTS = [
    (1, "new_ckpts.jsonl", 0),
    (2, "new_ckpts_run2.jsonl", 268506),
    (3, "new_ckpts_run3.jsonl", 605116),
    (4, "new_ckpts_run4.jsonl", 711449),
    # Run 5 resumed from run 4's step46000, so its offset is run 4's + 46000.
    (5, "new_ckpts_run5.jsonl", 757449),
]


def load():
    rows, seen = [], set()
    for run, fname, offset in SEGMENTS:
        path = os.path.join(MON, fname)
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = re.search(r"step(\d+)", d["model"])
            if not m:
                continue
            raw = int(m.group(1))
            # A checkpoint can be probed twice (retry after a failed probe);
            # keep the first reading so the series stays reproducible.
            key = (run, raw)
            if key in seen:
                continue
            seen.add(key)
            # Early run-1 probes predate avgRank / policy_logit_abs_max, so those
            # are optional and render as "·" rather than breaking the build.
            def opt(key, nd):
                v = d.get(key)
                return None if v is None else round(v, nd)

            rows.append({
                "run": run,
                "raw": raw,
                "cum": offset + raw,
                "pelo": round(d["pElo"], 1),
                "top1": round(100.0 * d["argmaxCorrect"] / d["n"], 1),
                "nll": round(d["nll"], 4),
                "logit": opt("policy_logit_abs_max", 1),
                "rank": opt("avgRank", 2),
            })
    rows.sort(key=lambda r: (r["cum"], r["run"]))
    return rows


rows = load()
records = json.load(open(os.path.join(MON, "records.json")))

# Match records to rows by cumulative step in thousands, the unit records.json
# stores. Tolerance covers the rounding the record was originally written with.
best_cum = records["best_pelo_step"] * 1000
low_cum = records["low_nll_step"] * 1000
for r in rows:
    r["recPelo"] = abs(r["cum"] - best_cum) < 600
    r["recNll"] = abs(r["cum"] - low_cum) < 600

latest = rows[-1]
prev = rows[-2] if len(rows) > 1 else latest
by_run = {}
for r in rows:
    by_run.setdefault(r["run"], []).append(r)

running = os.popen("pgrep -f 'MacOS/DrewsChessMachine' | head -1").read().strip()
status = "training" if running else "stopped"

payload = json.dumps({
    "rows": rows,
    "records": records,
    "status": status,
    "pid": running,
    "seams": [268.506, 605.116],
    "counts": {str(k): len(v) for k, v in by_run.items()},
})

HTML = r"""<title>v5 strength telemetry</title>
<style>
  :root {
    --paper:      #eef1f4;
    --card:       #f8fafb;
    --ink:        #16202b;
    --ink-soft:   #4a5a6a;
    --ink-faint:  #8496a6;
    --rule:       #d3dae1;
    --rule-soft:  #e3e8ed;
    --signal:     #2f6fb0;
    --counter:    #a8552f;
    --record:     #1a7a5c;
    --warn:       #b4532a;
    --grid:       rgba(22,32,43,.07);
    --shadow:     0 1px 2px rgba(22,32,43,.06), 0 4px 16px rgba(22,32,43,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper:      #0e151d;
      --card:       #141d27;
      --ink:        #e4ebf1;
      --ink-soft:   #9cb0c2;
      --ink-faint:  #64798c;
      --rule:       #26333f;
      --rule-soft:  #1c2731;
      --signal:     #6ba8e0;
      --counter:    #d98a5e;
      --record:     #46b990;
      --warn:       #e0885c;
      --grid:       rgba(228,235,241,.08);
      --shadow:     0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
    }
  }
  :root[data-theme="light"] {
    --paper:#eef1f4; --card:#f8fafb; --ink:#16202b; --ink-soft:#4a5a6a;
    --ink-faint:#8496a6; --rule:#d3dae1; --rule-soft:#e3e8ed; --signal:#2f6fb0;
    --counter:#a8552f; --record:#1a7a5c; --warn:#b4532a; --grid:rgba(22,32,43,.07);
    --shadow:0 1px 2px rgba(22,32,43,.06), 0 4px 16px rgba(22,32,43,.05);
  }
  :root[data-theme="dark"] {
    --paper:#0e151d; --card:#141d27; --ink:#e4ebf1; --ink-soft:#9cb0c2;
    --ink-faint:#64798c; --rule:#26333f; --rule-soft:#1c2731; --signal:#6ba8e0;
    --counter:#d98a5e; --record:#46b990; --warn:#e0885c; --grid:rgba(228,235,241,.08);
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
  }

  * { box-sizing: border-box; }

  .wrap {
    background: var(--paper);
    color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    padding: 28px 20px 64px;
  }
  .inner { max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 22px; }

  .mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; }
  .tnum { font-variant-numeric: tabular-nums; }

  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px 18px; }
  h1 {
    margin: 0; font-size: 25px; font-weight: 640; letter-spacing: -.021em;
    text-wrap: balance;
  }
  .ident {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12.5px; color: var(--ink-faint);
  }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 620; letter-spacing: .08em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 3px; border: 1px solid transparent;
  }
  .pill.on  { color: var(--record); border-color: color-mix(in srgb, var(--record) 40%, transparent);
              background: color-mix(in srgb, var(--record) 10%, transparent); }
  .pill.off { color: var(--warn);   border-color: color-mix(in srgb, var(--warn) 40%, transparent);
              background: color-mix(in srgb, var(--warn) 10%, transparent); }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .pill.on .dot { animation: pulse 2.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
  @media (prefers-reduced-motion: reduce) { .pill.on .dot { animation: none; } }

  .kpis {
    display: grid; gap: 1px; background: var(--rule);
    grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
    border: 1px solid var(--rule); border-radius: 5px; overflow: hidden;
    box-shadow: var(--shadow);
  }
  .kpi { background: var(--card); padding: 14px 16px 15px; display: flex; flex-direction: column; gap: 3px; }
  .kpi .lab {
    font-size: 10.5px; font-weight: 620; letter-spacing: .09em;
    text-transform: uppercase; color: var(--ink-faint);
  }
  .kpi .val {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 25px; font-weight: 550; letter-spacing: -.02em;
    font-variant-numeric: tabular-nums; line-height: 1.15;
  }
  .kpi .sub { font-size: 11.5px; color: var(--ink-soft); font-variant-numeric: tabular-nums; }
  .kpi.rec .val { color: var(--record); }

  .panel {
    background: var(--card); border: 1px solid var(--rule);
    border-radius: 5px; box-shadow: var(--shadow); overflow: hidden;
  }
  .phead {
    display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid var(--rule-soft);
  }
  .ptitle { font-size: 12.5px; font-weight: 650; letter-spacing: .03em; }
  .spacer { flex: 1 1 auto; }

  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: var(--ink-soft); }
  .legend i { display: inline-block; width: 16px; height: 2px; vertical-align: middle; margin-right: 5px; }

  #chart { display: block; width: 100%; height: 320px; }
  .chartbox { padding: 12px 8px 4px; }

  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  input[type="search"], select {
    font: inherit; font-size: 13px; color: var(--ink);
    background: var(--paper); border: 1px solid var(--rule);
    border-radius: 4px; padding: 5px 9px;
  }
  input[type="search"] { width: 150px; }
  input[type="search"]:focus-visible, select:focus-visible, button:focus-visible {
    outline: 2px solid var(--signal); outline-offset: 1px;
  }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
  .seg button {
    font: inherit; font-size: 12px; font-weight: 560; color: var(--ink-soft);
    background: var(--paper); border: 0; border-right: 1px solid var(--rule);
    padding: 5px 11px; cursor: pointer;
  }
  .seg button:last-child { border-right: 0; }
  .seg button[aria-pressed="true"] { background: var(--signal); color: #fff; }

  .scroll { max-height: 560px; overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    position: sticky; top: 0; z-index: 2; background: var(--card);
    font-size: 10.5px; font-weight: 640; letter-spacing: .08em; text-transform: uppercase;
    color: var(--ink-faint); text-align: right; white-space: nowrap;
    padding: 9px 12px; border-bottom: 1px solid var(--rule); cursor: pointer;
    user-select: none;
  }
  thead th:first-child, thead th.l { text-align: left; }
  thead th:hover { color: var(--ink); }
  thead th .ar { opacity: .45; font-size: 9px; margin-left: 3px; }
  tbody td {
    padding: 6px 12px; text-align: right; white-space: nowrap;
    border-bottom: 1px solid var(--rule-soft);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }
  tbody td:first-child, tbody td.l { text-align: left; }
  tbody tr:hover td { background: color-mix(in srgb, var(--signal) 7%, transparent); }
  tbody tr.rec td { background: color-mix(in srgb, var(--record) 9%, transparent); }
  tbody tr.rec:hover td { background: color-mix(in srgb, var(--record) 15%, transparent); }
  .tag {
    display: inline-block; font-family: inherit; font-size: 9.5px; font-weight: 700;
    letter-spacing: .06em; padding: 1px 5px; border-radius: 2px; margin-left: 6px;
    color: var(--record); border: 1px solid color-mix(in srgb, var(--record) 45%, transparent);
  }
  .runcell { color: var(--ink-faint); font-size: 11.5px; }
  .up { color: var(--record); } .dn { color: var(--warn); }
  .bar { display: inline-block; height: 3px; border-radius: 2px; background: var(--signal); opacity: .55; vertical-align: middle; }

  .foot { font-size: 11.5px; color: var(--ink-faint); line-height: 1.7; }
  .foot code { font-family: ui-monospace, Menlo, monospace; color: var(--ink-soft); }
  .count { font-size: 11.5px; color: var(--ink-faint); font-variant-numeric: tabular-nums; }
  .notes { font-size: 13.5px; color: var(--ink-soft); max-width: 74ch; }
  .notes b { color: var(--ink); font-weight: 620; }
  .notes p { margin: 0 0 9px; }
</style>

<div class="wrap"><div class="inner">

  <header>
    <h1>v5 strength telemetry</h1>
    <span class="pill" id="statuspill"><span class="dot"></span><span id="statustext"></span></span>
    <span class="spacer"></span>
    <span class="ident" id="ident"></span>
  </header>

  <div class="kpis" id="kpis"></div>

  <div class="panel">
    <div class="phead">
      <span class="ptitle">Strength over cumulative training steps</span>
      <span class="seg" id="rangeseg">
        <button data-r="all" aria-pressed="true">All</button>
        <button data-r="1" aria-pressed="false">Run 1</button>
        <button data-r="2" aria-pressed="false">Run 2</button>
        <button data-r="3" aria-pressed="false">Run 3</button>
        <button data-r="50" aria-pressed="false">Last 50</button>
      </span>
      <span class="spacer"></span>
      <span class="legend">
        <span><i style="background:var(--signal)"></i>pElo (10-pt EMA)</span>
        <span><i style="background:var(--counter)"></i>policy NLL</span>
        <span><i style="background:var(--ink-faint);height:0;border-top:2px dashed var(--ink-faint)"></i>restart seam</span>
      </span>
    </div>
    <div class="chartbox"><canvas id="chart"></canvas></div>
  </div>

  <div class="panel">
    <div class="phead">
      <span class="ptitle">All checkpoints</span>
      <span class="controls">
        <span class="seg" id="runseg">
          <button data-run="all" aria-pressed="true">All</button>
          <button data-run="1" aria-pressed="false">Run 1</button>
          <button data-run="2" aria-pressed="false">Run 2</button>
          <button data-run="3" aria-pressed="false">Run 3</button>
        </span>
        <select id="viewsel">
          <option value="all">Every checkpoint</option>
          <option value="recent">Most recent 25</option>
          <option value="rec">Records only</option>
          <option value="top">Top 25 by pElo</option>
        </select>
        <input type="search" id="q" placeholder="step…" aria-label="Filter by step number">
      </span>
      <span class="spacer"></span>
      <span class="count" id="count"></span>
    </div>
    <div class="scroll">
      <table>
        <thead><tr>
          <th class="l" data-k="cum">cum step<span class="ar"></span></th>
          <th class="l" data-k="run">run<span class="ar"></span></th>
          <th data-k="raw">raw<span class="ar"></span></th>
          <th data-k="pelo">pElo<span class="ar"></span></th>
          <th data-k="d">Δ<span class="ar"></span></th>
          <th data-k="top1">top1 %<span class="ar"></span></th>
          <th data-k="nll">NLL<span class="ar"></span></th>
          <th data-k="rank">avg rank<span class="ar"></span></th>
          <th data-k="logit">logit max<span class="ar"></span></th>
        </tr></thead>
        <tbody id="tb"></tbody>
      </table>
    </div>
  </div>

  <div class="notes" id="notes"></div>
  <p class="foot" id="foot"></p>

</div></div>

<script>
const DATA = __PAYLOAD__;
const R = DATA.rows;

// Delta against the previous checkpoint in the same run, so the seam between
// runs never manufactures a spurious jump.
const lastOf = {};
for (const r of R) {
  const p = lastOf[r.run];
  r.d = p === undefined ? null : +(r.pelo - p).toFixed(1);
  lastOf[r.run] = r.pelo;
}

const fmtK = c => (c / 1000).toFixed(1) + "k";
const el = id => document.getElementById(id);

/* ---------- header + KPIs ---------- */
const last = R[R.length - 1];
const rec = DATA.records;
el("statuspill").className = "pill " + (DATA.status === "training" ? "on" : "off");
el("statustext").textContent = DATA.status === "training" ? "training · pid " + DATA.pid : "stopped";
el("ident").textContent = "8.45M params · 5×7×7 @128 · WDL head · wide battery n=4435";

const peloVals = R.map(r => r.pelo);
el("kpis").innerHTML = [
  ["latest pElo", last.pelo.toFixed(1), "cum " + fmtK(last.cum) + " · run " + last.run, false],
  ["best pElo", rec.best_pelo.toFixed(1), "cum " + rec.best_pelo_step.toFixed(1) + "k", true],
  ["lowest NLL", rec.low_nll.toFixed(4), "cum " + rec.low_nll_step.toFixed(1) + "k", true],
  ["latest NLL", last.nll.toFixed(4), "top1 " + last.top1.toFixed(1) + "%", false],
  ["checkpoints", String(R.length), Object.entries(DATA.counts).map(([k, v]) => "r" + k + " " + v).join(" · "), false],
  ["logit max", last.logit === null ? "\u00b7" : last.logit.toFixed(1), "peak seen " + Math.max(...R.filter(r => r.logit !== null).map(r => r.logit)).toFixed(0), false],
].map(([l, v, s, isRec]) =>
  `<div class="kpi${isRec ? " rec" : ""}"><span class="lab">${l}</span><span class="val">${v}</span><span class="sub">${s}</span></div>`
).join("");

/* ---------- chart ---------- */
const cv = el("chart"), cx = cv.getContext("2d");
function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
let range = "all";

function chartRows() {
  if (range === "all") return R;
  if (range === "50") return R.slice(-50);
  return R.filter(r => String(r.run) === range);
}

// Round a raw axis span up to a readable increment (1/2/2.5/5 x a power of ten).
function niceStep(span, target) {
  const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
}

function draw() {
  const S = chartRows();
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cx.clearRect(0, 0, w, h);
  if (!S.length) return;

  const L = 54, Rp = 56, T = 14, B = 30;
  const pw = w - L - Rp, ph = h - T - B;
  if (pw <= 0 || ph <= 0) return;

  // Axes fit the visible slice, so zooming into a short run stays readable
  // instead of collapsing onto a full-history scale.
  const cums = S.map(r => r.cum);
  let xmin = Math.min(...cums), xmax = Math.max(...cums);
  if (xmin === xmax) { xmin -= 500; xmax += 500; }
  const xp = (xmax - xmin) * .02; xmin -= xp; xmax += xp;

  const pe = S.map(r => r.pelo);
  let ymin = Math.min(...pe), ymax = Math.max(...pe);
  const yp = Math.max(20, (ymax - ymin) * .12); ymin -= yp; ymax += yp;

  const nl = S.map(r => r.nll);
  let nmin = Math.min(...nl), nmax = Math.max(...nl);
  const np = Math.max(.02, (nmax - nmin) * .12); nmin -= np; nmax += np;

  const X = v => L + (v - xmin) / (xmax - xmin) * pw;
  const Y = v => T + (1 - (v - ymin) / (ymax - ymin)) * ph;
  const YN = v => T + (1 - (v - nmin) / (nmax - nmin)) * ph;

  cx.font = "10px ui-monospace, Menlo, monospace";

  const ys = niceStep(ymax - ymin, 6);
  cx.strokeStyle = css("--grid"); cx.lineWidth = 1;
  cx.textAlign = "right"; cx.textBaseline = "middle";
  for (let v = Math.ceil(ymin / ys) * ys; v <= ymax; v += ys) {
    const y = Math.round(Y(v)) + .5;
    cx.beginPath(); cx.moveTo(L, y); cx.lineTo(L + pw, y); cx.stroke();
    cx.fillStyle = css("--ink-faint"); cx.fillText(v.toFixed(0), L - 8, y);
  }

  const ns = niceStep(nmax - nmin, 5);
  cx.textAlign = "left"; cx.fillStyle = css("--counter");
  for (let v = Math.ceil(nmin / ns) * ns; v <= nmax; v += ns) cx.fillText(v.toFixed(ns < .05 ? 3 : 2), L + pw + 8, YN(v));

  const xt = niceStep(xmax - xmin, 6);
  cx.fillStyle = css("--ink-faint"); cx.textAlign = "center"; cx.textBaseline = "top";
  for (let v = Math.ceil(xmin / xt) * xt; v <= xmax; v += xt) cx.fillText((v / 1000).toFixed(xt < 1000 ? 1 : 0) + "k", X(v), T + ph + 8);

  cx.save(); cx.setLineDash([4, 4]); cx.strokeStyle = css("--ink-faint"); cx.globalAlpha = .8;
  for (const s of DATA.seams) {
    const v = s * 1000; if (v < xmin || v > xmax) continue;
    const x = Math.round(X(v)) + .5;
    cx.beginPath(); cx.moveTo(x, T); cx.lineTo(x, T + ph); cx.stroke();
  }
  cx.restore();

  const dense = S.length > 120;
  cx.fillStyle = css("--signal"); cx.globalAlpha = dense ? .18 : .55;
  for (const r of S) { cx.beginPath(); cx.arc(X(r.cum), Y(r.pelo), dense ? 1.4 : 2.8, 0, 6.284); cx.fill(); }
  cx.globalAlpha = 1;

  cx.strokeStyle = css("--counter"); cx.lineWidth = 1.4; cx.globalAlpha = .55;
  cx.beginPath();
  S.forEach((r, i) => { const x = X(r.cum), y = YN(r.nll); i ? cx.lineTo(x, y) : cx.moveTo(x, y); });
  cx.stroke(); cx.globalAlpha = 1;

  // EMA per run so a restart seam never smears across two lineages.
  cx.strokeStyle = css("--signal"); cx.lineWidth = 1.9;
  cx.lineJoin = "round"; cx.lineCap = "round";
  const a = 2 / 11;
  for (const run of [1, 2, 3]) {
    const seg = S.filter(r => r.run === run);
    if (seg.length < 2) continue;
    let e = seg[0].pelo; cx.beginPath();
    seg.forEach((r, i) => { e = i ? e + a * (r.pelo - e) : e; const x = X(r.cum), y = Y(e); i ? cx.lineTo(x, y) : cx.moveTo(x, y); });
    cx.stroke();
  }

  for (const r of S) {
    if (!r.recPelo && !r.recNll) continue;
    cx.strokeStyle = css("--record"); cx.lineWidth = 1.6;
    cx.beginPath(); cx.arc(X(r.cum), r.recPelo ? Y(r.pelo) : YN(r.nll), 4.5, 0, 6.284); cx.stroke();
  }

  const tail = S[S.length - 1];
  cx.fillStyle = css("--signal");
  cx.beginPath(); cx.arc(X(tail.cum), Y(tail.pelo), 3.4, 0, 6.284); cx.fill();
}

el("rangeseg").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  range = b.dataset.r;
  el("rangeseg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  draw();
});

/* ---------- table ---------- */
let sortK = "cum", sortDir = 1, runF = "all", view = "all", q = "";

function rowsFor() {
  let out = R.slice();
  if (runF !== "all") out = out.filter(r => String(r.run) === runF);
  if (q) out = out.filter(r => String(r.raw).includes(q) || fmtK(r.cum).includes(q));
  if (view === "rec") out = out.filter(r => r.recPelo || r.recNll);
  else if (view === "top") out = out.sort((x, y) => y.pelo - x.pelo).slice(0, 25);
  else if (view === "recent") out = out.slice(-25);
  out.sort((x, y) => {
    const A = x[sortK], B = y[sortK];
    if (A === null) return 1; if (B === null) return -1;
    return (A > B ? 1 : A < B ? -1 : 0) * sortDir;
  });
  return out;
}

function render() {
  const rows = rowsFor();
  const lo = Math.min(...R.map(r => r.pelo)), hi = Math.max(...R.map(r => r.pelo));
  el("tb").innerHTML = rows.map(r => {
    const tags = (r.recPelo ? '<span class="tag">pElo REC</span>' : "") + (r.recNll ? '<span class="tag">NLL REC</span>' : "");
    const d = r.d === null ? "·" : `<span class="${r.d > 0 ? "up" : r.d < 0 ? "dn" : ""}">${r.d > 0 ? "+" : ""}${r.d.toFixed(1)}</span>`;
    const bw = Math.max(2, Math.round((r.pelo - lo) / (hi - lo) * 46));
    return `<tr class="${r.recPelo || r.recNll ? "rec" : ""}">
      <td class="l">${fmtK(r.cum)}${tags}</td>
      <td class="l runcell">${r.run}</td>
      <td>${r.raw.toLocaleString()}</td>
      <td>${r.pelo.toFixed(1)} <span class="bar" style="width:${bw}px"></span></td>
      <td>${d}</td><td>${r.top1.toFixed(1)}</td><td>${r.nll.toFixed(4)}</td>
      <td>${r.rank === null ? "\u00b7" : r.rank.toFixed(2)}</td><td>${r.logit === null ? "\u00b7" : r.logit.toFixed(1)}</td></tr>`;
  }).join("");
  el("count").textContent = rows.length + " of " + R.length + " shown";
  document.querySelectorAll("thead th").forEach(th => {
    th.querySelector(".ar").textContent = th.dataset.k === sortK ? (sortDir > 0 ? "▲" : "▼") : "";
  });
}

document.querySelectorAll("thead th").forEach(th => th.addEventListener("click", () => {
  const k = th.dataset.k;
  if (k === sortK) sortDir *= -1; else { sortK = k; sortDir = (k === "cum" || k === "run" || k === "raw") ? 1 : -1; }
  render();
}));
el("runseg").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  runF = b.dataset.run;
  el("runseg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  render();
});
el("viewsel").addEventListener("change", e => { view = e.target.value; render(); });
el("q").addEventListener("input", e => { q = e.target.value.trim(); render(); });

/* ---------- prose ---------- */
el("notes").innerHTML = __NOTES__;
el("foot").textContent = R.length + " checkpoints · cum steps "
  + fmtK(R[0].cum) + "–" + fmtK(last.cum)
  + " · restart seams at " + DATA.seams.map(s => s.toFixed(1) + "k").join(" and ")
  + " · probes are deterministic on a fixed 4435-puzzle battery · 10-pt EMA α=0.182";

render(); draw();
addEventListener("resize", draw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
</script>
"""

NOTES = (
    "<p><b>Reading it.</b> Each point is one checkpoint on the fixed 4435-puzzle battery. "
    "Single points swing ±80–100 pElo, so the per-run EMA is the trend and an individual "
    "row is not. <b>NLL</b> (right axis, lower is better) runs opposite to pElo.</p>"
    "<p><b>Run 1</b> — original v5 training to step 268,506: volatile early, a rough trough "
    "76k–104k with deep single-checkpoint craters, recovery to a run-1 high near 129k, then a "
    "choppy band. Stopped cleanly for a macOS update.</p>"
    "<p><b>Run 2</b> — corpus-replay warm start, resumed at the exact corpus position. Opened "
    "~1650 and ratcheted into a <b>1740–1770</b> peak cluster around cum 520–550k, setting both "
    "standing records within ~2k steps of each other, then settled into a lower ~1675-centre band. "
    "Stopped by hand at cum 605.1k.</p>"
    "<p><b>Run 3 — complete.</b> Jul 28 22:31 → Aug 2 07:23, stopped by its own epoch budget at "
    "step 106,333 (13.7M games, 909M positions) with a clean <code>nextGame=0, shard=0, epoch=5</code> "
    "resume point. Restarted from the latest checkpoint, never the best-scoring one — the standing "
    "policy. The corpus stream resumed exactly, verified by <code>nextGame</code> continuing "
    "7,212,496 → 7,340,950 across the seam.</p>"
    "<p><b>It held strength without advancing it.</b> Mean pElo 1671 over the first half against "
    "1655 over the second; band 1433–1735 with seven sub-1600 craters, every one recovering within "
    "one to four checkpoints. Its best checkpoint arrived roughly nine hours in "
    "(<b>1735.4 pElo / 1.8630 NLL</b> at cum 614.1k) and was never beaten across the remaining four "
    "days. <b>No record fell</b> — the champion is still run 2's cum 538.5k.</p>"
    "<p>The one metric with direction was <b>policy logit-abs-max, which rose 214.9 → 316.1</b> across "
    "the run — past run 2's all-time peak — with no strength cost, continuing the decoupling that has "
    "held since run 1. Only checkpoints actually written since the restart appear here: run-2's "
    "enumerated files remain on disk and were progressively overwritten as run 3 climbed, so a "
    "checkpoint was admitted only when its mtime postdated the restart <i>and</i> its modelID matched "
    "the one minted at restart (<code>20260729-1-VZ2j</code>).</p>"
)

open(OUT, "w").write(
    HTML.replace("__PAYLOAD__", payload).replace("__NOTES__", json.dumps(NOTES))
)
counts = " / ".join("run%d %d" % (k, len(by_run[k])) for k in sorted(by_run))
print("wrote %s\n  %d checkpoints (%s)\n  latest %.1f @ cum %.1fk · records pElo %.1f@%gk nll %.4f@%gk · %s"
      % (OUT, len(rows), counts,
         latest["pelo"], latest["cum"] / 1000, records["best_pelo"], records["best_pelo_step"],
         records["low_nll"], records["low_nll_step"], status))
