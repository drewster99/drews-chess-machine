import json, numpy as np, html, sys
import os
S = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(f"{S}/relu_inputs_corpus.json"))
# The start position is position 0 only in the bot-position run (positions() prepends it); the corpus
# sample has no start position, so the start-position row reads the bot run.
START = {r["label"]: r["value"]["start_wdl"] for r in json.load(open(f"{S}/relu_inputs_bot.json"))["results"]}
NPOS, NGAMES = d["positions"], d["games"]
R = {r["label"]: r for r in d["results"]}
nets = [("fatconv", "fatconv", "1 block · 15×15 · 98 ch", "20261004-15-Pm6B"),
        ("R7", "R7 (baseline s1)", "3 blocks · 7×7 · 128 ch", "20261002-2-5tKN"),
        ("R8", "R8 (baseline s2)", "3 blocks · 7×7 · 128 ch", "20261002-4-T79u")]
SITE_NAMES = {"blocks.0.bn1": "block 1 · conv 1", "blocks.0.bn2": "block 1 · conv 2", "blocks.1.bn1": "block 2 · conv 1",
              "blocks.1.bn2": "block 2 · conv 2", "blocks.2.bn1": "block 3 · conv 1", "blocks.2.bn2": "block 3 · conv 2",
              "tower_final_bn": "tower end", "policy.pre_bn": "policy pre-conv", "value.bn": "value conv", "value.fc1": "value FC1 (no BN)"}
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"
def pct(x): return f"{100*x:.1f}%"
def sgn(x, places): return ("−" if x < 0 else "+") + f"{abs(x):.{places}f}"

def site_rows(r):
    rows = []
    for site, s in r["relu_inputs"].items():
        m = np.array(s["mean"]); fp = np.array(s["frac_positive"]); fn = np.array(s["frac_negative"])
        nn = int((fn == 0).sum()); np_ = int((fp == 0).sum()); gt1 = int((m > 1).sum()); ltm1 = int((m < -1).sum())
        def cell(v, bad): return f'<td class="num"><span class="chip {bad}">{v}</span></td>' if v else f'<td class="num zero">0</td>'
        rows.append(f"""<tr><th scope="row">{SITE_NAMES[site]}</th><td class="num">{s['channels']}</td>
<td class="num">{f3(s['overall_mean'])}</td><td class="num">{f3(m.min())}</td><td class="num">{f3(float(np.median(m)))}</td><td class="num">{f3(m.max())}</td>
{cell(gt1,'warn')}{cell(nn,'bad')}{cell(np_,'warn')}{cell(ltm1,'note')}
<td class="num">{pct(fp.min())}</td><td class="num">{pct(fp.max())}</td><td class="num">{f2(s['overall_min'])}</td><td class="num">{f2(s['overall_max'])}</td></tr>""")
    return "\n".join(rows)

def strip_svg(r):
    sites = list(r["relu_inputs"].keys())
    W, rowh, left, right = 760, 30, 150, 20
    lo, hi = -3.0, 1.5
    X = lambda v: left + (min(max(v, lo), hi) - lo) / (hi - lo) * (W - left - right)
    H = rowh * len(sites) + 40
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Per-channel mean of ReLU inputs, {r["label"]}" class="strip">']
    for t in [-3, -2, -1, 0, 1]:
        x = X(t); cls = "axis0" if t == 0 else ("axis1" if t == 1 else "grid")
        out.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{H-24}" class="{cls}"/>')
        out.append(f'<text x="{x:.1f}" y="{H-8}" class="tick" text-anchor="middle">{t:+d}</text>' if t else f'<text x="{x:.1f}" y="{H-8}" class="tick" text-anchor="middle">0</text>')
    for i, site in enumerate(sites):
        y = 20 + i * rowh
        out.append(f'<text x="{left-10}" y="{y+4}" class="lab" text-anchor="end">{SITE_NAMES[site]}</text>')
        m = np.array(r["relu_inputs"][site]["mean"])
        rng = np.random.default_rng(i)
        for v in m:
            jy = y + rng.uniform(-7, 7)
            cls = "dot hi" if v > 1 else ("dot lo" if v < -1 else "dot")
            out.append(f'<circle cx="{X(v):.1f}" cy="{jy:.1f}" r="2.4" class="{cls}"/>')
    out.append("</svg>")
    return "".join(out)

def policy_table():
    rows = []
    defs = [("All 4,864 logits: min", lambda p: f2(p["all_min"])), ("All 4,864 logits: max", lambda p: f2(p["all_max"])), ("All 4,864 logits: mean", lambda p: f3(p["all_mean"])),
            ("Legal moves: min", lambda p: f2(p["legal_min"])), ("Legal moves: max", lambda p: f2(p["legal_max"])), ("Legal moves: mean", lambda p: f3(p["legal_mean"])),
            ("Illegal moves: min", lambda p: f2(p["illegal_min"])), ("Illegal moves: max", lambda p: f2(p["illegal_max"])), ("Illegal moves: mean", lambda p: f3(p["illegal_mean"])),
            ("Per position: legal mean, median", lambda p: f3(p["per_position_legal_mean_pct"][2])),
            ("Per position: legal spread (std), median", lambda p: f3(p["per_position_legal_spread_median"])),
            ("Per position: best legal, median", lambda p: f2(p["per_position_legal_max_pct"][2])),
            ("Best legal − best illegal: smallest", lambda p: f2(p["legal_top_minus_illegal_max_min"])),
            ("Best legal − best illegal: median", lambda p: f2(p["legal_top_minus_illegal_max_pct"][2])),
            ("Softmax mass on illegal moves (all-move softmax): mean", lambda p: pct(p["illegal_softmax_mass_mean"])),
            ("Softmax mass on illegal moves: worst position", lambda p: pct(p["illegal_softmax_mass_max"]))]
    for name, fn in defs:
        rows.append("<tr><th scope='row'>" + name + "</th>" + "".join(f"<td class='num'>{fn(R[k]['policy'])}</td>" for k, *_ in nets) + "</tr>")
    return "\n".join(rows)

def value_table():
    rows = []
    for i, slot in enumerate(["win", "draw", "loss"]):
        for stat, key, f in (("min", "slot_min", f3), ("max", "slot_max", f3), ("mean", "slot_mean", f3)):
            rows.append(f"<tr><th scope='row'>{slot} logit: {stat}</th>" + "".join(f"<td class='num'>{f(R[k]['value'][key][i])}</td>" for k, *_ in nets) + "</tr>")
    rows.append("<tr><th scope='row'>all three logits: min / max</th>" + "".join(f"<td class='num'>{f3(R[k]['value']['all_min'])} / {f3(R[k]['value']['all_max'])}</td>" for k, *_ in nets) + "</tr>")
    rows.append("<tr><th scope='row'>shared offset (mean of 3), median</th>" + "".join(f"<td class='num'>{f3(R[k]['value']['shared_pct'][2])}</td>" for k, *_ in nets) + "</tr>")
    rows.append("<tr><th scope='row'>mean probability W / D / L</th>" + "".join("<td class='num'>" + " / ".join(f"{x:.3f}" for x in R[k]['value']['prob_mean']) + "</td>" for k, *_ in nets) + "</tr>")
    rows.append("<tr><th scope='row'>start position W / D / L</th>" + "".join("<td class='num'>" + " / ".join(f"{x:.3f}" for x in START[k]) + "</td>" for k, *_ in nets) + "</tr>")
    return "\n".join(rows)

def counts(label):
    r = R[label]["relu_inputs"]
    gt1 = sum(int((np.array(s["mean"]) > 1).sum()) for s in r.values())
    gt1_bn = sum(int((np.array(s["mean"]) > 1).sum()) for k, s in r.items() if k != "value.fc1")
    nn = sum(int((np.array(s["frac_negative"]) == 0).sum()) for s in r.values())
    silent = int((np.array(r["value.fc1"]["frac_positive"]) == 0).sum())
    bnmax = max(max(s["mean"]) for k, s in r.items() if k != "value.fc1")
    return gt1, gt1_bn, nn, silent, bnmax
C = {k: counts(k) for k, *_ in nets}

head = """<title>ReLU Inputs and Logits</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column of findings, then full-width data tables per net; instrument-panel restraint */
:root{
  --bg:#f5f6f4; --surface:#ffffff; --ink:#1b2422; --muted:#5d6966; --rule:#d9dedb; --accent:#1f6f66;
  --warn:#9a6200; --warn-bg:#fbefd8; --bad:#a3352b; --bad-bg:#f8e1dd; --note:#3c5f8c; --note-bg:#e2ebf6;
  --dot:#1f6f66; --dot-hi:#c07a00; --dot-lo:#3c5f8c;
  --display:"Newsreader",Georgia,"Times New Roman",serif; --body:"IBM Plex Sans",system-ui,-apple-system,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#121816; --surface:#1a2220; --ink:#e3e9e6; --muted:#9aa8a4; --rule:#2c3633; --accent:#5cc2b4;
  --warn:#f0b54a; --warn-bg:#3a2c10; --bad:#f08a7e; --bad-bg:#3d1e1a; --note:#8fb4e6; --note-bg:#1c2a3d;
  --dot:#5cc2b4; --dot-hi:#f0b54a; --dot-lo:#8fb4e6; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121816; --surface:#1a2220; --ink:#e3e9e6; --muted:#9aa8a4; --rule:#2c3633; --accent:#5cc2b4;
  --warn:#f0b54a; --warn-bg:#3a2c10; --bad:#f08a7e; --bad-bg:#3d1e1a; --note:#8fb4e6; --note-bg:#1c2a3d;
  --dot:#5cc2b4; --dot-hi:#f0b54a; --dot-lo:#8fb4e6; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.55;padding-inline:20px;padding-block:28px 64px}
.wrap{max-width:1080px;margin-inline:auto;display:grid;gap:36px}
h1{font-family:var(--display);font-weight:600;font-size:2.1rem;line-height:1.15;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-weight:600;font-size:1.45rem;margin:0 0 6px;text-wrap:balance}
h3{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 8px;font-weight:600}
p{margin:0;max-width:68ch} .lede{color:var(--muted);max-width:72ch}
section{display:grid;gap:14px}
.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.answer{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;display:grid;gap:6px;min-width:0}
.answer .q{font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
.answer .a{font-family:var(--display);font-size:1.25rem;font-weight:600}
.answer p{font-size:.92rem}
.tablebox{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
thead th{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600;vertical-align:bottom;white-space:normal;min-width:64px}
tbody th{font-weight:500}
td.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
td.zero{color:var(--muted)}
tr:last-child th,tr:last-child td{border-bottom:none}
.chip{display:inline-block;padding:1px 7px;border-radius:999px;font-weight:500}
.chip.warn{background:var(--warn-bg);color:var(--warn)} .chip.bad{background:var(--bad-bg);color:var(--bad)} .chip.note{background:var(--note-bg);color:var(--note)}
.net{display:grid;gap:12px}
.net header{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px}
.net header .meta{color:var(--muted);font-family:var(--mono);font-size:.8rem}
svg.strip{width:100%;height:auto;max-width:100%;background:var(--surface);border:1px solid var(--rule);border-radius:6px}
svg .grid{stroke:var(--rule);stroke-width:1} svg .axis0{stroke:var(--muted);stroke-width:1} svg .axis1{stroke:var(--dot-hi);stroke-width:1;stroke-dasharray:3 3}
svg .tick,svg .lab{fill:var(--muted);font-family:var(--mono);font-size:11px}
svg .dot{fill:var(--dot);fill-opacity:.55} svg .dot.hi{fill:var(--dot-hi);fill-opacity:.9} svg .dot.lo{fill:var(--dot-lo);fill-opacity:.85}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:.82rem}
.legend span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.legend .l1::before{background:var(--dot)} .legend .l2::before{background:var(--dot-hi)} .legend .l3::before{background:var(--dot-lo)}
ul{margin:0;padding-left:20px;display:grid;gap:6px;max-width:74ch}
code{font-family:var(--mono);font-size:.85em}
pre.mermaid{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:16px;overflow-x:auto;margin:0;text-align:center}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.cols > *{min-width:0}
td.wrap,th.wrap{white-space:normal;min-width:180px}
.kicker{font-family:var(--mono);font-size:.78rem;color:var(--accent);letter-spacing:.04em}
</style>"""


ARCH_ROWS = [
 ("Input encoding", "basic30: 30 planes × 8 × 8, from the side to move's view", "same"),
 ("Stem", "15×15 conv, 30 → 98, no bias; batch norm; no activation", "7×7 conv, 30 → 128, no bias; batch norm; no activation"),
 ("Residual blocks", "1", "3"),
 ("Block width", "98 channels", "128 channels"),
 ("Block layout (pre-activation)", "BN → ReLU → 15×15 conv → BN → ReLU → 15×15 conv, plus skip", "BN → ReLU → 7×7 conv → BN → ReLU → 7×7 conv, plus skip"),
 ("Skip merge", "plain add (clean_add)", "same"),
 ("Block output norm", "layer norm over channels, per square", "same"),
 ("Squeeze-excitation", "none", "none"),
 ("ReZero scaling", "off", "off"),
 ("Dropout", "0", "0"),
 ("Tower end", "batch norm → ReLU (feeds both heads)", "same"),
 ("Policy head", "intermediate_conv, 128-wide pre-conv (details below)", "same"),
 ("Value head", "W/D/L: 16-channel conv → FC 1,024 → 128 → 3 (details below)", "same"),
 ("Compute precision", "bf16; both head tails in fp32", "same"),
 ("Policy tail precision (run flag)", "fp32_from_pre_bn", "same"),
 ("Head final-layer init", "He-normal (policy and value)", "same"),
 ("Value draw prior", "0.75 (final bias [0, ln 6, 0])", "same"),
 ("Trainable parameters", "5,140,071", "5,167,983"),
 ("Conv/FC MACs per position", "320.6M (92.4M not on padding)", "322.3M (199.7M not on padding)"),
 ("Run ModelID", "20261004-15-Pm6B", "R7 20261002-2-5tKN · R8 20261002-4-T79u"),
]

def arch_table():
    rows = "".join(f"<tr><th scope='row' class='wrap'>{a}</th><td class='wrap'>{b}</td><td class='wrap'>{c}</td></tr>" for a, b, c in ARCH_ROWS)
    return f"<div class='tablebox'><table><thead><tr><th></th><th>fatconv</th><th>R7 / R8</th></tr></thead><tbody>{rows}</tbody></table></div>"

POLICY_LAYERS = [
 ("Trunk output", "tower-end BN → ReLU", "C × 8 × 8", "—", "—", "—", "bf16"),
 ("Pre-conv", "1×1 conv, C → 128, no bias", "128 × 8 × 8", "12,544 / 16,384", "802,816 / 1,048,576", "He-normal", "bf16"),
 ("Pre batch norm", "per-channel BN (running stats at play time)", "128 × 8 × 8", "256 (+256 running)", "—", "γ = 1, β = 0", "fp32"),
 ("ReLU", "max(x, 0)", "128 × 8 × 8", "—", "—", "—", "fp32"),
 ("Final projection", "1×1 conv, 128 → 76, with bias", "76 × 8 × 8", "9,728 + 76", "622,592", "He-normal weights, bias 0", "fp32"),
 ("Logits", "flatten: index = channel · 64 + row · 8 + col", "4,864", "—", "—", "—", "fp32"),
 ("Softmax (play)", "CPU: mask illegal moves → temperature → softmax over legal moves → sample", "legal moves", "—", "—", "—", "fp32"),
 ("Softmax (training)", "softmax over all 4,864; illegal mass is penalized separately", "4,864", "—", "—", "—", "fp32"),
]
VALUE_LAYERS = [
 ("Trunk output", "tower-end BN → ReLU", "C × 8 × 8", "—", "—", "—", "bf16"),
 ("Value conv", "1×1 conv, C → 16, no bias", "16 × 8 × 8", "1,568 / 2,048", "100,352 / 131,072", "He-normal", "bf16"),
 ("Value batch norm", "per-channel BN (running stats at play time)", "16 × 8 × 8", "32 (+32 running)", "—", "γ = 1, β = 0", "bf16"),
 ("ReLU", "max(x, 0)", "16 × 8 × 8", "—", "—", "—", "bf16"),
 ("Flatten", "index = channel · 64 + square", "1,024", "—", "—", "—", "bf16"),
 ("FC1", "dense 1,024 → 128, with bias", "128", "131,072 + 128", "131,072", "He-normal weights, bias 0", "bf16"),
 ("ReLU", "max(x, 0)", "128", "—", "—", "—", "bf16"),
 ("FC2", "dense 128 → 3, with bias", "3 logits: win, draw, loss", "384 + 3", "384", "He-normal weights, bias [0, ln 6, 0]", "fp32"),
 ("Softmax", "softmax over the 3 logits → p_win, p_draw, p_loss", "3", "—", "—", "—", "fp32"),
 ("Scalar value", "v = p_win − p_loss, in [−1, +1]; used by move choice and as the policy baseline", "1", "—", "—", "—", "fp32"),
]
def layer_table(rows):
    body = "".join("<tr><th scope='row'>" + r[0] + "</th>" + "".join(f"<td class='wrap'>{c}</td>" if i in (0, 4) else f"<td class='num'>{c}</td>" if i in (2, 3) else f"<td>{c}</td>" for i, c in enumerate(r[1:])) + "</tr>" for r in rows)
    return ("<div class='tablebox'><table><thead><tr><th>Step</th><th>Operation</th><th>Output per position</th><th>Parameters (fatconv / R7·R8)</th>"
            "<th>MACs per position (fatconv / R7·R8)</th><th>Init</th><th>Precision</th></tr></thead><tbody>" + body + "</tbody></table></div>")

POLICY_MERMAID = """flowchart TB
  T["Trunk: tower-end BN → ReLU<br/>C × 8 × 8 (fatconv C = 98, R7/R8 C = 128)"]
  T --> P1["1×1 conv C → 128, no bias<br/>(bf16)"]
  P1 --> P2["Batch norm, 128 channels<br/>(fp32 from here on)"]
  P2 --> P3["ReLU"]
  P3 --> P4["1×1 conv 128 → 76 + bias"]
  P4 --> P5["Flatten → 4,864 logits<br/>76 move types × 64 from-squares"]
  P5 --> P6["Play: mask illegal → temperature → softmax over legal → sample"]
  P5 --> P7["Training: softmax over all 4,864<br/>policy CE + illegal-mass penalty"]"""
VALUE_MERMAID = """flowchart TB
  T["Trunk: tower-end BN → ReLU<br/>C × 8 × 8 (fatconv C = 98, R7/R8 C = 128)"]
  T --> V1["1×1 conv C → 16, no bias"]
  V1 --> V2["Batch norm, 16 channels"]
  V2 --> V3["ReLU"]
  V3 --> V4["Flatten → 1,024"]
  V4 --> V5["Dense 1,024 → 128 + bias"]
  V5 --> V6["ReLU"]
  V6 --> V7["Dense 128 → 3 + bias<br/>(fp32 tail)"]
  V7 --> V8["Softmax → p_win, p_draw, p_loss"]
  V8 --> V9["v = p_win − p_loss"]
  V8 --> V10["Training: cross-entropy vs game result"]"""
CHANNEL_MAP = [
 ("0 – 55", "Queen-style moves: direction d (N, NE, E, SE, S, SW, W, NW) × distance 1–7; channel = d · 7 + (distance − 1)"),
 ("56 – 63", "Knight jumps, 8 directions"),
 ("64 – 72", "Under-promotions: knight, rook, bishop × {forward, capture left, capture right}"),
 ("73 – 75", "Queen promotions × {forward, capture left, capture right}"),
]
def channel_map():
    rows = "".join(f"<tr><th scope='row'>{a}</th><td class='wrap'>{b}</td></tr>" for a, b in CHANNEL_MAP)
    return f"<div class='tablebox'><table><thead><tr><th>Channels</th><th>Move type (from the mover's view; the board is flipped when Black moves)</th></tr></thead><tbody>{rows}</tbody></table></div>"

def net_section(k, name, arch, mid):
    r = R[k]
    return f"""<section class="net" id="net-{k}">
<header><h2>{name}</h2><span class="meta">{arch} · {mid} · step {int(r['training_step']):,}</span></header>
{strip_svg(r)}
<div class="tablebox"><table>
<thead><tr><th>ReLU site</th><th>channels</th><th>mean of all values</th><th>channel mean: min</th><th>median</th><th>max</th><th>channels with mean &gt; 1</th><th>channels never negative</th><th>channels never positive</th><th>channels with mean &lt; −1</th><th>least-active channel: % positive</th><th>most-active channel: % positive</th><th>min value</th><th>max value</th></tr></thead>
<tbody>{site_rows(r)}</tbody></table></div>
</section>"""

fc, r7, r8 = C["fatconv"], C["R7"], C["R8"]
NETS = [k for k, *_ in nets]
fc1_gt1_neg = [float(np.array(R[k]["relu_inputs"]["value.fc1"]["frac_negative"])[np.array(R[k]["relu_inputs"]["value.fc1"]["mean"]) > 1].min()) for k in NETS] + \
              [float(np.array(R[k]["relu_inputs"]["value.fc1"]["frac_negative"])[np.array(R[k]["relu_inputs"]["value.fc1"]["mean"]) > 1].max()) for k in NETS]
most_pos = max((max(s["frac_positive"]), k, site, int(np.argmax(s["frac_positive"]))) for k in NETS for site, s in R[k]["relu_inputs"].items())
pol_min = min(R[k]["policy"]["all_min"] for k in NETS); pol_max = max(R[k]["policy"]["all_max"] for k in NETS)
val_abs = max(max(abs(R[k]["value"]["all_min"]), abs(R[k]["value"]["all_max"])) for k in NETS)
legal_med = [R[k]["policy"]["per_position_legal_mean_pct"][2] for k in NETS]
ill_mean = [R[k]["policy"]["illegal_mean"] for k in NETS]
gap = [R[k]["policy"]["legal_mean"] - R[k]["policy"]["illegal_mean"] for k in NETS]
spread = [R[k]["policy"]["per_position_legal_spread_median"] for k in NETS]
beaten = [k for k in NETS if R[k]["policy"]["legal_top_minus_illegal_max_min"] < 0]
clean = [k for k in NETS if k not in beaten]
def names(ks): return " and ".join(ks) if len(ks) < 3 else ", ".join(ks[:-1]) + " and " + ks[-1]
if beaten:
    margin_text = (names(beaten) + (" has" if len(beaten) == 1 else " have") + " at least one position where an illegal move outscores the best legal one (by " +
                   " and ".join(f"{abs(R[k]['policy']['legal_top_minus_illegal_max_min']):.2f}" for k in beaten) + ")" +
                   (("; " + names(clean) + " never " + ("does" if len(clean) == 1 else "do") + " (smallest margin " + " and ".join(f"{R[k]['policy']['legal_top_minus_illegal_max_min']:.2f}" for k in clean) + ")") if clean else "") +
                   ". Legal-move masking makes this harmless at play time.")
else:
    margin_text = "No net ever puts an illegal move above its best legal one."
tr = R["fatconv"]["value"]["target_rates"]
ce_drop = ", ".join(f"{100 * (1 - R[k]['value']['value_ce'] / R[k]['value']['base_rate_ce']):.1f}% ({k})" for k in NETS)
body = f"""<main class="wrap">
<header style="display:grid;gap:10px">
<span class="kicker">DCM · final 33,000-step checkpoints · {NPOS:,} training-corpus positions</span>
<h1>What goes into each ReLU, and into each softmax</h1>
<p class="lede">Fatconv (one 15×15 block, 98 channels) against the two seeds of the three-block baseline, R7 and R8. Every value entering every ReLU was measured per channel, plus the policy and value logits just before their softmaxes. The positions are {NPOS:,} plies drawn from {NGAMES:,} games of the training corpus (Lichess standard rated games, May 2026), from shards these runs had not reached by step 33,000.</p>
</header>

<section aria-labelledby="ans">
<h2 id="ans">Short answers</h2>
<div class="answers">
<div class="answer"><span class="q">Any ReLU input with mean above 1?</span><span class="a">Not at any batch-norm-fed ReLU</span>
<p>Highest channel mean: fatconv {fc[4]:.2f}, R7 {r7[4]:.2f}, R8 {r8[4]:.2f} (all at the tower-end ReLU). Only the value head's FC1 has units above 1: fatconv {fc[0]}, R7 {r7[0]}, R8 {r8[0]}, each still negative {pct(min(fc1_gt1_neg))}–{pct(max(fc1_gt1_neg))} of the time.</p></div>
<div class="answer"><span class="q">Any channel with no negatives?</span><span class="a">None, in any net</span>
<p>Every channel at every ReLU goes negative on some positions, so no ReLU is acting as a plain pass-through. The most-active channel is positive {pct(max(max(s['frac_positive']) for s in R['fatconv']['relu_inputs'].values()))} of the time ({most_pos[1]}, value FC1 unit {most_pos[3]}).</p></div>
<div class="answer"><span class="q">Any channel that never goes positive?</span><span class="a">Only in value FC1</span>
<p>{fc[3]} of 128 units (fatconv), {r7[3]} (R7), {r8[3]} (R8) stay negative on all {NPOS:,} positions. They also receive almost no gradient: their optimizer velocity (stored [in, out]; per-unit column norm) is 0.007–0.21× the layer median, all inside the app audit's low-velocity class, and R8's unit 0 is ≈ 0. Dormant, close to dead.</p></div>
<div class="answer"><span class="q">Pre-softmax range</span><span class="a">Policy {sgn(pol_min, 0)} to {sgn(pol_max, 0)} · value ±{val_abs:.1f}</span>
<p>Legal moves sit near {sgn(min(legal_med), 1)} to {sgn(max(legal_med), 1)} (illegal near {sgn(min(ill_mean), 2)} to {sgn(max(ill_mean), 2)}); the choice among legal moves lives in a spread of about {min(spread):.2f}–{max(spread):.2f}. Value logits stay within ±{val_abs:.1f}.</p></div>
</div>
</section>


<section aria-labelledby="arch">
<h2 id="arch">The networks</h2>
<p>Everything except the tower is identical: same input, same heads, same training settings (build 2275, the same parameters file and corpus, 33,000 steps at batch 4,096). Fatconv spends the budget on one block of whole-board 15×15 convolutions at 98 channels; R7 and R8 are two seeds of three 7×7 blocks at 128 channels.</p>
{arch_table()}
</section>

<section aria-labelledby="heads">
<h2 id="heads">The two heads</h2>
<p>Both heads read the same trunk output: the tower-end batch norm followed by ReLU. The only difference between the nets is that trunk width, C: 98 for fatconv, 128 for R7 and R8.</p>
<div class="cols">
<div style="display:grid;gap:8px"><h3>Policy head</h3><pre class="mermaid">{POLICY_MERMAID}</pre></div>
<div style="display:grid;gap:8px"><h3>Value head</h3><pre class="mermaid">{VALUE_MERMAID}</pre></div>
</div>
<h3>Policy head, layer by layer</h3>
{layer_table(POLICY_LAYERS)}
<h3>Policy output channels</h3>
{channel_map()}
<h3>Value head, layer by layer</h3>
{layer_table(VALUE_LAYERS)}
<ul>
<li>Head totals (trainable): policy 22,604 (fatconv) / 26,444 (R7·R8); value 133,187 / 133,667. The value head's FC1 alone is 131,200 parameters, about 2.5% of each net.</li>
<li>Precision: the trunk and the early head layers run in bf16. The policy head runs in fp32 from its batch norm onward (the run's <code>fp32_from_pre_bn</code> setting); the value head's final dense layer, softmax and scalar always run in fp32.</li>
<li>Training losses, from the trainer: total = value weight · value CE + policy weight · policy CE + illegal-mass weight · illegal-mass penalty (all weights 1, no entropy bonus). The policy CE targets the played move with label smoothing ε = 0.1 and includes the complement branch for negative-advantage samples; the value CE targets the game result with smoothing 0.013.</li>
<li>The value head's final bias [0, ln 6, 0] centers the untrained head on W/D/L 0.125 / 0.75 / 0.125 (the draw prior); its He-initialized weights add position-to-position variation around that.</li>
</ul>
</section>
<section aria-labelledby="relu">
<h2 id="relu">ReLU inputs, per channel</h2>
<p>Each dot is one channel's mean input over all positions and squares (value FC1: one unit, over positions). The dashed line marks +1; dots are clamped to the −3 to +1.5 range shown.</p>
<div class="legend"><span class="l1">channel mean between −1 and +1</span><span class="l2">mean above +1</span><span class="l3">mean below −1</span></div>
</section>
{''.join(net_section(*n) for n in nets)}

<section aria-labelledby="pol">
<h2 id="pol">Policy logits, just before the softmax</h2>
<p>4,864 logits per position (76 move types × 64 squares). At play time the softmax runs over legal moves only; the training loss also sees the illegal ones.</p>
<div class="tablebox"><table><thead><tr><th></th>{''.join(f'<th>{n[1]}</th>' for n in nets)}</tr></thead><tbody>{policy_table()}</tbody></table></div>
<ul>
<li>All three put legal moves {min(gap):.1f}–{max(gap):.1f} above illegal ones on average, so the all-move softmax leaves on average {100*R['fatconv']['policy']['illegal_softmax_mass_mean']:.2f}% (fatconv), {100*R['R7']['policy']['illegal_softmax_mass_mean']:.2f}% (R7) and {100*R['R8']['policy']['illegal_softmax_mass_mean']:.2f}% (R8) of probability on illegal moves.</li>
<li>{margin_text}</li>
<li>Near +17, bf16 can only represent steps of 0.125, about a fifth of the legal spread. These runs train with the policy tail in fp32 (<code>fp32_from_pre_bn</code>), so training is not affected.</li>
<li>The app's default tail, <code>mixed_final_projection</code>, runs the final 1×1 projection in bf16 and widens only its output, so its logits do carry that rounding. Probing the 33,000-step checkpoints both ways (build 2320, <code>--probe-set wide</code>): fatconv pElo 1397.9 / NLL 2.3134 in fp32 against 1395.8 / 2.3135 mixed; R7 1492.8 / 2.1927 against 1493.4 / 2.1930; R8 1515.4 / 2.2042 against 1515.9 / 2.2041. At this logit size the rounding changes nothing measurable; it would matter if the legal logits grew toward 32 (step 0.25) or 64 (step 0.5).</li>
</ul>
</section>

<section aria-labelledby="val">
<h2 id="val">Value logits, just before the softmax</h2>
<p>Three logits per position, in the order win / draw / loss, from the side to move's point of view.</p>
<div class="tablebox"><table><thead><tr><th></th>{''.join(f'<th>{n[1]}</th>' for n in nets)}</tr></thead><tbody>{value_table()}</tbody></table></div>
<ul>
<li>All values are small (|x| ≤ {val_abs:.1f}); nothing is close to overflow in bf16 or fp16.</li>
<li>Against the actual game results the value CE is {R['fatconv']['value']['value_ce']:.4f} (fatconv), {R['R7']['value']['value_ce']:.4f} (R7) and {R['R8']['value']['value_ce']:.4f} (R8). Always predicting the sample's own result frequencies (W / D / L {tr[0]:.3f} / {tr[1]:.3f} / {tr[2]:.3f}, side to move) scores {R['fatconv']['value']['base_rate_ce']:.4f}; the heads are {ce_drop} below that baseline.</li>
<li>Fatconv's logits sit lower overall: a shared offset of {R['fatconv']['value']['shared_pct'][2]:.2f} against {R['R7']['value']['shared_pct'][2]:.2f} and {R['R8']['value']['shared_pct'][2]:.2f}. The softmax ignores a shared offset, and the three nets' probabilities agree closely.</li>
</ul>
</section>

<section aria-labelledby="method">
<h2 id="method">How this was measured</h2>
<ul>
<li>An fp32 numpy forward pass of each checkpoint in inference mode (running batch-norm statistics), reading the weights from the <code>.safetensors</code> files and following <code>ChessNetwork.swift</code>, <code>BoardEncoder.swift</code> (basic30) and <code>PolicyEncoding.swift</code>.</li>
<li>Checked against the app's own <code>--analyze-numerics</code> run (build 2320) on the same checkpoints, on the app audit's default positions (see below), where every figure matched; the corpus run uses the same code. The figures compared were start-position W/D/L, the per-position legal-logit quartiles and median, the legal spread median, the all-move mean, the value offset quartiles and the largest |logit|, each to every digit the app prints.</li>
<li>"ReLU input" is the tensor right before each ReLU: the batch-norm output at every tower and head site, and the FC1 output plus bias in the value head. The stem has no ReLU.</li>
<li>Positions: the training corpus <code>20260624-192615-w3aA5b</code> (lichess_db_standard_rated_2026-05, bot and human standard games). 1,024 games drawn at random from each of shards 20, 26, 32 and 38 (games shorter than two plies skipped), two distinct random plies per game: {NPOS:,} positions from {NGAMES:,} games, seed 20261004. These 33,000-step runs ended in shard 9, so none of these positions were trained on. Every replayed move is checked legal. The value target is the game result from the side to move's point of view.</li>
<li>The same analysis on the app audit's default positions (4,097 plies of 63 games played by DCM's own Lichess bot) gave the same picture: legal logits near +17.3, spread about 0.57, every channel negative somewhere, no batch-norm-fed channel mean above 1.</li>
<li>Checkpoints: <code>20261004-fatconv98-b2275-replay-step33000</code>, <code>20261002-bench_v5s3_noSE_noReZero-replay-step33000</code> (R7) and <code>…-seed2-replay-step33000</code> (R8).</li>
</ul>
</section>
</main>"""
open(f"{S}/report.html", "w").write(head + body)
print("ok", len(head + body))
