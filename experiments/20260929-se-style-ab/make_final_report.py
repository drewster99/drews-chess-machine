"""Regenerate the final SE-style report: data files, SVG charts, REPORT-final.md and report-final.html.

    python3 experiments/20260929-se-style-ab/make_final_report.py

Step 1 runs final_data.py (data/*.csv from the dashboard probe CSVs, the archived logs and the
checkpoints). Everything after that reads only data/*.csv, the archived logs and checkpoint
metadata, so every number in the report is computed, never typed in.
"""
import csv, gzip, html, json, math, os, re, statistics as st, struct
from decimal import Decimal, ROUND_HALF_UP
import sys

sys.dont_write_bytecode = True  # keep __pycache__ out of the experiment folder
import final_data as FD  # noqa: E402

EXP = FD.EXP
DATA = FD.OUT
CHARTS = os.path.join(EXP, 'charts')
MODELS_DIR = FD.MODELS_DIR

ARM_COLOR = {'sb': '#B5532E', 'att': '#2F6F98', 'none': '#2F7D55', 'zb': '#8A5AA8'}
ARM_NAME = {'sb': 'scale+bias', 'att': 'attenuate-only', 'none': 'none', 'zb': 'zero-β s+b'}
SEED_COLOR = {1: '#2F6F98', 2: '#B5532E'}
SEED_DASH = {1: '', 2: '6 4'}
PARAMS = {'sb': '5,208,050', 'att': '5,195,378', 'none': '5,170,322', 'zb': '5,208,050'}
BLANK = ''


# ------------------------------------------------------------------------------------------------
# data access
# ------------------------------------------------------------------------------------------------
def read(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def num(x):
    return None if x in (None, '') else float(x)


PROBES = read('probes_all_runs.csv')
GAPS = read('paired_gaps.csv')
SEEDG = read('seed_gaps.csv')
NORMS = read('se_fc2_norms.csv')
LR = {int(r['step']): (float(r['lr']), float(r['momentum'])) for r in read('lr_schedule.csv')}

P = {}  # (run, step) -> row
for r in PROBES:
    P[(r['run'], int(r['step']))] = r


def pv(run, step, col='pElo'):
    r = P.get((run, step))
    return None if r is None else num(r[col])


def marks(run):
    return sorted(s for (k, s) in P if k == run and s % 1000 == 0)


def stop_save(run):
    xs = [s for (k, s) in P if k == run and s % 1000 != 0]
    return max(xs) if xs else None


def gap_series(comp, seed, col='d_pElo', lo=1000, hi=10 ** 9):
    return {int(r['step']): float(r[col]) for r in GAPS
            if r['comparison'] == comp and int(r['seed']) == seed and lo <= int(r['step']) <= hi}


def summary(xs):
    xs = list(xs)
    return dict(n=len(xs), mean=st.mean(xs), sd=st.stdev(xs) if len(xs) > 1 else float('nan'),
                min=min(xs), max=max(xs))


def rms(xs):
    xs = list(xs)
    return math.sqrt(st.mean([x * x for x in xs]))


def norm_rows(run):
    return [r for r in NORMS if r['run'] == run]


def norm_at(run, step, block, col):
    for r in NORMS:
        if r['run'] == run and int(r['step']) == step and int(r['block']) == block:
            return num(r[col])
    return None


def ckpt_meta(pattern_stem, step):
    path = os.path.join(MODELS_DIR, f'20260929-test_SE_{pattern_stem}-replay-step{step}.safetensors')
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return json.loads(fh.read(n))['__metadata__']


def log_span(stem):
    first = last = None
    with gzip.open(os.path.join(EXP, 'logs', stem + '.txt.gz'), 'rt', errors='replace') as fh:
        for line in fh:
            if first is None:
                first = line[:8]
            if '[REPLAY] done:' in line:
                last = line[:8]
    return first, last


# formatting ---------------------------------------------------------------------------------------
def dec(x, d):
    """Round half away from zero on the value's shortest decimal form (after trimming float noise),
    so a difference like 850.03 − 940.38 = −90.35 displays as −90.4, not binary-float −90.3."""
    q = Decimal(1).scaleb(-d)
    return Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP)


def fp(x, d=2):
    return BLANK if x is None else f'{dec(x, d):f}'


def fs(x, d=2):
    if x is None:
        return BLANK
    v = dec(x, d)
    return (('+' if v >= 0 else '') + f'{v:f}').replace('-', '−')


def fm(x, d=2):
    return BLANK if x is None else f'{dec(x, d):f}'.replace('-', '−')


# ------------------------------------------------------------------------------------------------
# SVG charts
# ------------------------------------------------------------------------------------------------
SVG_STYLE = ('.bg{fill:#FFFFFF}.t{fill:#5E6770;font:11px ui-monospace,Menlo,monospace}'
             '.tt{fill:#1C2126;font:600 12px ui-monospace,Menlo,monospace}'
             '.g{stroke:#E4E7EA;stroke-width:1}.z{stroke:#9AA3AB;stroke-width:1}'
             '.lr{stroke:#8C959D}.lrf{fill:#8C959D}'
             '@media (prefers-color-scheme:dark){.bg{fill:#14171A}.t{fill:#9AA3AB}.tt{fill:#E6E9EC}'
             '.g{stroke:#262B30}.z{stroke:#5E6770}.lr{stroke:#7A838B}.lrf{fill:#7A838B}}')


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def fmt_tick(v):
    if abs(v) >= 1000:
        return f'{v / 1000:g}k'
    if v == 0:
        return '0'
    return f'{v:g}'.replace('-', '−')


def chart(name, title, xlo, xhi, panels, xlabel='training step', width=760, lr_strip=True):
    """panels: list of dict(h, ylabel, series=[dict(label, color, dash, pts, dots)], zero, ylog, ypad)."""
    ml, mr, mt = 70, 18, 30
    legend = []
    for p in panels:
        for s in p['series']:
            key = (s['label'], s['color'], s.get('dash', ''))
            if s.get('legend', True) and key not in legend:
                legend.append(key)
    # legend layout
    lx, ly, rows = ml, mt, []
    row = []
    x = ml
    for lab, col, dash in legend:
        w = 34 + 7 * len(lab) + 18
        if x + w > width - mr and row:
            rows.append(row)
            row, x = [], ml
        row.append((x, lab, col, dash))
        x += w
    if row:
        rows.append(row)
    top = mt + 18 * len(rows) + 10 + (16 if len(panels[0]['ylabel']) > 14 else 0)
    all_panels = list(panels)
    if lr_strip:
        pts = [(s, LR[s][0]) for s in sorted(LR) if xlo <= s <= xhi]
        all_panels.append(dict(h=70, ylabel='LR (log)', ylog=True,
                               series=[dict(label='learning rate', color=None, cls='lr', dash='', pts=pts,
                                            dots=False, legend=False)]))
    gap = 34
    H = top + sum(p['h'] for p in all_panels) + gap * (len(all_panels) - 1) + 44
    pw = width - ml - mr
    X = lambda v: ml + (v - xlo) / (xhi - xlo) * pw
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {H}" role="img" '
         f'aria-label="{html.escape(title)}"><style>{SVG_STYLE}</style>',
         f'<rect class="bg" width="{width}" height="{H}"/>',
         f'<text class="tt" x="{ml}" y="18">{html.escape(title)}</text>']
    for r_i, row in enumerate(rows):
        yy = mt + 18 * r_i + 8
        for (x0, lab, col, dash) in row:
            da = f' stroke-dasharray="{dash}"' if dash else ''
            o.append(f'<line x1="{x0}" y1="{yy}" x2="{x0 + 26}" y2="{yy}" stroke="{col}" stroke-width="2.4"{da}/>')
            o.append(f'<text class="t" x="{x0 + 32}" y="{yy + 4}">{html.escape(lab)}</text>')
    y0 = top
    xt = nice_ticks(xlo, xhi, 7)
    for p_i, p in enumerate(all_panels):
        h = p['h']
        ys = [v for s in p['series'] for (_, v) in s['pts'] if v is not None]
        if p.get('zero'):
            ys.append(0.0)
        if p.get('ylog'):
            lo_v, hi_v = math.log10(min(ys)), math.log10(max(ys))
            lo_v, hi_v = math.floor(lo_v), math.ceil(hi_v)
            Y = lambda v, lo_v=lo_v, hi_v=hi_v: y0 + h - (math.log10(v) - lo_v) / (hi_v - lo_v) * h
            yt = [10 ** e for e in range(int(lo_v), int(hi_v) + 1)]
        else:
            lo_v, hi_v = min(ys), max(ys)
            pad = (hi_v - lo_v) * 0.06 or 1.0
            lo_v, hi_v = lo_v - pad, hi_v + pad
            yt = nice_ticks(lo_v, hi_v, max(3, h // 45))
            Y = lambda v, lo_v=lo_v, hi_v=hi_v: y0 + h - (v - lo_v) / (hi_v - lo_v) * h
        for t in yt:
            yy = Y(t)
            o.append(f'<line class="g" x1="{ml}" y1="{yy:.1f}" x2="{width - mr}" y2="{yy:.1f}"/>')
            o.append(f'<text class="t" x="{ml - 6}" y="{yy + 4:.1f}" text-anchor="end">{fmt_tick(t)}</text>')
        for t in xt:
            o.append(f'<line class="g" x1="{X(t):.1f}" y1="{y0}" x2="{X(t):.1f}" y2="{y0 + h}"/>')
        if p.get('zero') and not p.get('ylog'):
            o.append(f'<line class="z" x1="{ml}" y1="{Y(0):.1f}" x2="{width - mr}" y2="{Y(0):.1f}"/>')
        if len(p['ylabel']) > 14:
            o.append(f'<text class="t" x="{ml}" y="{y0 - 8}">{html.escape(p["ylabel"])}</text>')
        else:
            o.append(f'<text class="t" transform="translate(14,{y0 + h / 2:.1f}) rotate(-90)" '
                     f'text-anchor="middle">{html.escape(p["ylabel"])}</text>')
        for s in p['series']:
            pts = [(x, v) for (x, v) in s['pts'] if v is not None]
            if not pts:
                continue
            d = ' '.join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(v):.1f}' for i, (x, v) in enumerate(pts))
            da = f' stroke-dasharray="{s["dash"]}"' if s.get('dash') else ''
            if s.get('cls'):
                o.append(f'<path class="{s["cls"]}" d="{d}" fill="none" stroke-width="1.6"{da}/>')
            else:
                o.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"{da}/>')
                if s.get('dots', True):
                    for (x, v) in pts:
                        o.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{s["color"]}"/>')
        y0 += h + gap
    yb = y0 - gap
    for t in xt:
        o.append(f'<text class="t" x="{X(t):.1f}" y="{yb + 16}" text-anchor="middle">{fmt_tick(t)}</text>')
    o.append(f'<text class="t" x="{ml + pw / 2:.1f}" y="{yb + 34}" text-anchor="middle">{html.escape(xlabel)}</text>')
    o.append('</svg>')
    svg = '\n'.join(o)
    os.makedirs(CHARTS, exist_ok=True)
    open(os.path.join(CHARTS, name), 'w').write(svg + '\n')
    return name


def run_series(col, lo, hi, codes=('sb', 'att', 'none', 'zb')):
    out = []
    for code in codes:
        for seed in (1, 2):
            run = FD.KEY[(code, seed)]
            pts = [(s, pv(run, s, col)) for s in marks(run) if lo <= s <= hi]
            out.append(dict(label=f'{ARM_NAME[code]} seed {seed}', color=ARM_COLOR[code],
                            dash=SEED_DASH[seed], pts=pts))
    return out


def build_charts():
    names = {}
    names['pelo'] = chart('final-pelo-1k-7k.svg', 'pElo by step, marks 1k–7k (solid = seed 1, dashed = seed 2)',
                          1000, 7000, [dict(h=300, ylabel='pElo', series=run_series('pElo', 1000, 7000))])
    names['nll'] = chart('final-nll-1k-7k.svg', 'nll by step, marks 1k–7k (lower is better)',
                         1000, 7000, [dict(h=300, ylabel='nll', series=run_series('nll', 1000, 7000))])
    names['pelo_all'] = chart('final-pelo-all.svg', 'pElo by step, every run, full length',
                              1000, 33000, [dict(h=300, ylabel='pElo',
                                                 series=run_series('pElo', 1000, 33000))])
    panels = []
    for comp, lab in (('none-sb', 'none − scale+bias'), ('none-att', 'none − attenuate-only'),
                      ('att-sb', 'attenuate-only − scale+bias')):
        ser = []
        for seed in (1, 2):
            g = gap_series(comp, seed, hi=7000)
            ser.append(dict(label=f'seed {seed}', color=SEED_COLOR[seed], dash=SEED_DASH[seed],
                            pts=sorted(g.items())))
        panels.append(dict(h=150, ylabel=f'{lab} (pElo)', zero=True, series=ser))
    names['armgaps'] = chart('final-arm-gaps.svg', 'Paired pElo gaps within each seed (positive = first arm ahead)',
                             1000, 7000, panels)
    panels = []
    for col, lab in (('d_pElo', 'zero-β − parent (pElo)'), ('d_nll', 'zero-β − parent (nll)')):
        ser = []
        for seed in (1, 2):
            g = gap_series('zb-sb', seed, col=col)
            ser.append(dict(label=f'seed {seed}', color=SEED_COLOR[seed], dash=SEED_DASH[seed],
                            pts=sorted(g.items())))
        panels.append(dict(h=170, ylabel=lab, zero=True, series=ser))
    names['zb'] = chart('final-zerobeta-vs-parent.svg',
                        'Zero-β minus its Glorot-β parent, same seed, same step',
                        1000, 5000, panels)
    ser = []
    for code in ('sb', 'att', 'none', 'zb'):
        pts = sorted((int(r['step']), float(r['d_pElo'])) for r in SEEDG if r['arm'] == code)
        ser.append(dict(label=ARM_NAME[code], color=ARM_COLOR[code], dash='', pts=pts))
    names['seedgaps'] = chart('final-seed-gaps.svg', 'Seed 2 − seed 1, same arm, same step (pElo)',
                              1000, 7000, [dict(h=240, ylabel='seed 2 − seed 1 (pElo)', zero=True, series=ser)])
    panels = []
    for b in range(3):
        ser = []
        for run, code, seed in (('se_sb', 'sb', 1), ('se_sb2', 'sb', 2), ('se_zb1', 'zb', 1), ('se_zb2', 'zb', 2)):
            rows = sorted((int(r['step']), float(r['beta_W_fro'])) for r in NORMS
                          if r['run'] == run and int(r['block']) == b and int(r['step']) <= 7300)
            ser.append(dict(label=f'{"Glorot-β" if code == "sb" else "zero-β"} ‖β‖ seed {seed}',
                            color=ARM_COLOR[code], dash=SEED_DASH[seed], pts=rows))
        for run, seed in (('se_sb', 1), ('se_sb2', 2)):
            rows = sorted((int(r['step']), float(r['beta_orth_to_init_fro'])) for r in NORMS
                          if r['run'] == run and int(r['block']) == b and int(r['step']) <= 7300)
            ser.append(dict(label=f'Glorot-β learned part (⊥ init) seed {seed}', color='#C9A227',
                            dash=SEED_DASH[seed], pts=rows))
        panels.append(dict(h=150, ylabel=f'block {b} β norm', series=ser))
    names['beta'] = chart('final-beta-norms.svg',
                          'SE fc2 β-half Frobenius norm, and the Glorot runs’ component orthogonal to their init',
                          0, 7300, panels)
    panels = []
    for b in range(3):
        ser = []
        for col, lab, color in (('beta_W_fro', '‖β‖', ARM_COLOR['sb']),
                                ('beta_orth_to_init_fro', 'β ⊥ init', '#C9A227'),
                                ('gamma_orth_to_init_fro', 'γ ⊥ init', ARM_COLOR['att'])):
            rows = sorted((int(r['step']), float(r[col])) for r in NORMS
                          if r['run'] == 'se_sb' and int(r['block']) == b)
            ser.append(dict(label=lab, color=color, dash='', pts=rows, dots=False))
        panels.append(dict(h=130, ylabel=f'block {b}', series=ser))
    names['beta_long'] = chart('final-beta-seed1-long.svg',
                               'Seed-1 scale+bias, full run: β norm, and the parts of β and γ that moved off their init',
                               0, 33014, panels)
    return names


# ------------------------------------------------------------------------------------------------
# document model: blocks rendered to both Markdown and HTML
# ------------------------------------------------------------------------------------------------
DOC = []


def h(level, text, anchor=None):
    DOC.append(('h', level, text, anchor))


def p(text):
    DOC.append(('p', text))


def ul(items):
    DOC.append(('ul', items))


def table(header, rows, align=None):
    DOC.append(('table', header, rows, align))


def fig(name, alt, caption):
    DOC.append(('fig', name, alt, caption))


def codeblock(text):
    DOC.append(('code', text))


def md_render():
    out = []
    for b in DOC:
        if b[0] == 'h':
            if b[3]:
                out.append(f'<a id="{b[3]}"></a>\n')
            out.append('#' * b[1] + ' ' + b[2] + '\n')
        elif b[0] == 'p':
            out.append(b[1] + '\n')
        elif b[0] == 'ul':
            out.append('\n'.join(('- ' + i) if not i.startswith('  ') else ('  - ' + i.strip()) for i in b[1]) + '\n')
        elif b[0] == 'table':
            header, rows, align = b[1], b[2], b[3]
            esc = lambda c: str(c).replace('|', '\\|')
            out.append('| ' + ' | '.join(esc(c) for c in header) + ' |')
            out.append('|' + '|'.join(('---:' if (align and align[i] == 'r') else '---') for i in range(len(header))) + '|')
            for r in rows:
                out.append('| ' + ' | '.join(esc(c) for c in r) + ' |')
            out.append('')
        elif b[0] == 'fig':
            out.append(f'![{b[2]}](charts/{b[1]})\n\n*{b[3]}*\n')
        elif b[0] == 'code':
            out.append('```sh\n' + b[1] + '\n```\n')
    return '\n'.join(out)


def inline_html(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    return t


def slug(t):
    return re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')


def html_render(title):
    body = []
    toc = []
    for b in DOC:
        if b[0] == 'h':
            sid = b[3] or slug(b[2])
            if b[1] == 1:
                body.append(f'<h1>{inline_html(b[2])}</h1>')
            else:
                body.append(f'<h{b[1]} id="{sid}">{inline_html(b[2])}</h{b[1]}>')
                if b[1] == 2:
                    toc.append(f'<a href="#{sid}">{inline_html(b[2])}</a>')
        elif b[0] == 'p':
            body.append(f'<p>{inline_html(b[1])}</p>')
        elif b[0] == 'ul':
            items, buf, sub = b[1], [], []
            for i in items:
                if i.startswith('  '):
                    sub.append(f'<li>{inline_html(i.strip())}</li>')
                else:
                    if sub:
                        buf[-1] = buf[-1][:-5] + '<ul>' + ''.join(sub) + '</ul></li>'
                        sub = []
                    buf.append(f'<li>{inline_html(i)}</li>')
            if sub:
                buf[-1] = buf[-1][:-5] + '<ul>' + ''.join(sub) + '</ul></li>'
            body.append('<ul>' + ''.join(buf) + '</ul>')
        elif b[0] == 'table':
            header, rows, align = b[1], b[2], b[3]
            cls = lambda i: ' class="n"' if (align and align[i] == 'r') else ''
            t = ['<div class="tw"><table><thead><tr>'
                 + ''.join(f'<th{cls(i)}>{inline_html(x)}</th>' for i, x in enumerate(header)) + '</tr></thead><tbody>']
            for r in rows:
                t.append('<tr>' + ''.join(f'<td{cls(i)}>{inline_html(str(c))}</td>' for i, c in enumerate(r)) + '</tr>')
            t.append('</tbody></table></div>')
            body.append(''.join(t))
        elif b[0] == 'fig':
            svg = open(os.path.join(CHARTS, b[1])).read()
            svg = re.sub(r'<style>.*?</style>', '', svg, flags=re.S)
            body.append(f'<figure>{svg}<figcaption>{inline_html(b[3])} '
                        f'<a href="charts/{b[1]}">SVG</a></figcaption></figure>')
        elif b[0] == 'code':
            body.append(f'<pre><code>{html.escape(b[1])}</code></pre>')
    css = """
:root{--bg:#F7F8F6;--panel:#FFFFFF;--ink:#1C2126;--muted:#5E6770;--rule:#D9DDD8;--grid:#E4E7EA;--accent:#2F6F98;--code:#EEF1EE;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171A;--panel:#1B1F23;--ink:#E6E9EC;--muted:#9AA3AB;--rule:#2C3238;--grid:#262B30;--accent:#6FA8D2;--code:#22272C;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14171A;--panel:#1B1F23;--ink:#E6E9EC;--muted:#9AA3AB;--rule:#2C3238;--grid:#262B30;--accent:#6FA8D2;--code:#22272C;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.6 "Iowan Old Style","Apple Garamond",Georgia,serif;margin:0;padding-inline:16px;padding-block:24px 64px}
main{max-width:980px;margin:0 auto}
h1{font-size:30px;line-height:1.2;text-wrap:balance;margin:8px 0 4px}
h2{font-size:21px;margin:40px 0 8px;padding-top:12px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:16px;margin:24px 0 6px}
p,li{max-width:74ch}
a{color:var(--accent)}
code,pre{font:12.5px ui-monospace,Menlo,monospace;background:var(--code);border-radius:4px}
code{padding:1px 4px} pre{padding:12px;overflow-x:auto}
pre code{padding:0;background:none}
.tw{overflow-x:auto;margin:10px 0 18px;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;font:12.5px ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}
td.n,th.n{text-align:right}
tr:last-child td{border-bottom:0}
figure{margin:16px 0 22px}
figure svg{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:6px}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
.bg{fill:var(--panel)} .t{fill:var(--muted);font:11px ui-monospace,Menlo,monospace}
.tt{fill:var(--ink);font:600 12px ui-monospace,Menlo,monospace}
.g{stroke:var(--grid);stroke-width:1} .z{stroke:var(--muted);stroke-width:1} .lr{stroke:var(--muted)}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin:12px 0 4px}
.meta{color:var(--muted);font-size:13px}
"""
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{html.escape(title)}</title>\n<style>{css}</style>\n</head>\n<body>\n<main>\n'
            + body[0] + '\n<p class="meta">Generated by <code>make_final_report.py</code> from the files in '
            '<a href="data/README.md">data/</a>. Markdown version: <a href="REPORT-final.md">REPORT-final.md</a>.</p>\n'
            + '<nav class="toc">' + ''.join(toc) + '</nav>\n' + '\n'.join(body[1:]) + '\n</main>\n</body>\n</html>\n')


# ------------------------------------------------------------------------------------------------
# the report
# ------------------------------------------------------------------------------------------------
def build_report(ch):
    R = FD.RUNS
    s1, s2 = 1, 2
    K = FD.KEY

    # run metadata from checkpoints and logs
    meta = {}
    stems = {'se_sb': 'scale+bias', 'se_att': 'attenuate-only', 'se_none': 'none',
             'se_sb2': 'scale+bias-seed2', 'se_att2': 'attenuate-only-seed2', 'se_none2': 'none-seed2',
             'se_zb1': 'zerobeta-seed1', 'se_zb2': 'zerobeta-seed2'}
    for key, arm, code, seed, binit, mid, log in R:
        md = ckpt_meta(stems[key], 1000)
        start, stop = log_span(log)
        assert md['parent_model_id'] == mid, (key, md['parent_model_id'], mid)
        meta[key] = dict(trained=md['model_id'], build=md['built_by_build'], git=md['built_by_git'],
                         start=start, stop=stop, last=max(marks(key)), stop_step=stop_save(key))

    lr_at = lambda s: LR[s][0]
    trough1 = min((s for s in LR if 5000 < s < 15000), key=lambda s: LR[s][0])
    peak2 = max((s for s in LR if 15000 < s < 25000), key=lambda s: LR[s][0])
    trough2 = min((s for s in LR if 25000 < s <= 33000), key=lambda s: LR[s][0])

    def gs(comp, seed, lo, hi, col='d_pElo'):
        return summary(gap_series(comp, seed, col=col, lo=lo, hi=hi).values())

    # seed noise
    sg = {code: {int(r['step']): float(r['d_pElo']) for r in SEEDG if r['arm'] == code}
          for code in ('sb', 'att', 'none', 'zb')}
    three = [v for c in ('sb', 'att', 'none') for s, v in sg[c].items() if s <= 7000]
    rms3 = rms(three)
    sd_run = rms3 / math.sqrt(2)
    need10 = math.ceil((rms3 / 10) ** 2)
    need15 = math.ceil((rms3 / 15) ** 2)
    zbd = {s: gap_series('zb-sb', 2)[s] - gap_series('zb-sb', 1)[s] for s in gap_series('zb-sb', 1)
           if s in gap_series('zb-sb', 2)}
    armd = {}
    for comp in ('none-sb', 'none-att', 'att-sb'):
        a, b = gap_series(comp, 1, hi=7000), gap_series(comp, 2, hi=7000)
        armd[comp] = [b[s] - a[s] for s in a if s in b]

    zb1 = gs('zb-sb', 1, 1000, 5000); zb2 = gs('zb-sb', 2, 1000, 5000)
    na_means = [gs('none-att', 1, 1000, 7000)['mean'], gs('none-att', 2, 1000, 7000)['mean'],
                gs('none-att', 1, 1000, 31000)['mean']]
    zb1n = gs('zb-sb', 1, 1000, 5000, 'd_nll'); zb2n = gs('zb-sb', 2, 1000, 5000, 'd_nll')

    # ---- title & summary
    h(1, 'SE style experiment — final report')
    p('Corpus-replay comparison of squeeze-and-excitation (SE) variants on an otherwise identical 3-block 7×7 @128 '
      'network: **scale+bias** SE, **attenuate-only** SE, **no SE**, plus a fourth arm, **zero-β scale+bias**, derived '
      'from the scale+bias fresh nets with only the β path zeroed. Two independent seeds per arm. Experiment record, '
      'launch details and the full reproduce recipe: [README.md](README.md). Earlier single-seed analysis at step '
      '30,000: [REPORT-30k.md](REPORT-30k.md).')

    h(2, 'Summary')
    n1 = gs('none-sb', 1, 1000, 31000)
    ul([
        f'**Scale+bias SE is worse than both attenuate-only and no SE.** Both seeds agree on pElo and nll. '
        f'Over marks 1k–7k the mean paired gap none − scale+bias is {fs(gs("none-sb", 1, 1000, 7000)["mean"], 1)} (seed 1) '
        f'and {fs(gs("none-sb", 2, 1000, 7000)["mean"], 1)} (seed 2) pElo; attenuate-only − scale+bias is '
        f'{fs(gs("att-sb", 1, 1000, 7000)["mean"], 1)} and {fs(gs("att-sb", 2, 1000, 7000)["mean"], 1)}. Seed 1’s '
        f'full run (1k–31k) gives none − scale+bias {fs(n1["mean"], 1)}.',
        f'**No SE is probably at least as good as attenuate-only, at worst similar.** none − attenuate-only is positive '
        f'on average in both seeds ({fs(gs("none-att", 1, 1000, 7000)["mean"], 1)} and '
        f'{fs(gs("none-att", 2, 1000, 7000)["mean"], 1)} over 1k–7k; {fs(gs("none-att", 1, 1000, 31000)["mean"], 1)} over '
        f'seed 1’s full run), but the per-mark spread is large and the two seeds disagree on nll.',
        f'**Zero-β did not rescue scale+bias in any consistent way.** Paired with its own Glorot-β parent (identical '
        f'weights except β) over 1k–5k: {fs(zb1["mean"], 1)} pElo (sd {fm(zb1["sd"], 1)}) in seed 1 and '
        f'{fs(zb2["mean"], 1)} (sd {fm(zb2["sd"], 1)}) in seed 2. At 5k it was the best seed-2 arm and the worst '
        f'seed-1 arm. Against no SE it averaged {fs(gs("zb-none", 1, 1000, 5000)["mean"], 1)} and '
        f'{fs(gs("zb-none", 2, 1000, 5000)["mean"], 1)}.',
        f'**β does learn, in both inits.** The part of the Glorot β that moved away from its random init reaches about the '
        f'same size by 5k as the whole learned β in the zero-β runs (block 0: '
        f'{fm(norm_at("se_sb", 5000, 0, "beta_orth_to_init_fro"), 2)} vs '
        f'{fm(norm_at("se_zb1", 5000, 0, "beta_W_fro"), 2)} in seed 1). The random component (norm ≈ 5) is only '
        f'shrunk slowly by weight decay. The earlier reading that β "barely trains" was wrong; see '
        f'[β learning dynamics](#beta-learning-dynamics).',
        f'**Run-to-run noise is about {fm(sd_run, 0)} pElo (sd) per run and is mostly training noise, not init '
        f'noise.** Sharing the init (zero-β vs parent) did not make the paired comparison any quieter than comparing '
        f'different inits: the seed-to-seed difference of the paired gap has RMS {fm(rms(zbd.values()), 1)} pElo (1k–5k), vs '
        f'{fm(min(rms(v) for v in armd.values()), 1)}–{fm(max(rms(v) for v in armd.values()), 1)} for the '
        f'independently initialized arm gaps (1k–7k). The minibatch sampler is unseeded, so paired runs do not share batches.',
        f'**Recommendation:** prefer no SE on this network (attenuate-only as a close second); drop scale+bias. A '
        f'decisive answer on attenuate-only vs none, or on β init, needs about {need15}–{need10} seeds per arm, or the '
        f'seeded-sampler work in plan #8 so paired runs actually share their training noise.',
    ])

    # ---- question
    h(2, 'Question and hypotheses')
    ul([
        'Does the SE style matter on this net under corpus replay, holding everything else fixed? SE variants '
        '(per channel c, pooled features → FC1 → FC2):',
        '  scale+bias: `y_c = σ(γ_c)·x_c + β_c`, with γ and β both from FC2 (2C outputs).',
        '  attenuate-only: `y_c = σ(γ_c)·x_c` (FC2 has C outputs).',
        '  none: no SE block.',
        '  zero-β scale+bias: scale+bias whose β half of FC2 starts at exactly 0.',
        'H1 (seeds 1–2): the seed-1 ordering none ≥ attenuate-only > scale+bias replicates on fresh random inits.',
        'H2 (zero-β): scale+bias trails because its β path starts from random weights, adding an input-dependent '
        'random offset about the size of the signal. If so, zero-β should close the gap to attenuate-only / none.',
    ])

    # ---- design
    h(2, 'Design')
    ul([
        'Network (all arms): v5, input basic30 (30 planes), stem 7×7→128, 3 blocks of 7×7+7×7 @128, ReLU pre-activation, '
        'ReZero α 0.447 (tanh cap 0.447), clean add, per-block output LayerNorm, policy intermediate_conv (4864), '
        'value WDL (16→FC128), bf16 compute. Only the SE style differs (and β init for zero-β).',
        'Training: corpus replay of `20260624-192615-w3aA5b` (lichess 2026-05 standard, first 20,935,171 games) in corpus '
        'order, 500k-position buffer, 250k prefill, batch 4096, weight decay 3e-4, LR cycle peak 1e-1 / trough 1e-3 '
        'over 20k steps with decay, warmup 1000 steps. Identical `parameters.json` and flags for every run.',
        f'LR schedule (identical in all eight logs, checked point by point): warmup to {fm(lr_at(1000), 3)} at 1k, '
        f'down to {fm(lr_at(5000), 4)} at 5k and {fm(lr_at(7000), 5)} at 7k, trough {lr_at(trough1):.3g} at '
        f'{trough1:,}, peak {lr_at(peak2):.3g} at {peak2:,}, trough {lr_at(trough2):.3g} at {trough2:,}. '
        'Everything in the two-seed comparison (1k–7k) sits on the descent from the first peak.',
        'Seeds: seed 1 and seed 2 are two independent random inits of each preset (new fresh nets, new ModelIDs). The '
        'build has no seed option, so "seed" means an independent mint, not a numbered RNG seed.',
        'Zero-β derivation: `--derive-model --set-se-beta-init zero` on each seed’s scale+bias fresh net. The '
        'derived file is byte-identical to its parent except rows 128–255 (the β half) of each block’s SE fc2 '
        'weight [256, 32], which are set to 0. The β biases were already 0 at init. Verified byte-by-byte on all three '
        'blocks of both nets.',
        'Paired design: each zero-β run is compared with its parent run (same seed, same step). Sharing the init '
        'removes init noise, but not training noise: minibatch sampling is unseeded, so the two runs see different '
        'batches from step 1.',
        'Probe: every 1k-step checkpoint (and each stop save) is scored with `--probe-model <ckpt> --probe-set wide`, '
        'the 4,435-puzzle set. **pElo** is the puzzle-rating estimate (higher is better); **nll** is the mean negative '
        'log-likelihood of the solution moves (lower is better).',
    ])

    h(3, 'Runs')
    rows = []
    for key, arm, code, seed, binit, mid, log in R:
        m = meta[key]
        rows.append([f'`{key}`', ARM_NAME[code], seed, PARAMS[code], f'`{mid}`', f'`{m["trained"]}`',
                     f'{m["build"]} (`{m["git"]}`)', f'`{log}`', f'{m["start"]}–{m["stop"]}',
                     f'{m["last"]:,}', f'{m["stop_step"]:,}' if m['stop_step'] else ''])
    table(['run', 'arm', 'seed', 'params', 'fresh ModelID', 'trained ModelID', 'build (git)', 'log',
           'log start–end (CDT)', 'last 1k mark', 'stop save'], rows,
          ['l', 'l', 'r', 'r', 'l', 'l', 'l', 'l', 'l', 'r', 'r'])
    ul([
        'Seed 1 ran 2026-09-29 15:07 → 2026-09-30 ~10:37 (three arms concurrently on one GPU). Seed 2 ran 2026-09-30 '
        '10:41 → 15:05 (three concurrently), stopped early by decision at the 7k marks. Zero-β ran 2026-09-30 15:05 → '
        '17:15 (two concurrently), stopped at the 5k marks. Every stop was SIGINT right after an enumerated checkpoint, '
        'so each run also wrote a stop save.',
        'Builds: seed 1 = 2255 (`5826e1c` + the uncommitted trainer-config change later committed as `cbc1894`). '
        'Seed 2 = 2259 (`5f46da0`), which adds optimizer state to checkpoints (`d15f706`), so seed-2 files are about '
        'twice the size; training math for a fresh start is unchanged. Zero-β = 2261 (`31253d5` with uncommitted '
        'changes = the code of `8926221`, #7), which adds `se_beta_init`, format v4 and `--derive-model`; for Glorot β '
        'the graph is unchanged. Build numbers come from each checkpoint’s `built_by_build` metadata.',
        'Concurrency differs by phase (3, 3, then 2 processes on one GPU), so wall-clock throughput is not comparable. '
        'All comparisons here are by training step.',
    ])

    # ---- per-mark tables
    h(2, 'Results: every run, every mark')
    p('Blank cells: the run never reached that step. Nothing is carried forward. LR is the scheduled learning rate at '
      'that step (from the logs).')
    allsteps = sorted({s for (k, s) in P if s % 1000 == 0})
    order = [K[('sb', 1)], K[('att', 1)], K[('none', 1)], K[('zb', 1)], K[('sb', 2)], K[('att', 2)], K[('none', 2)],
             K[('zb', 2)]]
    hdr = ['step', 'LR'] + [f'{ARM_NAME[FD.RUN[k][2]]} s{FD.RUN[k][3]}' for k in order]
    for col, d, lab in (('pElo', 2, 'pElo'), ('nll', 4, 'nll')):
        h(3, f'{lab} at each 1k mark')
        table(hdr, [[f'{s:,}', f'{lr_at(s):.3g}'] + [fp(pv(k, s, col), d) for k in order] for s in allsteps],
              ['r'] * len(hdr))
    h(3, 'Stop saves')
    table(['run', 'step', 'pElo', 'nll'],
          [[f'`{k}`', f'{meta[k]["stop_step"]:,}', fp(pv(k, meta[k]['stop_step'])), fp(pv(k, meta[k]['stop_step'], 'nll'), 4)]
           for k in order], ['l', 'r', 'r', 'r'])
    fig(ch['pelo'], 'pElo by step 1k–7k', 'pElo, marks 1k–7k, all eight runs. Solid = seed 1, dashed = seed 2. '
        'Bottom strip: learning rate (log scale).')
    fig(ch['nll'], 'nll by step 1k–7k', 'nll (lower is better), marks 1k–7k.')
    fig(ch['pelo_all'], 'pElo by step, full length', 'Every run at full length. Only seed 1 continued past 7k. The '
        'LR strip shows the whole first cycle. Seed-1 detail: [REPORT-30k.md](REPORT-30k.md).')

    # ---- paired
    h(2, 'Paired comparisons within a seed')
    p('Each gap is row arm minus column arm at the same step in the same seed, then summarized over the marks in the '
      'window. Marks within one run are strongly correlated, so n counts marks, not independent samples; the '
      'independent sample size is the number of seeds (2).')
    rows = []
    for comp, lab in (('none-sb', 'none − scale+bias'), ('none-att', 'none − attenuate-only'),
                      ('att-sb', 'attenuate-only − scale+bias')):
        a, b = gs(comp, 1, 1000, 7000), gs(comp, 2, 1000, 7000)
        pool = summary(list(gap_series(comp, 1, hi=7000).values()) + list(gap_series(comp, 2, hi=7000).values()))
        full = gs(comp, 1, 1000, 31000)
        an, bn = gs(comp, 1, 1000, 7000, 'd_nll'), gs(comp, 2, 1000, 7000, 'd_nll')
        rows.append([lab, f'{fs(a["mean"], 1)} ({fm(a["sd"], 1)})', f'{fs(b["mean"], 1)} ({fm(b["sd"], 1)})',
                     f'{fs(pool["mean"], 1)} ({fm(pool["sd"], 1)})', f'{fs(pool["min"], 1)} … {fs(pool["max"], 1)}',
                     f'{fs(full["mean"], 1)} ({fm(full["sd"], 1)}), n={full["n"]}',
                     f'{fs(an["mean"], 4)} / {fs(bn["mean"], 4)}'])
    table(['comparison', 'seed 1, 1k–7k: mean (sd)', 'seed 2, 1k–7k', 'pooled', 'range, both seeds',
           'seed 1, 1k–31k', 'nll mean, s1 / s2 (1k–7k)'], rows, ['l', 'r', 'r', 'r', 'r', 'r', 'r'])
    h(3, 'Per-mark gaps, 1k–7k (pElo)')
    comps = [('none-sb', 'none − s+b'), ('none-att', 'none − att'), ('att-sb', 'att − s+b')]
    hdr = ['step'] + [f'{lab} s{sd}' for comp, lab in comps for sd in (1, 2)]
    table(hdr, [[f'{s:,}'] + [fs(gap_series(comp, sd).get(s), 2) for comp, _ in comps for sd in (1, 2)]
                for s in range(1000, 8000, 1000)], ['r'] * len(hdr))
    h(3, 'Per-mark gaps, 1k–7k (nll; negative = first arm better)')
    table(hdr, [[f'{s:,}'] + [fs(gap_series(comp, sd, col='d_nll').get(s), 4) for comp, _ in comps for sd in (1, 2)]
                for s in range(1000, 8000, 1000)], ['r'] * len(hdr))
    fig(ch['armgaps'], 'Paired arm gaps', 'Paired pElo gaps per seed. Seed 2’s gaps against scale+bias are larger, '
        'partly because seed 2’s scale+bias init sits consistently low (see seed noise).')
    ul([
        f'none − scale+bias is positive at {sum(1 for sd in (1, 2) for v in gap_series("none-sb", sd, hi=7000).values() if v > 0)} '
        f'of 14 seed×mark points; attenuate-only − scale+bias at '
        f'{sum(1 for sd in (1, 2) for v in gap_series("att-sb", sd, hi=7000).values() if v > 0)} of 14; none − '
        f'attenuate-only at {sum(1 for sd in (1, 2) for v in gap_series("none-att", sd, hi=7000).values() if v > 0)} of 14.',
        f'The attenuate-only − scale+bias gap moves almost in lockstep across seeds (correlation of the per-mark gaps '
        f'between seeds {st.correlation([gap_series("att-sb", 1)[s] for s in range(1000, 8000, 1000)], [gap_series("att-sb", 2)[s] for s in range(1000, 8000, 1000)]):.2f}); '
        f'the gaps involving none do not ({st.correlation([gap_series("none-sb", 1)[s] for s in range(1000, 8000, 1000)], [gap_series("none-sb", 2)[s] for s in range(1000, 8000, 1000)]):.2f} for none − s+b, '
        f'{st.correlation([gap_series("none-att", 1)[s] for s in range(1000, 8000, 1000)], [gap_series("none-att", 2)[s] for s in range(1000, 8000, 1000)]):.2f} for none − att). '
        'The two SE arms appear to respond to the LR descent in the same way, the no-SE arm differently.',
        f'Seed 1’s early window understates its full-run gaps: none − scale+bias {fs(gs("none-sb", 1, 1000, 7000)["mean"], 1)} '
        f'over 1k–7k vs {fs(n1["mean"], 1)} over 1k–31k; the gap grew around the second LR peak (see '
        '[REPORT-30k.md](REPORT-30k.md)). Seed 2 never reached that phase.',
    ])

    # ---- zero-beta
    h(2, 'Zero-β scale+bias vs its parent', 'zero-beta')
    zrows = []
    for s in range(1000, 6000, 1000):
        row = [f'{s:,}']
        for sd in (1, 2):
            zk, pk = K[('zb', sd)], K[('sb', sd)]
            row += [fp(pv(zk, s)), fp(pv(pk, s)), fs(pv(zk, s) - pv(pk, s)), fp(pv(K[('att', sd)], s)),
                    fp(pv(K[('none', sd)], s))]
        zrows.append(row)
    table(['step', 'zero-β s1', 'parent s1', 'Δ s1', 'att s1', 'none s1', 'zero-β s2', 'parent s2', 'Δ s2', 'att s2',
           'none s2'], zrows, ['r'] * 11)
    zrows = []
    for s in range(1000, 6000, 1000):
        row = [f'{s:,}']
        for sd in (1, 2):
            zk, pk = K[('zb', sd)], K[('sb', sd)]
            row += [fp(pv(zk, s, 'nll'), 4), fp(pv(pk, s, 'nll'), 4), fs(pv(zk, s, 'nll') - pv(pk, s, 'nll'), 4),
                    fp(pv(K[('att', sd)], s, 'nll'), 4), fp(pv(K[('none', sd)], s, 'nll'), 4)]
        zrows.append(row)
    p('nll (Δ negative = zero-β better):')
    table(['step', 'zero-β s1', 'parent s1', 'Δ s1', 'att s1', 'none s1', 'zero-β s2', 'parent s2', 'Δ s2', 'att s2',
           'none s2'], zrows, ['r'] * 11)
    rows = []
    for comp, lab in (('zb-sb', 'zero-β − parent (scale+bias)'), ('zb-att', 'zero-β − attenuate-only'),
                      ('zb-none', 'zero-β − none')):
        a, b = gs(comp, 1, 1000, 5000), gs(comp, 2, 1000, 5000)
        an, bn = gs(comp, 1, 1000, 5000, 'd_nll'), gs(comp, 2, 1000, 5000, 'd_nll')
        rows.append([lab, f'{fs(a["mean"], 1)} ({fm(a["sd"], 1)})', f'{fs(a["min"], 1)} … {fs(a["max"], 1)}',
                     f'{fs(b["mean"], 1)} ({fm(b["sd"], 1)})', f'{fs(b["min"], 1)} … {fs(b["max"], 1)}',
                     f'{fs(an["mean"], 4)} / {fs(bn["mean"], 4)}'])
    table(['comparison (1k–5k)', 'seed 1 mean (sd)', 'seed 1 range', 'seed 2 mean (sd)', 'seed 2 range',
           'nll mean s1 / s2'], rows, ['l', 'r', 'r', 'r', 'r', 'r'])
    fig(ch['zb'], 'Zero-beta minus parent', 'Zero-β minus its Glorot-β parent, per seed. pElo: positive = zero-β '
        'ahead; nll: negative = zero-β better. The two seeds move in the same direction for the first four marks and split at 5k.')
    ul([
        f'The paired gap swings by up to {fm(max(abs(v) for sd in (1, 2) for v in gap_series("zb-sb", sd).values()), 1)} pElo '
        f'between adjacent marks, and its mean over 1k–5k is {fs(zb1["mean"], 1)} (seed 1) and {fs(zb2["mean"], 1)} '
        '(seed 2), both well inside one sd. There is no consistent effect of β init on pElo.',
        f'nll agrees: {fs(zb1n["mean"], 4)} (seed 1) and {fs(zb2n["mean"], 4)} (seed 2), opposite signs.',
        f'At 5k the seeds disagree outright: seed 2’s zero-β is the best seed-2 arm '
        f'({fp(pv("se_zb2", 5000))} vs none {fp(pv("se_none2", 5000))}); seed 1’s is the worst seed-1 arm '
        f'({fp(pv("se_zb1", 5000))} vs scale+bias {fp(pv("se_sb", 5000))}).',
        f'Against no SE, zero-β averages {fs(gs("zb-none", 1, 1000, 5000)["mean"], 1)} and '
        f'{fs(gs("zb-none", 2, 1000, 5000)["mean"], 1)} over 1k–5k, and against attenuate-only '
        f'{fs(gs("zb-att", 1, 1000, 5000)["mean"], 1)} and {fs(gs("zb-att", 2, 1000, 5000)["mean"], 1)}: on average it '
        'did not reach either.',
        'H2 is not supported: removing the random β start did not produce the improvement it predicted, at the '
        'resolution this test has. It is not ruled out either; an effect of 10–20 pElo would be invisible here.',
    ])

    # ---- beta dynamics
    h(2, 'β learning dynamics', 'beta-learning-dynamics')
    p('Computed from every enumerated checkpoint of the four scale+bias-style runs (plus the fresh nets at step 0), '
      'identified by `__metadata__` ModelID lineage. For each block, FC2’s weight is [256, 32]: rows 0–127 produce '
      'γ, rows 128–255 produce β. ‖·‖ is the Frobenius norm of that half. "⊥ init" is the norm of the part of the '
      'weights orthogonal to their step-0 values: pure weight decay only rescales a vector, so it leaves this at 0 '
      'and anything above 0 is gradient-driven change of direction. For zero-β the init is 0, so ⊥ init equals ‖β‖.')
    rows = []
    for run in ('se_sb', 'se_sb2', 'se_zb1', 'se_zb2'):
        steps = sorted({int(r['step']) for r in norm_rows(run) if int(r['step']) <= 7300})
        for s in steps:
            rows.append([f'`{run}`', f'{s:,}']
                        + [fm(norm_at(run, s, b, 'beta_W_fro'), 3) for b in range(3)]
                        + [fm(norm_at(run, s, b, 'beta_orth_to_init_fro'), 3) for b in range(3)]
                        + [fm(norm_at(run, s, b, 'gamma_orth_to_init_fro'), 3) for b in range(3)]
                        + [fm(norm_at(run, s, b, 'beta_bias_absmean'), 4) for b in range(3)])
    table(['run', 'step', '‖β‖ b0', '‖β‖ b1', '‖β‖ b2', 'β⊥init b0', 'β⊥init b1', 'β⊥init b2', 'γ⊥init b0',
           'γ⊥init b1', 'γ⊥init b2', 'mean|β bias| b0', 'b1', 'b2'], rows, ['l'] + ['r'] * 13)
    fig(ch['beta'], 'Beta norms', 'β-half norm per block, 0–7.3k. Glorot-β runs (rust) start near 5.3 and shrink '
        'slowly; zero-β runs (purple) grow from 0. Gold: the Glorot runs’ learned (orthogonal-to-init) part, which '
        'tracks the zero-β β closely.')
    rows = []
    for s in sorted({int(r['step']) for r in norm_rows('se_sb')}):
        if s % 5000 == 0 or s in (1000, 33014):
            rows.append([f'{s:,}'] + [fm(norm_at('se_sb', s, b, 'beta_W_fro'), 3) for b in range(3)]
                        + [fm(norm_at('se_sb', s, b, 'beta_cos_to_init'), 4) for b in range(3)]
                        + [fm(norm_at('se_sb', s, b, 'beta_orth_to_init_fro'), 3) for b in range(3)]
                        + [fm(norm_at('se_sb', s, b, 'gamma_orth_to_init_fro'), 3) for b in range(3)])
    h(3, 'Seed-1 scale+bias over the full run')
    table(['step', '‖β‖ b0', 'b1', 'b2', 'cos(β, β₀) b0', 'b1', 'b2', 'β⊥init b0', 'b1', 'b2', 'γ⊥init b0', 'b1', 'b2'],
          rows, ['r'] * 13)
    fig(ch['beta_long'], 'Seed 1 full-run beta', 'Seed-1 scale+bias, all 35 checkpoints: ‖β‖ (rust), the learned '
        'part of β (gold) and of γ (blue). The random part of β is still most of its norm at 33k.')
    bmax = max(abs(num(r['beta_bias_mean'])) for r in NORMS)
    ul([
        f'**Zero-β’s β grows steadily and is still growing at 5k** (block 0: '
        + ', '.join(fm(norm_at('se_zb1', s, 0, 'beta_W_fro'), 2) for s in range(1000, 6000, 1000))
        + ' at 1k–5k in seed 1), with growth slowing as the LR falls.',
        f'**The Glorot β learns about as much.** Its orthogonal-to-init part at 5k is '
        + '/'.join(fm(norm_at('se_sb', 5000, b, 'beta_orth_to_init_fro'), 2) for b in range(3))
        + ' (blocks 0/1/2, seed 1) vs zero-β’s '
        + '/'.join(fm(norm_at('se_zb1', 5000, b, 'beta_W_fro'), 2) for b in range(3))
        + '; seed 2: ' + '/'.join(fm(norm_at('se_sb2', 5000, b, 'beta_orth_to_init_fro'), 2) for b in range(3))
        + ' vs ' + '/'.join(fm(norm_at('se_zb2', 5000, b, 'beta_W_fro'), 2) for b in range(3)) + '.',
        f'**γ moves by a similar amount** (γ⊥init at 5k, seed 1: '
        + '/'.join(fm(norm_at('se_sb', 5000, b, 'gamma_orth_to_init_fro'), 2) for b in range(3)) + '). So β is not '
        'a dead path; it trains like the rest of the SE block.',
        f'**The random component persists.** Seed 1’s Glorot β keeps cos(β, β₀) = '
        + '/'.join(fm(norm_at('se_sb', 33014, b, 'beta_cos_to_init'), 3) for b in range(3))
        + ' at 33,014 steps; its norm fell from ' + '/'.join(fm(norm_at('se_sb', 0, b, 'beta_W_fro'), 2) for b in range(3))
        + ' to ' + '/'.join(fm(norm_at('se_sb', 33014, b, 'beta_W_fro'), 2) for b in range(3)) + '.',
        f'**The β bias mean stays at 0** (largest |mean| across all {len(NORMS)} block-checkpoints: {bmax:.1e}): the '
        'block-output LayerNorm cancels a common shift, so only the per-channel pattern of β bias can matter. '
        'Per-channel |β bias| grows to about 0.01.',
        'Correction to earlier notes: the seed-1 README said β "barely trains", based on its bias and its norm. The '
        'direction analysis shows that was wrong: β learned; the random init simply stays on top of what it learned.',
    ])

    # ---- noise
    h(2, 'Seed noise')
    rows = []
    for code in ('sb', 'att', 'none', 'zb'):
        v = sg[code]
        sm = summary(v.values())
        rows.append([ARM_NAME[code], sm['n'], fs(sm['mean'], 1), fm(sm['sd'], 1), fs(sm['min'], 1), fs(sm['max'], 1),
                     fm(st.mean(abs(x) for x in v.values()), 1), fm(rms(v.values()), 1)])
    table(['arm', 'marks', 'mean (s2 − s1)', 'sd', 'min', 'max', 'mean |gap|', 'RMS gap'], rows,
          ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r'])
    h(3, 'Per mark (seed 2 − seed 1, pElo)')
    table(['step'] + [ARM_NAME[c] for c in ('sb', 'att', 'none', 'zb')],
          [[f'{s:,}'] + [fs(sg[c].get(s), 2) for c in ('sb', 'att', 'none', 'zb')] for s in range(1000, 8000, 1000)],
          ['r'] * 5)
    fig(ch['seedgaps'], 'Seed gaps', 'Seed 2 − seed 1 for the same arm and step. Seed 2’s scale+bias sits below '
        'its seed-1 twin at every mark.')
    ul([
        f'Over the three original arms at 1k–7k ({len(three)} pairs): mean |gap| {fm(st.mean(abs(x) for x in three), 1)}, '
        f'median {fm(st.median(abs(x) for x in three), 1)}, RMS {fm(rms3, 1)}, max {fm(max(abs(x) for x in three), 1)} '
        f'pElo. One run’s seed noise is therefore about **{fm(sd_run, 1)} pElo (sd)** (RMS ÷ √2), consistent with '
        'the 6.4–43.7 band from the earlier nt8y seed study.',
        f'A difference between two single runs carries about ±{fm(rms3, 0)} pElo (1 sd) of noise, about the size of the '
        f'effects measured. With n seeds per arm the difference of means has sd ≈ {fm(rms3, 0)}/√n: '
        f'resolving it to ±10 takes about {need10} seeds per arm, to ±15 (enough to call a 30-point effect at 2 sd) '
        f'about {need15}.',
        f'Seed 2’s scale+bias is {fs(summary(sg["sb"].values())["mean"], 1)} below its seed-1 twin on average, with '
        f'the smallest sd of any arm ({fm(summary(sg["sb"].values())["sd"], 1)}): that one init draw was weaker, which '
        'inflates seed 2’s gaps against scale+bias. The true scale+bias deficit is probably between the two seeds’ numbers.',
        f'Sharing the init did not reduce noise. The seed-to-seed difference of the paired zero-β − parent gap has RMS '
        f'{fm(rms(zbd.values()), 1)} over 1k–5k, about the same as for comparisons between independently initialized arms '
        f'({", ".join(fm(rms(v), 1) for v in armd.values())} for none − s+b, none − att, att − s+b, over 1k–7k). Most of the noise '
        'comes from training (unseeded sampling order, nondeterministic GPU reductions), not from the init.',
    ])

    # ---- conclusions
    h(2, 'Conclusions')
    ul([
        '**Scale+bias SE is worse than attenuate-only and no SE on this network.** Supported by both seeds on pElo and '
        'nll and by seed 1’s full run. This is the one conclusion both seeds back without qualification.',
        '**No SE ≥ attenuate-only, at worst similar.** Positive mean gaps in both seeds and over seed 1’s full run, '
        'but with large per-mark spread and seed disagreement on nll.',
        '**β init does not explain the scale+bias deficit** at the resolution of this test. Zero-β is not consistently '
        'better than its parent and on average stays below both attenuate-only and no SE.',
        '**β trains in both inits;** the Glorot init adds a slowly decaying random component on top of what it learns.',
        '**SE adds parameters without measurable benefit here.** The cheapest variant (none, 5,170,322 params) is the '
        'best or tied-best.',
    ])
    h(3, 'Limits')
    ul([
        'Two seeds. With two, the defensible claim is the direction both agree on; if two arms were truly equal, both '
        'seeds favouring the same named arm would still happen 25% of the time by chance.',
        'Marks within a run are strongly correlated; they are not independent evidence.',
        'Seed 2 and zero-β cover only the high-LR descent (1k–7k and 1k–5k). In seed 1 the SE arms swung ±60–130 pElo '
        'per mark around the LR peak ([REPORT-30k.md](REPORT-30k.md)), and the SE gap was largest there and smaller '
        'at the troughs.',
        'Engine builds differ between phases (2255, 2259, 2261); the changes are checkpoint contents and the β-init '
        'option, not training math, but it is a difference.',
        'One corpus, one architecture (3 blocks, 128 channels, LayerNorm block output), one training recipe.',
    ])
    h(3, 'What we did not learn, and why')
    ul([
        f'Whether attenuate-only and no SE really differ: the mean gap ({fs(min(na_means), 1)} to {fs(max(na_means), 1)} pElo '
        f'across the windows above) is below the ±{fm(rms3, 0)} noise of a two-run comparison.',
        'Whether β init matters at the 10–20 pElo level: the paired design did not cut the noise, because training '
        'noise dominates and the sampler is unseeded.',
        'Whether zero-β catches up at low LR: it was stopped at 5k, before the first trough.',
        'Whether SE helps on deeper or wider nets, or without the per-block LayerNorm (which already cancels any '
        'common β shift).',
    ])
    h(3, 'Recommendations')
    ul([
        'Use no SE for this network family; attenuate-only if an SE is wanted. Drop scale+bias.',
        f'For future A/B tests: plan on {need15}+ seeds per arm for effects around 30 pElo, {need10}+ for 10-pElo '
        f'effects, and compare at LR troughs where runs are steadiest.',
        'Make paired designs work by landing plan #8 (issue #8): a seeded sampler and exact resume, so a derived model '
        f'and its parent see the same minibatches and differ only by the change under test. Today the paired gap still '
        f'carries about {fm(rms(zbd.values()), 0)} pElo (RMS) of seed-to-seed noise per mark.',
    ])

    # ---- reproduce
    h(2, 'Reproduce')
    p('Build, corpus and fresh-net setup: [README.md › Reproduce](README.md#reproduce). With `BIN` and `M` set as '
      'there, and the fresh nets copied from `models/`:')
    codeblock('# seeds 1 and 2: one process per arm, <arm> = scale+bias | attenuate-only | none\n'
         '# (seed 2 files carry -seed2 in the name)\n'
         '"$BIN" --replay-corpus 20260624-192615-w3aA5b \\\n'
         '  --start-model "$M/20260929-test_SE_<arm>[-seed2]-fresh.safetensors" \\\n'
         '  --out-model "$M/20260929-test_SE_<arm>[-seed2]-replay-latest.safetensors" \\\n'
         '  --parameters experiments/20260929-se-style-ab/parameters.json \\\n'
         '  --epochs 12 --enumerate-checkpoints\n\n'
         '# zero-beta nets (or use the stored copies in models/)\n'
         'for s in "" "-seed2"; do\n'
         '  "$BIN" --derive-model \\\n'
         '    --from "experiments/20260929-se-style-ab/models/20260929-test_SE_scale+bias${s}-fresh.safetensors" \\\n'
         '    --set-se-beta-init zero \\\n'
         '    --out "$M/20260929-test_SE_zerobeta${s:-"-seed1"}-fresh.safetensors"\n'
         'done\n\n'
         '# zero-beta runs, <seed> = seed1 | seed2\n'
         '"$BIN" --replay-corpus 20260624-192615-w3aA5b \\\n'
         '  --start-model "$M/20260929-test_SE_zerobeta-<seed>-fresh.safetensors" \\\n'
         '  --out-model "$M/20260929-test_SE_zerobeta-<seed>-replay-latest.safetensors" \\\n'
         '  --parameters experiments/20260929-se-style-ab/parameters.json \\\n'
         '  --epochs 12 --enumerate-checkpoints\n\n'
         '# stop: SIGINT right after the last wanted enumerated checkpoint (writes a stop save)\n\n'
         '# probe every checkpoint into documentation/dashboards/data/<run>.csv\n'
         'cd documentation/dashboards\n'
         'python3 -c "import replay; [replay.probe_backfill(r) for r in (\'se_sb\',\'se_att\',\'se_none\',\'se_sb2\',\'se_att2\',\'se_none2\',\'se_zb1\',\'se_zb2\')]"\n\n'
         '# regenerate every data file, chart and this report\n'
         'cd ../..\n'
         'python3 experiments/20260929-se-style-ab/make_final_report.py')
    ul([
        'Not bit-exact: minibatch sampling is unseeded and GPU reductions are not guaranteed order-stable (README › '
        'Expected exactness). Expect curves that track, within the seed noise above.',
        '`make_final_report.py` reads the enumerated checkpoints from `~/Library/Application Support/DrewsChessMachine/'
        'Models/` for the β analysis; only the fresh nets and final checkpoints are in `models/`, so a clone without '
        'the original Models folder can regenerate everything except `se_fc2_norms.csv`.',
    ])

    # ---- data files
    h(2, 'Data files')
    p('All in [data/](data/), described column by column in [data/README.md](data/README.md). Produced by '
      '[final_data.py](final_data.py).')
    table(['file', 'rows', 'what it holds'], [
        ['[probes_all_runs.csv](data/probes_all_runs.csv)', len(PROBES),
         'Every probed checkpoint of all eight runs, long format: pElo, nll, losses, legal mass, gNorm, games fed.'],
        ['[paired_gaps.csv](data/paired_gaps.csv)', len(GAPS),
         'Within-seed arm differences at each 1k mark (none−sb, none−att, att−sb, zb−sb, zb−att, zb−none), pElo and nll.'],
        ['[seed_gaps.csv](data/seed_gaps.csv)', len(SEEDG), 'Seed 2 − seed 1 for the same arm and mark, pElo and nll.'],
        ['[se_fc2_norms.csv](data/se_fc2_norms.csv)', len(NORMS),
         'Per checkpoint and block: γ/β norms of SE fc2, bias statistics, direction change vs init.'],
        ['[lr_schedule.csv](data/lr_schedule.csv)', len(LR), 'Learning rate and momentum at every logged step.'],
        ['[tensor_stats.csv](data/tensor_stats.csv)', len(read('tensor_stats.csv')),
         'Mean, min, max, std, norms, zero fraction and non-finite count of every tensor in every checkpoint '
         '(145 checkpoints, including optimizer velocity). Findings: [TENSOR-STATS.md](TENSOR-STATS.md).'],
    ], ['l', 'r', 'l'])
    p('Source series per run (the dashboard tracker’s CSVs): '
      + ', '.join(f'[{k}.csv](../../documentation/dashboards/data/{k}.csv)' for k, *_ in R) + '.')

    h(2, 'Charts')
    table(['file', 'shows'], [
        [f'[{ch["pelo"]}](charts/{ch["pelo"]})', 'pElo by step, 1k–7k, all runs, with LR strip'],
        [f'[{ch["nll"]}](charts/{ch["nll"]})', 'nll by step, 1k–7k'],
        [f'[{ch["pelo_all"]}](charts/{ch["pelo_all"]})', 'pElo by step, every run at full length'],
        [f'[{ch["armgaps"]}](charts/{ch["armgaps"]})', 'Paired arm gaps per seed'],
        [f'[{ch["zb"]}](charts/{ch["zb"]})', 'Zero-β minus parent, pElo and nll'],
        [f'[{ch["seedgaps"]}](charts/{ch["seedgaps"]})', 'Seed 2 − seed 1 per arm'],
        [f'[{ch["beta"]}](charts/{ch["beta"]})', 'β norms and learned β, 0–7.3k'],
        [f'[{ch["beta_long"]}](charts/{ch["beta_long"]})', 'Seed-1 scale+bias β/γ over the full run'],
        ['[chart-pelo-30k.svg](chart-pelo-30k.svg), [chart-nll-30k.svg](chart-nll-30k.svg)',
         'Seed-1 curves to 30k with trough/peak bands (from the 30k report)'],
    ], ['l', 'l'])

    h(2, 'Artifacts')
    ul([
        'Checkpoints (Git LFS) in [models/](models/), with ModelIDs, sizes and SHA-256s in [MODELS.md](MODELS.md): '
        'fresh nets for all eight runs; seed-1 step 30000, last marks and stop saves; seed-2 step 7000 and stop saves; '
        'zero-β step 5000 and stop saves.',
        'Gzipped run logs (Git LFS) in [logs/](logs/), one per run: '
        + ', '.join(f'`{FD.RUN[k][6]}.txt.gz`' for k in order) + '.',
        'Inputs: [parameters.json](parameters.json), [presets/](presets/).',
        'Generators: [final_data.py](final_data.py), [make_final_report.py](make_final_report.py); the 30k report’s '
        '[make_report.py](make_report.py); per-tensor statistics: [tensor_stats.py](tensor_stats.py), '
        '[tensor_report.py](tensor_report.py).',
    ])


def main():
    FD.main()
    global PROBES, GAPS, SEEDG, NORMS, LR, P
    PROBES = read('probes_all_runs.csv'); GAPS = read('paired_gaps.csv'); SEEDG = read('seed_gaps.csv')
    NORMS = read('se_fc2_norms.csv')
    LR = {int(r['step']): (float(r['lr']), float(r['momentum'])) for r in read('lr_schedule.csv')}
    P = {(r['run'], int(r['step'])): r for r in PROBES}
    ch = build_charts()
    build_report(ch)
    open(os.path.join(EXP, 'REPORT-final.md'), 'w').write(md_render())
    open(os.path.join(EXP, 'report-final.html'), 'w').write(html_render('SE Style Final Report'))
    print('wrote REPORT-final.md, report-final.html, charts:', ', '.join(sorted(ch.values())))


if __name__ == '__main__':
    main()
