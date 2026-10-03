"""SVG line charts for the experiment reports: pElo and NLL by training step for every
compared run, and each run's difference from its baseline. No third-party packages, so
any `python3` can render them; the drawing follows the SE experiment's final-report
charts (light and dark styles, solid lines for seed 1, dashed for seed 2).

A chart is written to `<experiment>/charts/<name>.svg`, replacing only a regular file
this module wrote before (an existing folder or link at that path is refused).
"""
import html
import math
import os

SVG_STYLE = ('.bg{fill:#FFFFFF}.t{fill:#5E6770;font:11px ui-monospace,Menlo,monospace}'
             '.tt{fill:#1C2126;font:600 12px ui-monospace,Menlo,monospace}'
             '.g{stroke:#E4E7EA;stroke-width:1}.z{stroke:#9AA3AB;stroke-width:1}'
             '@media (prefers-color-scheme:dark){.bg{fill:#14171A}.t{fill:#9AA3AB}.tt{fill:#E6E9EC}'
             '.g{stroke:#262B30}.z{stroke:#5E6770}}')

# One color per compared configuration; a run's seed is drawn by its dash pattern.
PALETTE = ["#5B6B7A", "#D9822B", "#2B8A3E", "#8E44AD", "#C0392B", "#1F77B4", "#B8860B"]
SEED2_DASH = "6 4"


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
        return f"{v / 1000:g}k"
    if v == 0:
        return "0"
    return f"{v:g}".replace("-", "−")


def series(label, points, color, dash=""):
    """A line: `points` is a list of (step, value) pairs; a value of None is a gap
    (no measurement at that step) and is skipped."""
    return dict(label=label, color=color, dash=dash, pts=points)


def render(title, panels, xlabel="training step", width=760):
    """SVG text. `panels` is a list of dict(h, ylabel, series=[...], zero=False); every
    panel shares the x axis, which spans every plotted step."""
    xs = [x for p in panels for s in p["series"] for (x, v) in s["pts"] if v is not None]
    if not xs:
        raise ValueError(f"{title}: nothing to plot")
    xlo, xhi = min(xs), max(xs)
    if xlo == xhi:
        xhi = xlo + 1
    ml, mr, mt = 70, 18, 30
    legend = []
    for p in panels:
        for s in p["series"]:
            key = (s["label"], s["color"], s.get("dash", ""))
            if key not in legend:
                legend.append(key)
    rows, row, x = [], [], ml
    for lab, col, dash in legend:
        w = 34 + 7 * len(lab) + 18
        if x + w > width - mr and row:
            rows.append(row)
            row, x = [], ml
        row.append((x, lab, col, dash))
        x += w
    if row:
        rows.append(row)
    # Room for the legend rows plus the first panel's y-axis label, which sits above it.
    top = mt + 18 * len(rows) + 26
    gap = 34
    height = top + sum(p["h"] for p in panels) + gap * (len(panels) - 1) + 44
    pw = width - ml - mr

    def X(v):
        return ml + (v - xlo) / (xhi - xlo) * pw

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{html.escape(title)}"><style>{SVG_STYLE}</style>',
           f'<rect class="bg" width="{width}" height="{height}"/>',
           f'<text class="tt" x="{ml}" y="18">{html.escape(title)}</text>']
    for r_i, legend_row in enumerate(rows):
        yy = mt + 18 * r_i + 8
        for (x0, lab, col, dash) in legend_row:
            da = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<line x1="{x0}" y1="{yy}" x2="{x0 + 26}" y2="{yy}" stroke="{col}" stroke-width="2.4"{da}/>')
            out.append(f'<text class="t" x="{x0 + 32}" y="{yy + 4}">{html.escape(lab)}</text>')
    y0 = top
    xt = nice_ticks(xlo, xhi, 7)
    for p in panels:
        h = p["h"]
        ys = [v for s in p["series"] for (_, v) in s["pts"] if v is not None]
        if p.get("zero"):
            ys.append(0.0)
        lo_v, hi_v = min(ys), max(ys)
        pad = (hi_v - lo_v) * 0.06 or 1.0
        lo_v, hi_v = lo_v - pad, hi_v + pad

        def Y(v, lo_v=lo_v, hi_v=hi_v, y0=y0, h=h):
            return y0 + h - (v - lo_v) / (hi_v - lo_v) * h

        for t in nice_ticks(lo_v, hi_v, max(3, h // 45)):
            out.append(f'<line class="g" x1="{ml}" y1="{Y(t):.1f}" x2="{width - mr}" y2="{Y(t):.1f}"/>')
            out.append(f'<text class="t" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt_tick(t)}</text>')
        for t in xt:
            out.append(f'<line class="g" x1="{X(t):.1f}" y1="{y0}" x2="{X(t):.1f}" y2="{y0 + h}"/>')
        if p.get("zero"):
            out.append(f'<line class="z" x1="{ml}" y1="{Y(0):.1f}" x2="{width - mr}" y2="{Y(0):.1f}"/>')
        out.append(f'<text class="t" x="{ml}" y="{y0 - 8}">{html.escape(p["ylabel"])}</text>')
        for s in p["series"]:
            pts = [(x, v) for (x, v) in s["pts"] if v is not None]
            if not pts:
                continue
            d = " ".join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(v):.1f}' for i, (x, v) in enumerate(pts))
            da = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"{da}/>')
            for (x, v) in pts:
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="2.4" fill="{s["color"]}"/>')
        y0 += h + gap
    yb = y0 - gap
    for t in xt:
        out.append(f'<text class="t" x="{X(t):.1f}" y="{yb + 16}" text-anchor="middle">{fmt_tick(t)}</text>')
    out.append(f'<text class="t" x="{ml + pw / 2:.1f}" y="{yb + 34}" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_chart(experiment_dir, name, svg):
    """Write `charts/<name>` in the experiment folder through a temporary sibling and a
    rename, replacing only a regular file. Returns the path relative to the folder."""
    charts = os.path.join(experiment_dir, "charts")
    if os.path.lexists(charts) and not (os.path.isdir(charts) and not os.path.islink(charts)):
        raise SystemExit(f"{charts} exists and is not a folder; refusing to write charts there")
    os.makedirs(charts, exist_ok=True)
    path = os.path.join(charts, name)
    if os.path.lexists(path) and (os.path.islink(path) or not os.path.isfile(path)):
        raise SystemExit(f"{path} exists and is not a regular file; refusing to replace it")
    temporary = path + ".tmp"
    if os.path.lexists(temporary):
        raise SystemExit(f"{temporary} already exists; remove it by hand if it is a leftover")
    with open(temporary, "x") as handle:
        handle.write(svg)
    os.replace(temporary, path)
    return os.path.join("charts", name)


def metric_points(points, index):
    """[(step, value)] from a table arm's {step: (pElo, nll)}; None values stay as gaps."""
    return [(step, points[step][index]) for step in sorted(points)]


def difference_points(points, baseline, index):
    """[(step, arm − baseline)] at the steps both measured (a None on either side is a gap)."""
    out = []
    for step in sorted(set(points) & set(baseline)):
        a, b = points[step][index], baseline[step][index]
        out.append((step, None if a is None or b is None else a - b))
    return out
