// Experiment summaries: navigation, related links, index search and charts.
//
// Every summary page carries its metadata in <meta name="dcm-…"> tags; build.py reads and
// validates those tags from every page and writes experiments.js (window.DCM_EXPERIMENTS), so
// the index, the previous/next links, the related list and the superseded banner all come from
// that one validated parse and cannot drift from the pages. This script only renders: it never
// holds experiment data of its own, and it reads no meta tag but the page's own dcm-id.
//
// Charts are declared in the page as
//   <figure class="chart"><script type="application/json">{spec}</script><figcaption>…</figcaption></figure>
// with spec = { x: {label}, y: {label, min?, max?}, series: [{name, points: [[x, y|null], …], dash?}],
//               vlines?: [{x, label}] }. A null y breaks the line (a gap, never interpolated); a point
// with no drawn neighbour on either side is shown as a dot. A chart that cannot be drawn
// faithfully (no values, a value or vline outside the plot) shows its error instead.
(function () {
  "use strict";
  const MISSING_DATA = "experiments.js missing — run build.py";
  const el = (tag, attrs, text) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    return node;
  };
  const link = e => el("a", { href: e.file, title: e.takeaway }, `${e.id} · ${e.title}`);

  function experiments() {
    if (window.DCM_EXPERIMENTS === undefined) {
      document.querySelector("main").prepend(el("div", { class: "banner" }, MISSING_DATA));
      throw new Error(MISSING_DATA);
    }
    return window.DCM_EXPERIMENTS.slice().sort((a, b) => a.id.localeCompare(b.id));
  }

  function renderPage(all) {
    const idTag = document.querySelector('meta[name="dcm-id"]');
    const id = idTag ? idTag.getAttribute("content") : null;
    const byId = new Map(all.map(e => [e.id, e]));
    const index = all.findIndex(e => e.id === id);
    const entry = index >= 0 ? all[index] : null;
    const missing = `${id || "this page"} is not in experiments.js — run build.py`;
    document.querySelectorAll(".dcm-nav").forEach(nav => {
      nav.textContent = "";
      if (!entry) {
        nav.append(el("span", {}, missing), el("a", { href: "index.html" }, "All experiments"));
        return;
      }
      const prev = all[index - 1], next = all[index + 1];
      nav.append(prev ? el("a", { href: prev.file, title: prev.title }, `← ${prev.id}`) : el("span", {}, "← first"));
      nav.append(el("a", { href: "index.html" }, "All experiments"));
      nav.append(next ? el("a", { href: next.file, title: next.title }, `${next.id} →`) : el("span", {}, "latest →"));
    });
    const box = document.getElementById("related");
    if (box && !entry) {
      box.append(el("p", { class: "note" }, missing));
    } else if (box && entry.related.length) {
      const ul = el("ul");
      entry.related.forEach(r => {
        const e = byId.get(r);
        const li = el("li");
        li.append(e ? link(e) : el("span", {}, `${r} (missing)`));
        if (e) li.append(el("span", { class: "note" }, ` — ${e.takeaway}`));
        ul.append(li);
      });
      box.append(ul);
    } else if (box) {
      box.closest("section").hidden = true;
    }
    if (entry && entry.supersededBy) {
      const e = byId.get(entry.supersededBy);
      const banner = el("div", { class: "banner" }, "Superseded: a later experiment revises this conclusion — ");
      banner.append(e ? link(e) : document.createTextNode(entry.supersededBy));
      document.querySelector("header.exp").after(banner);
    }
  }

  function renderIndex(all) {
    const body = document.getElementById("index-body");
    const input = document.getElementById("search");
    const tagBox = document.getElementById("tags");
    const count = document.getElementById("count");
    const active = new Set();
    const tags = [...new Set(all.flatMap(e => e.tags))].sort();
    tags.forEach(t => {
      const b = el("button", { type: "button", "aria-pressed": "false" }, t);
      b.addEventListener("click", () => {
        if (active.has(t)) active.delete(t); else active.add(t);
        b.setAttribute("aria-pressed", active.has(t) ? "true" : "false");
        draw();
      });
      tagBox.append(b);
    });
    function draw() {
      const q = input.value.trim().toLowerCase();
      const rows = all.slice().reverse().filter(e => {
        if (active.size && ![...active].every(t => e.tags.includes(t))) return false;
        if (!q) return true;
        return [e.id, e.date, e.title, e.status, e.takeaway, e.tags.join(" ")].join(" ").toLowerCase().includes(q);
      });
      body.textContent = "";
      rows.forEach(e => {
        const tr = el("tr");
        const idCell = el("td", { class: "id" }); idCell.append(el("a", { href: e.file }, e.id)); tr.append(idCell);
        tr.append(el("td", { class: "date" }, e.date));
        const t = el("td", { class: "title" }); t.append(el("a", { href: e.file }, e.title)); tr.append(t);
        const s = el("td", { class: "state" }); s.append(el("span", { class: `status ${e.status}` }, e.status)); tr.append(s);
        const tg = el("td", { class: "tagcell" }); const box = el("div", { class: "tags" });
        e.tags.forEach(x => box.append(el("span", { class: "tag" }, x))); tg.append(box); tr.append(tg);
        tr.append(el("td", { class: "take" }, e.takeaway));
        body.append(tr);
      });
      count.textContent = `${rows.length} of ${all.length}`;
    }
    input.addEventListener("input", draw);
    draw();
  }

  // ---- charts -------------------------------------------------------------------------
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  // Ticks at a 1/2/2.5/5 × 10^k step, and the fewest decimals that keep every label distinct.
  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    const mag = Math.pow(10, Math.floor(Math.log10(span / n)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
    let decimals = 0;
    while (Math.abs(step * 10 ** decimals - Math.round(step * 10 ** decimals)) > 1e-6) decimals++;
    return { ticks, step, decimals };
  }
  // One unit per axis: every label in thousands ("5k", "2.5k") or none.
  function xLabels({ ticks, step, decimals }) {
    const thousands = ticks.some(v => Math.abs(v) >= 1000) && step >= 100 && step % 100 === 0;
    return ticks.map(v => thousands ? (v === 0 ? "0" : `${+(v / 1000).toFixed(1)}k`) : v.toFixed(decimals));
  }

  // Validates the spec and fixes the data ranges once; draw() only lays out for a width.
  function prepareChart(fig) {
    const spec = JSON.parse(fig.querySelector('script[type="application/json"]').textContent);
    const pts = spec.series.flatMap(s => s.points.filter(p => p[1] != null));
    if (!pts.length) throw new Error("chart has no non-null points");
    const xlo = Math.min(...pts.map(p => p[0])), xhi = Math.max(...pts.map(p => p[0]));
    if (!(xhi > xlo)) throw new Error(`every point has x = ${xlo}; a chart needs an x range`);
    const dlo = Math.min(...pts.map(p => p[1])), dhi = Math.max(...pts.map(p => p[1]));
    const pad = (dhi - dlo) * 0.06 || 1;
    const ylo = spec.y.min != null ? spec.y.min : dlo - pad;
    const yhi = spec.y.max != null ? spec.y.max : dhi + pad;
    if (!(yhi > ylo)) throw new Error(`y range ${ylo}…${yhi} is empty`);
    if (dlo < ylo || dhi > yhi) throw new Error(`values ${dlo}…${dhi} fall outside y.min/y.max ${ylo}…${yhi}`);
    const vlines = spec.vlines || [];
    vlines.forEach(v => {
      if (v.x < xlo || v.x > xhi) throw new Error(`vline "${v.label}" at x = ${v.x} is outside the data's x range ${xlo}…${xhi}`);
    });
    const legend = el("div", { class: "legend" });
    spec.series.forEach((s, i) => {
      const key = el("span", s.dash ? { class: "dash" } : {}, s.name);
      key.style.setProperty("--c", `var(--s${(i % 6) + 1})`);
      legend.append(key);
    });
    fig.prepend(legend);
    let drawnWidth = 0;
    function draw() {
      const W = Math.max(320, Math.round(fig.clientWidth - 20));
      if (W === drawnWidth) return;
      drawnWidth = W;
      const narrow = W < 600;
      const H = narrow ? 240 : 300, L = 56, R = 16, T = vlines.length ? 22 : 12, B = 40;
      const X = v => L + (v - xlo) / (xhi - xlo) * (W - L - R);
      const Y = v => T + (1 - (v - ylo) / (yhi - ylo)) * (H - T - B);
      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${spec.y.label} by ${spec.x.label}` });
      const yTicks = niceTicks(ylo, yhi, narrow ? 5 : 6);
      yTicks.ticks.forEach(v => {
        svg.append(svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "grid" }));
        svg.append(svgEl("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v.toFixed(yTicks.decimals)));
      });
      const xTicks = niceTicks(xlo, xhi, narrow ? 4 : 8);
      const xText = xLabels(xTicks);
      xTicks.ticks.forEach((v, i) => svg.append(svgEl("text", { x: X(v), y: H - B + 16, "text-anchor": "middle" }, xText[i])));
      svg.append(svgEl("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }));
      svg.append(svgEl("line", { x1: L, x2: L, y1: T, y2: H - B, class: "axis" }));
      svg.append(svgEl("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, spec.x.label));
      const mid = (T + H - B) / 2;
      svg.append(svgEl("text", { x: 12, y: mid, "text-anchor": "middle", transform: `rotate(-90 12 ${mid})` }, spec.y.label));
      vlines.forEach(v => {
        const x = X(v.x);
        svg.append(svgEl("line", { x1: x, x2: x, y1: T, y2: H - B, class: "vline" }));
        svg.append(svgEl("text", { x, y: T - 6, "text-anchor": x > W - R - 30 ? "end" : "middle" }, v.label));
      });
      spec.series.forEach((s, i) => {
        const color = `var(--s${(i % 6) + 1})`;
        const drawn = j => j >= 0 && j < s.points.length && s.points[j][1] != null;
        let d = "";
        s.points.forEach(([x, y], j) => {
          if (y == null) return;
          d += `${drawn(j - 1) ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
          if (!drawn(j - 1) && !drawn(j + 1)) svg.append(svgEl("circle", { cx: X(x), cy: Y(y), r: 3, fill: color }));
        });
        svg.append(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": s.dash ? "5 4" : "none" }));
      });
      const old = fig.querySelector("svg");
      if (old) old.replaceWith(svg); else legend.after(svg);
    }
    draw();
    new ResizeObserver(draw).observe(fig);
  }

  function renderCharts() {
    const errors = [];
    document.querySelectorAll("figure.chart").forEach(fig => {
      try {
        prepareChart(fig);
      } catch (err) {
        fig.prepend(el("p", { class: "chart-error" }, `Chart not drawn: ${err.message}`));
        errors.push(err);
      }
    });
    throwAll(errors);
  }

  // Rethrows after every part has been attempted, so one failure never hides another.
  function throwAll(errors) {
    if (errors.length === 1) throw errors[0];
    if (errors.length) throw new AggregateError(errors, errors.map(e => e.message).join("; "));
  }

  document.addEventListener("DOMContentLoaded", () => {
    const errors = [];
    const renderData = () => {
      const all = experiments();
      if (document.body.dataset.page === "index") renderIndex(all); else renderPage(all);
    };
    for (const render of [renderData, renderCharts]) {
      try { render(); } catch (err) { errors.push(err); }
    }
    throwAll(errors);
  });
})();
