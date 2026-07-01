#!/usr/bin/env python3
"""
Standardized tracker / renderer for DCM corpus-replay runs.

One CSV per run under data/<run>.csv (canonical store); per-run config in
registry.json. Every row carries BOTH a step axis (cum_step / meta_step) and a
wall-time axis (elapsed_train_sec / wallclock_iso), so charts can be drawn vs
step or vs training time. elapsed_train_sec is PAUSE-AWARE: it is the sum of
completed segments' durations plus time-into-the-current-segment, computed from
[REPLAY] log timestamps, so stop/resume gaps are excluded and a warm-start
(new segment) is recorded explicitly.

Subcommands:
  track <run>     detect the out-model's current meta_step; if it's a new mark,
                  freeze (metadata-labeled) + probe + parse internals + join the
                  [REPLAY] log line + compute elapsed, and append a CSV row.
  migrate <run>   backfill a run's CSV from the legacy loop_state.txt marks,
                  computing elapsed/wallclock/ms from the segment logs.
  render          rebuild the markdown table(s), by-step + by-time PNGs, and the
                  HTML dashboard from all run CSVs + registry.

Idempotent: track/migrate never duplicate a cum_step already present.
"""
import os, re, sys, csv, json, glob, struct, math, argparse, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REG = json.load(open(os.path.join(HERE, "registry.json")))
MODELS = os.path.expanduser(REG["models_dir"])
LOGS = os.path.expanduser(REG["logs_dir"])
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
BIN = ("/Users/andrew/Library/Developer/Xcode/DerivedData/"
       "DrewsChessMachine-duuojsefpqabteeapbtaobbiymfs/Build/Products/Release/"
       "DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine")

FIELDS = ["cum_step", "meta_step", "segment", "elapsed_train_sec", "wallclock_iso",
          "ms_per_step", "pElo", "nll", "pLoss", "vLoss", "legalMass", "pIllM",
          "bn1Mean", "gNorm", "sae2", "eff_alpha", "pLogit_mean", "pLogit_peak",
          "frozen_file", "note"]

# ---------- safetensors ----------
def _st_load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n
        out = {}
        for k, m in hdr.items():
            if k == "__metadata__":
                out["__metadata__"] = m; continue
            s, e = m["data_offsets"]; f.seek(base + s); raw = f.read(e - s)
            if m["dtype"] == "BF16":
                a = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view("<f4")
            else:
                a = np.frombuffer(raw, {"F32": "<f4", "F16": "<f2"}[m["dtype"]]).astype(np.float32)
            out[k] = a.reshape(m["shape"]) if m["shape"] else a
        return out

def meta_step_of(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return int(json.loads(f.read(n))["__metadata__"]["training_step"])

def internals(path, cap):
    W = _st_load(path)
    bn1 = float(abs(W["blocks.0.bn1.running_mean"]).max())
    nblk = len({k.split(".")[1] for k in W if k.startswith("blocks.")})
    effs, tot = [], 0.0
    for b in range(nblk):
        a = float(W[f"blocks.{b}.rezero_alpha"][0]); e = cap * math.tanh(a / cap)
        effs.append(e); tot += e * e
    return bn1, tot, effs

# ---------- probe ----------
def probe(path):
    import subprocess
    out = subprocess.run([BIN, "--probe-model", path, "--probe-set", "wide"],
                         capture_output=True, text=True).stdout
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    d = json.loads(m.group(0))
    return dict(pElo=d.get("pElo"), nll=d.get("nll"),
                pLogit_mean=d.get("policy_logit_abs_max"),
                pLogit_peak=d.get("policy_logit_abs_max_peak"))

# ---------- log parsing ----------
_TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+)\s+\[REPLAY\] step=(\d+)\b")
_METRICS = re.compile(r"pLoss=([\d.]+) vLoss=([\d.]+).*?pIllM=([\d.]+).*?gNorm=([\d.]+).*?ms=([\d.]+)")

def _log_index(logname):
    """Return {meta_step: abs_sec} with midnight rollover handled, + ordered list.
    Missing log -> empty (elapsed will be left blank)."""
    path = os.path.join(LOGS, logname)
    idx, order = {}, []
    if not os.path.exists(path):
        return idx, order
    day, prev = 0, None
    for line in open(path, errors="replace"):
        m = _TS.match(line)
        if not m:
            continue
        h, mi, s, ms, step = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
        sod = h * 3600 + mi * 60 + s + int(ms) / (10 ** len(ms))
        if prev is not None and sod < prev - 1:
            day += 1
        prev = sod
        absolute = day * 86400 + sod
        idx[step] = absolute
        order.append((step, absolute))
    return idx, order

def _metrics_at(logname, meta_step):
    """pLoss/vLoss/pIllM/gNorm/ms from the [REPLAY] step=<meta> line (nearest <=)."""
    path = os.path.join(LOGS, logname)
    if not os.path.exists(path):
        return {}
    best = None
    for line in open(path, errors="replace"):
        mm = re.search(r"\[REPLAY\] step=(\d+)\b", line)
        if not mm:
            continue
        st = int(mm.group(1))
        if st <= meta_step:
            mt = _METRICS.search(line)
            if mt:
                best = (st, dict(pLoss=float(mt[1]), vLoss=float(mt[2]),
                                 pIllM=float(mt[3]), gNorm=float(mt[4]), ms=float(mt[5])))
        else:
            break
    return best[1] if best else {}

class SegTime:
    """Pause-aware elapsed-time across a run's segments."""
    def __init__(self, segments):
        self.segs = segments
        self.idx, self.first, self.dur = [], [], []
        prior = 0.0
        for sg in segments:
            idx, order = _log_index(sg["log"])
            first = order[0][1] if order else 0.0
            last = order[-1][1] if order else 0.0
            self.idx.append(idx); self.first.append(first)
            self.dur.append(prior)          # cumulative seconds before this segment
            prior += (last - first)

    def seg_for(self, cumstep):
        # last segment whose cumstep_base <= cumstep
        best = 0
        for i, sg in enumerate(self.segs):
            if sg["cumstep_base"] <= cumstep:
                best = i
        return best

    def elapsed_and_clock(self, cumstep):
        si = self.seg_for(cumstep)
        meta = cumstep - self.segs[si]["cumstep_base"]
        idx = self.idx[si]
        if not idx:                       # segment log missing -> no time axis
            return "", "", meta, si
        if meta in idx:
            abs_sec = idx[meta]
        else:  # nearest <= meta (e.g. abort save between log lines)
            le = [k for k in idx if k <= meta]
            abs_sec = idx[max(le)] if le else self.first[si]
        elapsed = self.dur[si] + (abs_sec - self.first[si])
        # wallclock_iso reconstructed from segment date + abs offset within segment day-space
        d0 = datetime.datetime.strptime(self.segs[si]["date"], "%Y%m%d")
        clock = d0 + datetime.timedelta(seconds=abs_sec)
        return round(elapsed, 1), clock.isoformat(timespec="seconds"), meta, si

# ---------- CSV ----------
def csv_path(run):
    return os.path.join(DATA, f"{run}.csv")

def read_csv(run):
    p = csv_path(run)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

def write_csv(run, rows):
    rows = sorted(rows, key=lambda r: int(r["cum_step"]))
    with open(csv_path(run), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

def has_step(rows, cum):
    return any(int(r["cum_step"]) == cum for r in rows)

# ---------- track ----------
def freeze(run, cfg, meta):
    base = cfg["segments"][-1]["cumstep_base"]
    cum = base + meta
    src = os.path.join(MODELS, cfg["out_model"])
    dst = os.path.join(MODELS, cfg["frozen_glob"].replace("step*", f"step{cum}"))
    if not os.path.exists(dst):
        import shutil; shutil.copy2(src, dst)
    return cum, dst

def track(run):
    cfg = REG["runs"][run]
    src = os.path.join(MODELS, cfg["out_model"])
    if not os.path.exists(src):
        print(f"{run}: out-model not found ({cfg['out_model']})"); return
    meta = meta_step_of(src)
    base = cfg["segments"][-1]["cumstep_base"]; cum = base + meta
    rows = read_csv(run)
    if has_step(rows, cum):
        print(f"{run}: cum_step {cum} already tracked (meta {meta}) — no-op"); return
    cum, frozen = freeze(run, cfg, meta)
    pr = probe(frozen)
    bn1, sae2, effs = internals(frozen, cfg["rezero_cap"])
    st = SegTime(cfg["segments"]); elapsed, clock, m2, si = st.elapsed_and_clock(cum)
    met = _metrics_at(cfg["segments"][si]["log"], meta)
    lm = round(1 - met["pIllM"], 4) if "pIllM" in met else ""
    row = dict(cum_step=cum, meta_step=meta, segment=si, elapsed_train_sec=elapsed,
               wallclock_iso=clock, ms_per_step=met.get("ms", ""),
               pElo=round(pr.get("pElo"), 2) if pr.get("pElo") else "",
               nll=round(pr.get("nll"), 4) if pr.get("nll") else "",
               pLoss=met.get("pLoss", ""), vLoss=met.get("vLoss", ""),
               legalMass=lm, pIllM=met.get("pIllM", ""),
               bn1Mean=round(bn1, 4), gNorm=met.get("gNorm", ""),
               sae2=round(sae2, 4), eff_alpha=";".join(f"{e:.4f}" for e in effs),
               pLogit_mean=round(pr.get("pLogit_mean"), 3) if pr.get("pLogit_mean") else "",
               pLogit_peak=pr.get("pLogit_peak", ""),
               frozen_file=os.path.basename(frozen), note="")
    rows.append(row); write_csv(run, rows)
    eh = f"{elapsed/3600:.2f}h" if isinstance(elapsed, (int, float)) else "n/a (log gone)"
    print(f"{run}: tracked cum_step {cum} (meta {meta}) pElo={row['pElo']} "
          f"elapsed={elapsed}s ({eh}) seg={si}")

# ---------- migrate from legacy loop_state.txt ----------
LOOP_STATE = os.path.join(HERE, "loop_state.txt")  # symlinked/copied from scratchpad if present

def _kv(line, key):
    m = re.search(rf"{re.escape(key)}=([0-9.]+)", line); return m.group(1) if m else ""

V5_DOC = os.path.join(HERE, "..", "v5-layernorm-output.md")

def migrate_v5():
    """v5 lives in the markdown table (per-subrun step), not loop_state.
    cum_step = subrun_step + offset. Three warm-start segments; elapsed left
    blank unless segment logs are present."""
    cfg = REG["runs"]["v5"]
    off = cfg["v5doc_offsets"]
    seg_for_name = {"wd1e-4": 0, "wd5e-4": 1, "m0.93": 2}
    st = SegTime(cfg["segments"])
    rows, seen = [], set()
    for line in open(V5_DOC):
        m = re.match(r"\|\s*(\*\*)?(wd1e-4|wd5e-4|m0\.93)(\*\*)?\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        name = m.group(2)
        c = [x.strip().replace("**", "") for x in m.group(4).split("|")]
        if len(c) < 10:
            continue
        meta = int(float(c[0])); cum = meta + off[name]
        if cum in seen:
            continue
        seen.add(cum)
        si = seg_for_name[name]
        elapsed, clock, _, _ = st.elapsed_and_clock(cum)
        pl = c[9].split("/")
        frozen = cfg["frozen_glob"].replace("step*", f"step{cum}")
        rows.append(dict(
            cum_step=cum, meta_step=meta, segment=si,
            elapsed_train_sec=elapsed, wallclock_iso=clock, ms_per_step="",
            pElo=c[1], nll=c[2], pLoss=c[3], vLoss=c[4], legalMass=c[5],
            pIllM="", bn1Mean=c[6], gNorm=c[7], sae2=c[8], eff_alpha="",
            pLogit_mean=pl[0], pLogit_peak=(pl[1] if len(pl) > 1 else ""),
            frozen_file=os.path.basename(frozen), note=name))
    write_csv("v5", rows)
    print(f"migrate v5: {len(rows)} rows -> {csv_path('v5')} (from v5-layernorm-output.md)")

def migrate(run, loop_state=None):
    if run == "v5" or REG["runs"][run].get("source") == "v5doc":
        return migrate_v5()
    cfg = REG["runs"][run]
    tag = {"mini2b": "MINI", "coxw": "COXW", "ykkk": "YKKK", "t97x": "T97X"}[run]
    ls = loop_state or LOOP_STATE
    if not os.path.exists(ls):
        print(f"migrate: loop_state not found at {ls}"); return
    st = SegTime(cfg["segments"])
    rows, seen = [], set()
    for line in open(ls):
        m = re.match(rf"#\s*{tag}\s+(\d+)\b", line)
        if not m:
            continue
        cum = int(m.group(1))
        if cum in seen:
            continue
        seen.add(cum)
        pl = re.search(r"pLogitAbsMax=([0-9.]+)/([0-9.]+)", line)
        elapsed, clock, meta, si = st.elapsed_and_clock(cum)
        met = _metrics_at(cfg["segments"][si]["log"], meta)
        frozen = cfg["frozen_glob"].replace("step*", f"step{cum}")
        if "FINAL" in line:
            frozen = frozen.replace("-frozen", "-FINAL-frozen")
        rows.append(dict(
            cum_step=cum, meta_step=meta, segment=si,
            elapsed_train_sec=elapsed, wallclock_iso=clock,
            ms_per_step=met.get("ms", ""),
            pElo=_kv(line, "pElo"), nll=_kv(line, "nll"),
            pLoss=_kv(line, "pLoss") or met.get("pLoss", ""),
            vLoss=_kv(line, "vLoss") or met.get("vLoss", ""),
            legalMass=_kv(line, "lm"), pIllM=_kv(line, "pIllM"),
            bn1Mean=_kv(line, "bn1Mean"), gNorm=_kv(line, "gNorm"),
            sae2=_kv(line, "Σαeff²"),
            eff_alpha=";".join(re.search(r"effα\[([0-9.,]+)\]", line).group(1).split(",")) if re.search(r"effα\[([0-9.,]+)\]", line) else "",
            pLogit_mean=pl.group(1) if pl else "", pLogit_peak=pl.group(2) if pl else "",
            frozen_file=os.path.basename(frozen),
            note="FINAL" if "FINAL" in line else ("log-backfill" if not _kv(line, "pElo") else "")))
    write_csv(run, rows)
    print(f"migrate {run}: {len(rows)} rows -> {csv_path(run)} "
          f"(elapsed {rows[0]['elapsed_train_sec']}..{rows[-1]['elapsed_train_sec']}s)")

# ---------- render ----------
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def render():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = {}
    for run in REG["runs"]:
        rows = read_csv(run)
        for r in rows:
            for k in r:
                pass
        runs[run] = rows
    # ---- markdown table + per-run doc (mini2b focus) ----
    def md_table(rows):
        hdr = ("| step | elapsed(h) | pElo | nll | pLoss | vLoss | legalMass | bn1Mean | "
               "gNorm | Σαeff² | pLogit μ/peak |\n|" + "---|" * 11)
        body = []
        for i, r in enumerate(rows):
            el = _f(r["elapsed_train_sec"])
            elh = f"{el/3600:.2f}" if el is not None else "—"
            pl = f"{r['pLogit_mean']}/{r['pLogit_peak']}" if r["pLogit_mean"] else "—"
            cells = [r["cum_step"], elh,
                     r["pElo"] or "—", r["nll"] or "—", r["pLoss"] or "—", r["vLoss"] or "—",
                     r["legalMass"] or "—", r["bn1Mean"] or "—", r["gNorm"] or "—",
                     r["sae2"] or "—", pl]
            if i == len(rows) - 1:
                cells = [f"**{c}**" for c in cells]
            body.append("| " + " | ".join(str(c) for c in cells) + " |")
        return hdr + "\n" + "\n".join(body)

    # ---- charts: every metric, by-step AND by-time ----
    COLOR = {r: REG["runs"][r]["color"] for r in REG["runs"]}
    LAB = {r: REG["runs"][r]["label"] for r in REG["runs"]}
    metrics = [("pElo", "pElo"), ("nll", "nll"), ("pLoss", "pLoss"), ("vLoss", "vLoss"),
               ("legalMass", "legalMass"), ("gNorm", "gNorm"), ("bn1Mean", "bn1Mean (log y)"),
               ("sae2", "Σαeff²"), ("pLogit_mean", "pLogitAbsMax mean")]
    import io, base64
    def png(fig):
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig); return base64.b64encode(b.getvalue()).decode()

    def _ema(ys, span):
        a = 2.0 / (span + 1); s = None; out = []
        for y in ys:
            s = y if s is None else a * y + (1 - a) * s
            out.append(s)
        return out

    def chart(key, title, xkey, xlabel, logy=False, xmax=None, ema=None, overlay_ema=None):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for run, rows in runs.items():
            xs, ys = [], []
            for r in rows:
                y = _f(r[key]); x = _f(r[xkey])
                if y is None or x is None:
                    continue
                xs.append(x / 3600 if xkey == "elapsed_train_sec" else x); ys.append(y)
            if xs:
                if overlay_ema:
                    ax.plot(xs, ys, "-o", ms=2.0, lw=0.8, color=COLOR[run], alpha=0.3)
                    ax.plot(xs, _ema(ys, overlay_ema), "-", lw=1.9,
                            color=COLOR[run], label=LAB[run])
                elif ema:
                    ax.plot(xs, _ema(ys, ema), "-", lw=1.8, color=COLOR[run], label=LAB[run])
                else:
                    ax.plot(xs, ys, "-o", ms=2.4, lw=1.0, color=COLOR[run], label=LAB[run])
        if logy:
            ax.set_yscale("log")
        if xmax:
            ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel); ax.set_ylabel(key); ax.set_title(title)
        ax.grid(alpha=.22); ax.legend(fontsize=9, handlelength=2.6, markerscale=2.2)
        return png(fig)

    # early-window zoom target: clip x to the shortest substantial (newest/active) run
    # so short runs fill the frame. Tracks min per-run max-step among runs past 5k.
    run_max = {run: max((int(r["cum_step"]) for r in rows), default=0)
               for run, rows in runs.items() if rows}
    substantial = [s for s in run_max.values() if s >= 5000]
    zx = 1.2 * min(substantial) if substantial else None
    active = [r for r, s in run_max.items() if s == min(substantial)] if substantial else []

    # one chart set per metric: 10-pt EMA (foreground) over the raw marks at 30%
    # opacity (background), by SGD step and by training time. Under the pElo pair,
    # a collapsed <details> holds the same view clipped to the early-window zoom.
    ch = ""
    for n, (key, title) in enumerate(metrics, 1):
        logy = key == "bn1Mean"
        b_step = chart(key, f"{title} vs step", "cum_step", "cumulative SGD step",
                       logy, overlay_ema=10)
        b_time = chart(key, f"{title} vs training time", "elapsed_train_sec",
                       "elapsed training time (h)", logy, overlay_ema=10)
        ch += (f"<h2>{n} · {title}</h2><div class=pair>"
               f"<figure><figcaption>by SGD step</figcaption>"
               f"<img src='data:image/png;base64,{b_step}'></figure>"
               f"<figure><figcaption>by training time</figcaption>"
               f"<img src='data:image/png;base64,{b_time}'></figure></div>")
        if key == "pElo" and zx:
            z_step = chart(key, f"{title} vs step — zoom ≤{int(zx):,}", "cum_step",
                           "cumulative SGD step", overlay_ema=10, xmax=zx)
            ch += (f"<details><summary>▸ early-window zoom — x clipped to ≤{int(zx):,} steps "
                   f"(tracks shortest active run: {', '.join(active)})</summary>"
                   f"<figure><figcaption>pElo by SGD step — zoom</figcaption>"
                   f"<img src='data:image/png;base64,{z_step}'></figure></details>")
    cols = ["step", "elapsed(h)", "pElo", "nll", "pLoss", "vLoss", "legalMass",
            "bn1Mean", "gNorm", "Σαeff²", "pLogit μ/peak"]
    def html_table(rows):
        head = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        body = []
        for i, r in enumerate(rows):
            el = _f(r["elapsed_train_sec"]); elh = f"{el/3600:.2f}" if el is not None else "—"
            pl = f"{r['pLogit_mean']}/{r['pLogit_peak']}" if r["pLogit_mean"] else "—"
            vals = [r["cum_step"], elh, r["pElo"] or "—", r["nll"] or "—", r["pLoss"] or "—",
                    r["vLoss"] or "—", r["legalMass"] or "—", r["bn1Mean"] or "—",
                    r["gNorm"] or "—", r["sae2"] or "—", pl]
            last = i == len(rows) - 1
            tag = "<td><b>" if last else "<td>"; end = "</b></td>" if last else "</td>"
            body.append("<tr>" + "".join(f"{tag}{v}{end}" for v in vals) + "</tr>")
        return "<table>" + head + "".join(body) + "</table>"
    tb = ""
    for run, rows in runs.items():
        if not rows:
            continue
        peak = max((_f(r["pElo"]) for r in rows if _f(r["pElo"])), default=0)
        el = _f(rows[-1]["elapsed_train_sec"])
        tb += (f"<details open><summary><b>{LAB[run]}</b> — {len(rows)} marks, peak pElo {peak:.0f}, "
               f"{(el/3600 if el else 0):.1f}h training</summary>" + html_table(rows) + "</details>")
    # ---- summary: one row per run (arch + final metrics) ----
    def _last(rows, key):
        for r in reversed(rows):
            if _f(r.get(key)) is not None:
                return r[key]
        return "—"
    scols = ["run", "input + stem", "blocks", "heads", "params", "steps", "hrs",
             "final pElo", "final NLL", "final pLogitAbsMax"]
    srows = ""
    for run in REG["runs"]:
        rows = runs.get(run) or []
        cfg = REG["runs"][run]
        stem = cfg.get("arch_stem", cfg.get("arch_summary", ""))
        blocks = cfg.get("arch_blocks", "")
        heads = cfg.get("arch_heads", "")
        acells = (f"<td class=arch>{stem}</td><td class=arch>{blocks}</td>"
                  f"<td class=arch>{heads}</td>")
        if not rows:
            srows += (f"<tr><td>{run}</td>{acells}<td>{cfg['params']:,}</td>"
                      "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>")
            continue
        steps = max(int(r["cum_step"]) for r in rows)
        hrs = max((_f(r["elapsed_train_sec"]) or 0) for r in rows) / 3600
        srows += (f"<tr><td>{run}</td>{acells}<td>{cfg['params']:,}</td>"
                  f"<td>{steps:,}</td><td>{hrs:.1f}</td>"
                  f"<td>{_last(rows,'pElo')}</td><td>{_last(rows,'nll')}</td>"
                  f"<td>{_last(rows,'pLogit_mean')}</td></tr>")
    summ = ("<h2>Summary — architecture &amp; final metrics</h2>"
            "<table class=summary><tr>" + "".join(f"<th>{c}</th>" for c in scols) +
            "</tr>" + srows + "</table>")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>DCM replay runs</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;max-width:min(1900px,96vw);color:#1a1a1a}}
h1{{font-size:21px}}h2{{font-size:14px;color:#444;margin-top:26px}}img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
table{{border-collapse:collapse;font-size:11px;margin:8px 0}}td,th{{border:1px solid #ccc;padding:2px 7px;text-align:right}}
summary{{cursor:pointer;font-size:13px;margin-top:10px}}.cap{{color:#666;font-size:12px}}
.pair{{display:flex;gap:14px;align-items:flex-start;margin-top:6px}}.pair figure{{flex:1;width:50%;margin:0}}
.pair img{{width:100%}}figcaption{{font-size:11px;color:#666;text-align:center;margin-bottom:2px}}
table.summary td,table.summary th{{text-align:left;font-size:11.5px;vertical-align:top}}
table.summary td.arch{{max-width:230px;white-space:normal;color:#333}}</style></head><body>
<h1>DCM — replay-run dashboard</h1>
<p class=cap>Auto-generated {now} from per-run CSVs in <code>data/</code> via <code>replay.py render</code>.
Each metric: 10-point EMA (foreground) over raw marks at 30% opacity (background), shown vs cumulative SGD step and vs elapsed training time (pause-aware). The pElo panel has a collapsible early-window zoom. Reload to refresh.</p>
{summ}
{ch}<h2>Data tables</h2>{tb}</body></html>"""
    out = os.path.join(HERE, "dcm_dashboard.html")
    open(out, "w").write(html)
    # markdown table to stdout for the run we care about
    for run in REG["runs"]:
        if runs.get(run):
            print(f"\n### {run}\n" + md_table(runs[run]))
    print(f"\nrendered {out} ({len(html)//1024} KB)")

# ---------- cli ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("track"); p.add_argument("run")
    p = sub.add_parser("migrate"); p.add_argument("run"); p.add_argument("--loop-state")
    sub.add_parser("render")
    a = ap.parse_args()
    if a.cmd == "track":
        track(a.run)
    elif a.cmd == "migrate":
        migrate(a.run, a.loop_state)
    elif a.cmd == "render":
        render()
