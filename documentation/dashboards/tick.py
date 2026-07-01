#!/usr/bin/env python3
"""One per-mark monitoring tick for a run: track -> backfill missed 1000-marks
from the log -> render dashboard -> print the FULL data table from the CSV
(authoritative; never hand-typed). Usage: python3 tick.py <run>"""
import sys, os, csv
import replay

run = sys.argv[1] if len(sys.argv) > 1 else "ykkk"
cfg = replay.REG["runs"][run]

# 1. track current out-model mark (idempotent)
replay.track(run)

# 2. backfill any missed 1000-marks in the latest segment from the log
seg = cfg["segments"][-1]; base = seg["cumstep_base"]
out = os.path.join(replay.MODELS, cfg["out_model"])
st = replay.SegTime(cfg["segments"])
rows = replay.read_csv(run); filled = 0
if os.path.exists(out):
    cur = replay.meta_step_of(out)
    for meta in range(1000, cur, 1000):
        cum = base + meta
        if replay.has_step(rows, cum):
            continue
        el, clk, _, si = st.elapsed_and_clock(cum)
        met = replay._metrics_at(cfg["segments"][si]["log"], meta)
        if not met:
            continue
        rows.append(dict(cum_step=cum, meta_step=meta, segment=si, elapsed_train_sec=el,
            wallclock_iso=clk, ms_per_step=met.get("ms", ""), pElo="", nll="",
            pLoss=met.get("pLoss", ""), vLoss=met.get("vLoss", ""),
            legalMass=round(1 - met["pIllM"], 4) if "pIllM" in met else "", pIllM=met.get("pIllM", ""),
            bn1Mean="", gNorm=met.get("gNorm", ""), sae2="", eff_alpha="",
            pLogit_mean="", pLogit_peak="", frozen_file="", note="log-backfill (fast-net)"))
        filled += 1
    if filled:
        replay.write_csv(run, rows)
print(f"backfilled {filled}")

# 3. render dashboard (silence its per-run stdout dump; we only want this run's table)
import contextlib
with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
    replay.render()

# 4. print the FULL table straight from the CSV
rows = replay.read_csv(run)
def g(r, k):
    v = r.get(k, ""); return v if v else "—"
def elh(r):
    v = r.get("elapsed_train_sec")
    try: return f"{float(v)/3600:.2f}"
    except (TypeError, ValueError): return "—"
print("\n| step | elapsed(h) | pElo | nll | pLoss | vLoss | legalMass | bn1Mean | gNorm | Σαeff² | pLogit μ/peak | seg |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(rows):
    pl = f"{r['pLogit_mean']}/{r['pLogit_peak']}" if r.get("pLogit_mean") else "—"
    cells = [r["cum_step"], elh(r), g(r, "pElo"), g(r, "nll"), g(r, "pLoss"), g(r, "vLoss"),
             g(r, "legalMass"), g(r, "bn1Mean"), g(r, "gNorm"), g(r, "sae2"), pl, r.get("segment", "")]
    if i == len(rows) - 1:
        cells = [f"**{c}**" for c in cells]
    print("| " + " | ".join(str(c) for c in cells) + " |")
