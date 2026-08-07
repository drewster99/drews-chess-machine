#!/usr/bin/env python3
"""One monitoring tick for the qeu8 resume3 replay run.

Mirrors tick.py's track -> probe enumerated checkpoints -> backfill missed
1000-marks pipeline, but skips replay.render() (matplotlib is absent in this
env; the PNG render is the only thing that needs it) and rebuilds the shared
dashboard via master.py instead. Prints the latest CSV mark.
"""
import os, subprocess
import replay

RUN = "qeu8"
HERE = os.path.dirname(os.path.abspath(__file__))
ALERTS = os.path.expanduser("~/dcm-qeu8-backup/alerts.log")
cfg = replay.REG["runs"][RUN]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bests(rows):
    """All-time best pElo (max) and nll (min) over the given rows."""
    bp = bn = None
    for x in rows:
        pe, nl = _f(x.get("pElo")), _f(x.get("nll"))
        if pe is not None and (bp is None or pe > bp):
            bp = pe
        if nl is not None and (bn is None or nl < bn):
            bn = nl
    return bp, bn


# Snapshot pre-existing records + the cum_steps already present, so a new mark
# is judged with a STRICT comparison against prior probes only (ties never
# re-alert). Records are appended to ALERTS, which a Monitor tails for pushes.
_prev = replay.read_csv(RUN)
best_pelo, best_nll = _bests(_prev)
_seen = {x["cum_step"] for x in _prev}

# 1. track current out-model mark + 1b. probe any preserved-but-unprobed frozen
replay.track(RUN)
replay.probe_backfill(RUN)

# 2. backfill any missed 1000-marks in the latest segment from the log
seg = cfg["segments"][-1]
base = seg["cumstep_base"]
out = os.path.join(replay.MODELS, cfg["out_model"])
st = replay.SegTime(cfg["segments"], RUN)
rows = replay.read_csv(RUN)
filled = 0
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
        rows.append(dict(
            cum_step=cum, meta_step=meta, segment=si, elapsed_train_sec=el,
            wallclock_iso=clk, ms_per_step=met.get("ms", ""), pElo="", nll="",
            loss=met.get("loss", ""), pLoss=met.get("pLoss", ""), vLoss=met.get("vLoss", ""),
            legalMass=round(1 - met["pIllM"], 4) if "pIllM" in met else "",
            pIllM=met.get("pIllM", ""), bn1Mean="", gNorm=met.get("gNorm", ""),
            sae2="", eff_alpha="", pLogit_mean="", pLogit_peak="",
            frozen_file="", note="log-backfill"))
        filled += 1
    if filled:
        replay.write_csv(RUN, rows)

# 2b. record detection — any newly-probed mark that strictly beats the prior
# all-time best pElo (max) or nll (min) is a record. Append to ALERTS (a Monitor
# tails it and pushes). Running bests so multiple new rows compare correctly.
rows = replay.read_csv(RUN)
msgs = []
bp, bn = best_pelo, best_nll
for x in rows:
    if x["cum_step"] in _seen:
        continue
    pe, nl = _f(x.get("pElo")), _f(x.get("nll"))
    if pe is not None:
        if bp is None or pe > bp:
            prev = f"{bp:.1f}" if bp is not None else "none"
            msgs.append(f"RECORD pElo {pe:.1f} (prev {prev}) @ cum {x['cum_step']} (resume3 meta {x['meta_step']})")
            bp = pe
    if nl is not None:
        if bn is None or nl < bn:
            prev = f"{bn:.4f}" if bn is not None else "none"
            msgs.append(f"RECORD nll {nl:.4f} (prev {prev}) @ cum {x['cum_step']} (resume3 meta {x['meta_step']})")
            bn = nl
if msgs:
    os.makedirs(os.path.dirname(ALERTS), exist_ok=True)
    with open(ALERTS, "a") as fh:
        for m in msgs:
            fh.write(m + "\n")

# 3. rebuild the shared master dashboard (no matplotlib needed)
subprocess.run(["python3", "master.py"], cwd=HERE, check=False)

# 4. print the latest mark
rows = replay.read_csv(RUN)
if rows:
    r = rows[-1]
    try:
        elh = f"{float(r['elapsed_train_sec'])/3600:.2f}h"
    except (TypeError, ValueError):
        elh = "—"
    print(f"cum {r['cum_step']} (meta {r['meta_step']})  "
          f"pElo {r.get('pElo') or '—'}  nll {r.get('nll') or '—'}  "
          f"elapsed {elh}  (backfilled {filled})")
for m in msgs:
    print("  ** " + m)
