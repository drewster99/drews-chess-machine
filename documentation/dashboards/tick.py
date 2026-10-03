#!/usr/bin/env python3
"""One per-mark monitoring tick for a run: track -> backfill missed 1000-marks
from the log -> render dashboard -> print the FULL data table from the CSV
(authoritative; never hand-typed). Usage: python3 tick.py <run>

A checkpoint that cannot be probed (replay.ProbeFailure from `track`) or a
probe-backfill pass that left checkpoints behind (replay.BackfillIncomplete,
raised after its rows are saved) does not stop the tick: the log backfill, the
render and the table still run, and only then is each failure printed to stderr
and the tick exits 1. Without that, one checkpoint that keeps failing froze the
dashboard at its last render for as long as it stayed on disk. Anything else —
a lineage record contradicting the registry, an unreadable CSV — is not a
per-checkpoint failure and still stops the tick where it happens."""
import sys, os, csv, subprocess, contextlib
import replay

def active_run():
    """Detect which registered run is currently training by matching the live
    replay-corpus process's --out-model against each run's out_model. A failing `ps`
    raises: "could not look" is not the same answer as "no run is training"."""
    ps = subprocess.check_output(["ps", "-Ao", "args="], text=True)
    for line in ps.splitlines():
        if "replay-corpus" in line and "--out-model" in line:
            for r, cfg in replay.REG["runs"].items():
                om = cfg.get("out_model")
                if om and om in line:
                    return r
    return None

# What a tick reports and carries on past (see the module doc).
PER_CHECKPOINT_FAILURES = (replay.ProbeFailure, replay.BackfillIncomplete)

def main(arg):
    """Run one tick for `arg` (a registered run, or "auto" for the one training now).
    Returns the exit status: 1 when a step reported a probe or backfill failure."""
    run = active_run() if arg == "auto" else arg
    if not run:
        print("no active replay run detected — nothing to track")
        return 0
    cfg = replay.REG["runs"][run]
    failures = []

    # 1. track current out-model mark (idempotent)
    try:
        replay.track(run)
    except PER_CHECKPOINT_FAILURES as error:
        failures.append(f"track: {error}")

    # 1b. recover pElo for any preserved-but-unprobed frozen (self-heals monitoring gaps)
    try:
        replay.probe_backfill(run)
    except PER_CHECKPOINT_FAILURES as error:
        failures.append(f"probe-backfill: {error}")

    # 2. backfill any missed 1000-marks in the latest segment from the log
    seg = cfg["segments"][-1]; base = seg["cumstep_base"]
    out = os.path.join(replay.MODELS, cfg["out_model"])
    st = replay.SegTime(cfg["segments"], run)
    rows, snapshot = replay.read_csv_for_update(run); filled = 0
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
                loss=met.get("loss", ""), pLoss=met.get("pLoss", ""), vLoss=met.get("vLoss", ""),
                legalMass=round(1 - met["pIllM"], 4) if "pIllM" in met else "", pIllM=met.get("pIllM", ""),
                bn1Mean="", gNorm=met.get("gNorm", ""), sae2="", eff_alpha="",
                pLogit_mean="", pLogit_peak="", frozen_file="", note="log-backfill (fast-net)"))
            filled += 1
        if filled:
            replay.write_csv(run, rows, snapshot)
    print(f"backfilled {filled}")

    # 3. render dashboard (silence its per-run stdout dump; we only want this run's table)
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

    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "auto"))
