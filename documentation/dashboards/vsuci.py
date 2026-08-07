#!/usr/bin/env python3
"""Train-vs-UCI run tracker — the THIRD dashboard run type (alongside replay.py
and selfplay.py). Rebuilds data/<key>.csv (shared _schema.FIELDS layout) for each
run in vsuci_registry.json, then master.py renders it on the same charts.

Why a separate tracker: a vs-UCI run trains against external UCI engines as move
oracles (both-sides distillation), so it is neither corpus replay nor self-play.
Two things differ from replay.py and are handled here:

  * pElo/nll come from a live-probe JSONL (`pelo_jsonl`), keyed by CUMULATIVE
    step — vs-UCI keeps only a rolling `-latest` checkpoint (no per-1000-step
    frozen files to re-probe), so the JSONL is the trajectory's source of truth.
  * training-side metrics (loss/pLoss/vLoss/gNorm/ms) are parsed from the
    [VS-UCI] step lines. Those lines carry no legalMass/pIllM/bn1Mean/sae2/
    pLogit, so those columns are left blank (the charts simply skip them).

elapsed_train_sec is pause-aware: each segment's own duration is measured from
its [VS-UCI] timestamps (with wall-clock midnight rollovers stitched), and a
later segment's elapsed is offset by the sum of earlier segments' durations, so
the by-time axis is continuous across a warm restart and excludes the stop gap.
The build is a full idempotent rebuild each run (the run is small), so re-running
after new autosaves/probes just extends the CSV.
"""
import os, re, json, csv, sys
from _schema import FIELDS

HERE = os.path.dirname(os.path.abspath(__file__))
REG = json.load(open(os.path.join(HERE, "vsuci_registry.json")))
LOGS = os.path.expanduser(REG["logs_dir"])
DATA = os.path.join(HERE, "data")

STEP_RE = re.compile(
    r"^(\d\d):(\d\d):(\d\d)\.\d+\s+\[VS-UCI\] step=(\d+) loss=([\d.]+) pLoss=([-\d.]+) "
    r"vLoss=([-\d.]+) pEnt=([-\d.]+|--) playedP=([-\d.]+|--) gNorm=([-\d.]+) "
    r"lr=([\d.e-]+) ms=([\d.]+) buf=(\d+)")


def parse_segment(log_path, seg_index, cfg):
    """Return {meta_step: dict(elapsed_in_seg, ms, loss, pLoss, vLoss, gNorm)} and
    the segment's total elapsed seconds. Timestamps are HH:MM:SS only, so a
    decrease vs the previous line is treated as a midnight rollover (+86400).

    `elapsed` is CORRECTED training time, not raw wall clock, via two rules that
    are documented in `_timing_comment` in the registry:

      idle removal (automatic, everywhere) — a logged interval spans `ds` steps
        that each took `ms`, so at most `ds*ms` of it was spent computing. Any
        wall-clock excess beyond that is time the process was not running at all
        (system sleep/suspend) and is dropped. This is provable from the data,
        needs no configuration, and is a no-op on a segment that never slept.

      throttle rescale (only inside declared windows) — a clock-throttled step
        and a genuinely slow step are indistinguishable by duration, and seg1
        contains real ~9s stalls that must survive. So rescaling is applied only
        within an explicitly declared `throttled_windows` entry, and only to
        intervals slower than `throttle_trigger x baseline_ms_per_step`; faster
        intervals inside the window keep their measured time.
    """
    baseline = cfg.get("baseline_ms_per_step")
    trigger = cfg.get("throttle_trigger", 1.25)
    windows = [w for w in cfg.get("throttled_windows", []) if w["segment"] == seg_index]

    pts, prev, day = [], None, 0
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            if "[VS-UCI] step=" not in line:   # skip giant [BATCH-STATS] lines fast
                continue
            m = STEP_RE.match(line)
            if not m:
                continue
            sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            if prev is not None and sec < prev:
                day += 86400
            prev = sec
            pts.append((sec + day, int(m.group(4)), dict(
                ms=float(m.group(12)), loss=float(m.group(5)),
                pLoss=float(m.group(6)), vLoss=float(m.group(7)),
                gNorm=float(m.group(10)))))

    def throttled(step):
        return any(w["from_step"] <= step <= w["to_step"] for w in windows)

    per_step, adj = {}, 0.0
    for i, (abs_sec, meta, met) in enumerate(pts):
        if i:
            t0, s0, _ = pts[i - 1]
            dt, ds = abs_sec - t0, meta - s0
            if ds > 0:
                span = min(dt, ds * met["ms"] / 1000.0)          # drop idle/sleep
                if baseline and throttled(meta) and met["ms"] > trigger * baseline:
                    span = ds * baseline / 1000.0                # undo clock clamp
            else:
                span = dt
            adj += span
        per_step[meta] = dict(elapsed=adj, **met)
    return per_step, adj


def nearest_at(per_step, meta):
    """Metrics for the largest logged meta_step <= meta (per-50 log vs per-1000 marks)."""
    cands = [k for k in per_step if k <= meta]
    return per_step[max(cands)] if cands else None


def load_probes(path):
    """cum_step -> {pElo, nll} from the live-probe JSONL."""
    out = {}
    if not os.path.exists(path):
        sys.stderr.write(f"WARNING: pelo_jsonl missing: {path}\n")
        return out
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        out[int(d["step"])] = {"pElo": d.get("pElo"), "nll": d.get("nll")}
    return out


def build(key, cfg):
    probes = load_probes(os.path.expanduser(cfg["pelo_jsonl"]))
    segs = cfg["segments"]
    rows, elapsed_base = [], 0.0
    for si, seg in enumerate(segs):
        per_step, total = parse_segment(os.path.join(LOGS, seg["log"]), si, cfg)
        base = seg["cumstep_base"]
        seg_max = max(per_step) if per_step else 0
        for meta in range(1000, seg_max + 1, 1000):
            met = nearest_at(per_step, meta)
            if not met:
                continue
            cum = base + meta
            pr = probes.get(cum, {})
            rows.append({
                "cum_step": cum, "meta_step": meta, "segment": si,
                "elapsed_train_sec": round(elapsed_base + met["elapsed"], 1),
                "wallclock_iso": "", "ms_per_step": round(met["ms"], 1),
                "pElo": (f"{pr['pElo']:.1f}" if pr.get("pElo") is not None else ""),
                "nll": (f"{pr['nll']:.4f}" if pr.get("nll") is not None else ""),
                "loss": met["loss"], "pLoss": met["pLoss"], "vLoss": met["vLoss"],
                "legalMass": "", "pIllM": "", "bn1Mean": "", "gNorm": met["gNorm"],
                "sae2": "", "eff_alpha": "", "pLogit_mean": "", "pLogit_peak": "",
                "frozen_file": "", "note": (seg["label"] if meta == 1000 else ""),
            })
        elapsed_base += total
    return rows


def write_csv(key, rows):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, f"{key}.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def main():
    for key, cfg in REG["runs"].items():
        rows = build(key, cfg)
        p = write_csv(key, rows)
        peak = max((float(r["pElo"]) for r in rows if r["pElo"]), default=0.0)
        last = rows[-1] if rows else {}
        print(f"{key}: {len(rows)} marks -> {os.path.relpath(p, HERE)} · "
              f"peak pElo {peak:.0f} · to cum {last.get('cum_step', 0)} "
              f"({float(last.get('elapsed_train_sec', 0))/3600:.1f}h)")


if __name__ == "__main__":
    main()
