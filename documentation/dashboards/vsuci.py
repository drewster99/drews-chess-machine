#!/usr/bin/env python3
"""Train-vs-UCI run tracker — the THIRD dashboard run type (alongside replay.py
and selfplay.py). Rebuilds data/<key>.csv (shared _schema.FIELDS layout) for each
run in vsuci_registry.json, then master.py renders it on the same charts.

Why a separate tracker: a vs-UCI run trains against external UCI engines as move
oracles (both-sides distillation), so it is neither corpus replay nor self-play.
Two things differ from replay.py and are handled here:

  * pElo/nll come from a live-probe JSONL (`pelo_jsonl`), keyed by CUMULATIVE
    step — the registered run kept only a rolling `-latest` checkpoint (no
    per-1000-step frozen files to re-probe), so the JSONL is the trajectory's
    source of truth. Current builds save `.dcmsession` folders instead of the
    rolling file (`…-vsuci-periodic/-final/-abort/-health-stop`, trainer state in
    `trainer.safetensors`) and, with --enumerate-checkpoints, `-vsuci-step<N>`
    files; --derive-registry reads both.
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
import os, re, json, sys, argparse
from _schema import FIELDS
# Crash-safe, compare-and-swap, no-silent-shrink replace of data/<run>.csv.
from _guarded_csv import replace_rows, replay_row_key, snapshot_of

HERE = os.path.dirname(os.path.abspath(__file__))
REG = json.load(open(os.path.join(HERE, "vsuci_registry.json")))
LOGS = os.path.expanduser(REG["logs_dir"])
DATA = os.path.join(HERE, "data")

# A [VS-UCI] step line. `pLogitMean=` / `vLogitMean=` sit between `playedP=` and
# `gNorm=` in lines from builds that log them and are absent from older ones, so
# both forms match. `trainerStep=` trails the line (after `mom=` and an optional
# `lrCyc…`), so it is found by its own search (TRAINER_STEP_RE) on a matched line.
STEP_RE = re.compile(
    r"^(?P<hh>\d\d):(?P<mm>\d\d):(?P<ss>\d\d)\.\d+\s+\[VS-UCI\] step=(?P<step>\d+) "
    r"loss=(?P<loss>[\d.]+) pLoss=(?P<pLoss>[-\d.]+) "
    r"vLoss=(?P<vLoss>[-\d.]+) pEnt=(?:[-\d.]+|--) playedP=(?:[-\d.]+|--) "
    r"(?:pLogitMean=(?:[-\d.]+|--) vLogitMean=(?:[-\d.]+|--) )?gNorm=(?P<gNorm>[-\d.]+) "
    r"lr=(?:[\d.e-]+) ms=(?P<ms>[\d.]+) buf=(?:\d+)")
TRAINER_STEP_RE = re.compile(r" trainerStep=(\d+)")


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
        The bound uses the later line's single-step `ms`; with time-based step
        lines hundreds of steps apart (the cadence after trainer step 1000) it
        rests on that one step's duration, a coarser bound than with lines every
        50 steps.

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
            sec = int(m.group("hh")) * 3600 + int(m.group("mm")) * 60 + int(m.group("ss"))
            if prev is not None and sec < prev:
                day += 86400
            prev = sec
            trainer = TRAINER_STEP_RE.search(line)
            pts.append((sec + day, int(m.group("step")), dict(
                ms=float(m.group("ms")), loss=float(m.group("loss")),
                pLoss=float(m.group("pLoss")), vLoss=float(m.group("vLoss")),
                gNorm=float(m.group("gNorm")),
                trainerStep=int(trainer.group(1)) if trainer else None)))

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


def marks_of(per_step):
    """[(meta_step, metrics)] of a segment's 1000-step marks, in step order.

    A log whose step lines carry `trainerStep=` (builds that write lines on the
    trainer-step cadence, which saves and names checkpoints at trainer-step
    multiples of 1000) gives a mark at each line whose trainer step is a multiple
    of 1000 — a resumed segment's marks then sit where its checkpoints are, not at
    segment multiples of 1000. A log whose lines carry no `trainerStep=` (every
    registered run's) keeps the marks at segment multiples of 1000, each taking the
    nearest line at or below it, as before."""
    if any(met["trainerStep"] is not None for met in per_step.values()):
        return [(meta, met) for meta, met in sorted(per_step.items())
                if met["trainerStep"] is not None and met["trainerStep"] % 1000 == 0]
    seg_max = max(per_step) if per_step else 0
    marks = []
    for meta in range(1000, seg_max + 1, 1000):
        met = nearest_at(per_step, meta)
        if met:
            marks.append((meta, met))
    return marks


def load_probes(path):
    """cum_step -> {pElo, nll} from the live-probe JSONL.

    The file is the only record of these probes, so it must exist and every line must
    parse; the one exception is an unterminated last line (an append still in flight),
    which is reported and skipped."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"pelo_jsonl missing: {path}; a rebuild without it would blank every pElo/NLL")
    out = {}
    with open(path) as fh:
        lines = fh.readlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError as error:
            if number == len(lines) and not line.endswith("\n"):
                sys.stderr.write(f"{path}:{number}: unterminated last line (an append in flight); skipped\n")
                continue
            raise ValueError(f"{path}:{number}: not valid JSON ({error})") from error
        out[int(d["step"])] = {"pElo": d.get("pElo"), "nll": d.get("nll")}
    return out


def build(key, cfg):
    probes = load_probes(os.path.expanduser(cfg["pelo_jsonl"]))
    segs = cfg["segments"]
    rows, elapsed_base = [], 0.0
    for si, seg in enumerate(segs):
        per_step, total = parse_segment(os.path.join(LOGS, seg["log"]), si, cfg)
        base = seg["cumstep_base"]
        for position, (meta, met) in enumerate(marks_of(per_step)):
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
                # The segment's label on its first mark: meta 1000 on a log without
                # trainerStep= (as before), the first trainer-step mark otherwise.
                "frozen_file": "", "note": (seg["label"] if (meta == 1000 if met["trainerStep"] is None
                                                             else position == 0) else ""),
            })
        elapsed_base += total
    return rows


def write_csv(key, rows, snapshot, allow_shrink):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, f"{key}.csv")
    replace_rows(p, rows, FIELDS, snapshot, replay_row_key, allow_shrink=allow_shrink)
    return p


def main():
    ap = argparse.ArgumentParser(description="Rebuild data/<run>.csv for every train-vs-UCI run.")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="allow a rebuild to drop rows / blank values a CSV holds (each one is printed first)")
    ap.add_argument("--derive-registry", metavar="DIR", nargs="+",
                    help="instead of rebuilding: derive segment bases from the lineage records (format v7+) "
                         "of the train-vs-UCI files in each DIR — its .safetensors files (step checkpoints) and "
                         "its .dcmsession folders (session saves), e.g. Models/ and Sessions/ — and diff them "
                         "against vsuci_registry.json")
    ap.add_argument("--write", action="store_true",
                    help="with --derive-registry: fill the derived values the registry lacks "
                         "(refused if anything conflicts)")
    args = ap.parse_args()
    if args.write and not args.derive_registry:
        ap.error("--write applies only to --derive-registry")
    if args.derive_registry:
        from _lineage_registry import derive_registry
        import dcm_lineage  # on sys.path once _lineage_registry is imported
        paths = [p for d in args.derive_registry for p in dcm_lineage.model_paths(os.path.expanduser(d))]
        sys.exit(derive_registry(os.path.join(HERE, "vsuci_registry.json"), paths, "vsuci", args.write))
    for key, cfg in REG["runs"].items():
        snapshot = snapshot_of(os.path.join(DATA, f"{key}.csv"))
        rows = build(key, cfg)
        p = write_csv(key, rows, snapshot, args.allow_shrink)
        peak = max((float(r["pElo"]) for r in rows if r["pElo"]), default=0.0)
        last = rows[-1] if rows else {}
        print(f"{key}: {len(rows)} marks -> {os.path.relpath(p, HERE)} · "
              f"peak pElo {peak:.0f} · to cum {last.get('cum_step', 0)} "
              f"({float(last.get('elapsed_train_sec', 0))/3600:.1f}h)")


if __name__ == "__main__":
    main()
