#!/usr/bin/env python3
"""Self-play (non-replay) run tracker -> data/<key>.csv, schema-compatible with
the replay tracker so master.py graphs both in one dashboard.

Unlike the replay tracker (which probes per-1000-step frozen checkpoints), self-
play runs have NO enumerated per-step checkpoints — the intermediates were pruned
and only one .dcmsession per lineage survives. So this reads the dense `[STATS]`
telemetry the trainer already logged and turns it into a per-step time series.

Per lineage (combined continuations, keyed by champion base ModelID; see
selfplay_registry.json `logs`):
  * cum_step = the `[STATS] steps=` value. That counter is CUMULATIVE across a
    lineage's launches (a resume loads the session and continues counting), so no
    per-launch offset is applied. Resumes occasionally REWIND to a slightly-behind
    session save; collisions are resolved last-writer-wins (the resumed launch's
    weights supersede), then rows are sorted by step.
  * elapsed_train_sec = each launch's own `elapsed=` plus the summed final elapsed
    of all prior launches. Approximate (excludes inter-launch gaps, not intra-
    launch pauses) — the by-STEP charts are the authoritative axis.
  * pElo / nll are written ONLY on the final row, from probing the saved trainer
    weights (registry endpoint_*). There is no per-step pElo — no checkpoints to
    probe — so the pElo chart shows self-play runs as a single terminal point.
  * loss = pLoss + vLoss (the combined-loss column has no direct `[STATS]` field).
  * pLogit_mean carries `[STATS] pLogitAbsMax`. bn1Mean/sae2/eff_alpha/pLogit_peak
    are replay-probe internals with no self-play analog -> left blank.

Run:  python3 selfplay.py
"""
import os, re, csv, json, bisect
from _schema import FIELDS  # single source of the CSV column order (shared with replay.py)

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.expanduser("~/Library/Logs/DrewsChessMachine")
DATA = os.path.join(HERE, "data")

# gNorm is the PRE-clip gradient L2 norm; at an eBNC divergence moment it spikes to
# ~1e10 while the APPLIED update stays bounded by the clip threshold (reg clip=30).
# Those single-step transients would flatten every other run on the shared linear-y
# gNorm/loss charts, so graph-facing values are clamped to these ceilings (well above
# any healthy value: gNorm peaks ~40, loss ~8.5). The clamp and the true value are
# recorded in the row `note`; raw values remain in the logs. (The far larger 1e14
# spikes seen earlier belonged to an unrelated model interleaved in a shared log and
# are now excluded by the champion-base filter in build_run.)
GNORM_CEIL = 100.0
LOSS_CEIL = 20.0

# Nominal random-init pElo baseline for imported self-play runs whose recorded
# probe curve begins late (see build_run): a fresh net scores ~450 near step 1000.
BASELINE_STEP = 1000
BASELINE_PELO = 450

# A launch is a FRESH restart (steps= counter returned to ~1) iff its first
# [STATS] step is at/below this bound. The trainer logs one line PER STEP across
# an initial window at least this large, so a genuinely fresh launch is
# guaranteed to emit a first line at/below the bound — while a session RESUME
# rewinds to the saved step (typically many thousands) and must overwrite in
# place, not be offset. Keying off this absolute near-start bound (rather than a
# fraction of running_max) is what keeps a mid-range resume-rewind from being
# mis-scored as a fresh restart and double-counted onto the cumulative axis.
RESET_FRESH_STEP = 500


def _blank_row(cum, meta, seg):
    """An all-blank CSV row (used for probe points that have no [STATS] row)."""
    r = {k: "" for k in FIELDS}
    r["cum_step"] = cum
    r["meta_step"] = meta
    r["segment"] = seg
    return r

TS = re.compile(r"^(\d\d):(\d\d):(\d\d)")


def base_of(mid):
    """Lineage base `yyyymmdd-N-XXXX` of a ModelID (drops the -generation suffix)."""
    m = re.match(r"(\d{8}-\d+-[A-Za-z0-9]{4})", mid or "")
    return m.group(1) if m else None


def _f(pat, s):
    m = re.search(pat, s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None  # e.g. "--" placeholder on non-diagnostic lines


def elapsed_to_sec(line):
    m = re.search(r"\belapsed=(\d+):(\d\d):(\d\d)\b", line)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mi * 60 + s


def parse_stats(line):
    """Extract the graphed metrics from one [STATS] line. Returns None if it has
    no step= (not a real training stat line)."""
    ms = re.search(r"\bsteps=(\d+)", line)
    if not ms:
        return None
    d = {"step": int(ms.group(1))}
    mc = re.search(r"\bchampion=(\S+)", line)
    d["champion"] = mc.group(1) if mc else None
    d["elapsed_sec"] = elapsed_to_sec(line)
    d["ms"] = _f(r"timing=\(step=([\d.]+)", line)
    d["pLoss"] = _f(r"\bpLoss=([+\-\d.]+)", line)
    d["vLoss"] = _f(r"\bvLoss=([+\-\d.]+)", line)
    d["legalMass"] = _f(r"\blegalMass=([+\-\d.]+)", line)
    d["pIllM"] = _f(r"\bpIllM=([+\-\d.]+)", line)
    d["gNorm"] = _f(r"\bgNorm=([+\-\d.]+)", line)
    d["pLogit"] = _f(r"pLogitAbsMax=([\d.]+)", line)
    return d


def iso_of(log, hms, prev_hms, day_offset):
    """Build an ISO-ish wallclock from the log's start date + line HH:MM:SS,
    rolling the date forward each time the clock wraps past midnight."""
    date = log[8:16]  # YYYYMMDD from dcm_log_YYYYMMDD-HHMMSS.txt
    if prev_hms is not None and hms < prev_hms:
        day_offset[0] += 1
    y, m, dd = int(date[:4]), int(date[4:6]), int(date[6:8])
    # add day_offset days (naive; good enough for a wallclock label)
    import datetime
    base = datetime.date(y, m, dd) + datetime.timedelta(days=day_offset[0])
    return f"{base.isoformat()}T{hms[0]:02d}:{hms[1]:02d}:{hms[2]:02d}", hms


def build_run(key, cfg):
    rows = {}  # cum_step -> row dict (last-writer-wins over resume rewinds)
    launch_offset = 0.0     # summed prior-launch training seconds
    step_base = 0           # added to raw steps= to keep cum_step monotonic
    running_max = 0         # highest cum_step emitted so far
    resets = 0
    foreign = 0             # [STATS] lines from a different model in a shared log
    expected_base = cfg.get("base_modelID")  # this lineage's champion base
    base_by_raw = {}        # raw meta_step -> step_base applied (for probe merge)
    seg_base = {}           # launch index -> step_base applied to that launch
    for seg_i, log in enumerate(cfg["logs"]):
        path = os.path.join(LOGDIR, log)
        if not os.path.exists(path):
            continue
        seg_max_elapsed = 0.0
        prev_hms = None
        day_offset = [0]
        decided = False     # have we set step_base for this launch yet?
        for line in open(path, errors="replace"):
            if "[STATS]" not in line:
                continue
            st = parse_stats(line)
            if st is None:
                continue
            # Reject lines emitted by a DIFFERENT model that happens to share this
            # log (a launch can interleave an unrelated model's [STATS], e.g. bnP6
            # inside a wTp3 log). Filtering by champion base ModelID keeps the reset
            # heuristic and step axis from being corrupted by the foreign sequence.
            if expected_base and st["champion"] and base_of(st["champion"]) != expected_base:
                foreign += 1
                continue
            # On the launch's first stat line decide whether steps= reset. A raw
            # step back at the near-start bound means a FRESH restart (the counter
            # went back to ~1), so continue numbering after the max; any larger
            # backward jump is a resume-rewind and is left to overwrite its steps
            # in-place. (See RESET_FRESH_STEP: an absolute near-start bound, not a
            # fraction of running_max, so a mid-range rewind can't be mis-scored as
            # a restart and double-counted.)
            if not decided:
                if running_max > RESET_FRESH_STEP and st["step"] <= RESET_FRESH_STEP:
                    step_base = running_max
                    resets += 1
                decided = True
            cum = st["step"] + step_base
            running_max = max(running_max, cum)
            base_by_raw[st["step"]] = step_base
            mt = TS.match(line)
            hms = (int(mt.group(1)), int(mt.group(2)), int(mt.group(3))) if mt else None
            iso = ""
            if hms is not None:
                iso, prev_hms = iso_of(log, hms, prev_hms, day_offset)
            el = st["elapsed_sec"]
            if el is not None:
                seg_max_elapsed = max(seg_max_elapsed, el)
            g_el = (launch_offset + el) if el is not None else None
            loss = (st["pLoss"] + st["vLoss"]) if (st["pLoss"] is not None and st["vLoss"] is not None) else None
            gnorm = st["gNorm"]
            clamp_notes = []
            if gnorm is not None and gnorm > GNORM_CEIL:
                clamp_notes.append(f"gNorm {gnorm:.2g}→{GNORM_CEIL:.0f}")
                gnorm = GNORM_CEIL
            if loss is not None and loss > LOSS_CEIL:
                clamp_notes.append(f"loss {loss:.2g}→{LOSS_CEIL:.0f}")
                loss = LOSS_CEIL
            rows[cum] = {
                "cum_step": cum, "meta_step": st["step"], "segment": seg_i,
                "elapsed_train_sec": round(g_el, 1) if g_el is not None else "",
                "wallclock_iso": iso, "ms_per_step": st["ms"] if st["ms"] is not None else "",
                "pElo": "", "nll": "",
                "loss": round(loss, 5) if loss is not None else "",
                "pLoss": st["pLoss"] if st["pLoss"] is not None else "",
                "vLoss": st["vLoss"] if st["vLoss"] is not None else "",
                "legalMass": st["legalMass"] if st["legalMass"] is not None else "",
                "pIllM": st["pIllM"] if st["pIllM"] is not None else "",
                "bn1Mean": "", "gNorm": gnorm if gnorm is not None else "",
                "sae2": "", "eff_alpha": "",
                "pLogit_mean": st["pLogit"] if st["pLogit"] is not None else "",
                "pLogit_peak": "", "frozen_file": "", "note": "; ".join(clamp_notes),
            }
        seg_base[seg_i] = step_base
        launch_offset += seg_max_elapsed

    # (cum, elapsed) from the [STATS] rows — to interpolate elapsed onto any pElo-
    # only row (probe points, baseline) so the by-TIME charts render them too.
    el_pts = sorted((r["cum_step"], r["elapsed_train_sec"])
                    for r in rows.values() if r["elapsed_train_sec"] != "")
    el_cums = [c for c, _ in el_pts]

    def interp_elapsed(cum):
        if not el_cums:
            return ""
        i = bisect.bisect_left(el_cums, cum)
        if i == 0:
            return el_pts[0][1]
        if i >= len(el_cums):
            return el_pts[-1][1]
        (c0, e0), (c1, e1) = el_pts[i - 1], el_pts[i]
        return e0 if c1 == c0 else round(e0 + (e1 - e0) * (cum - c0) / (c1 - c0), 1)

    def put_pelo(cum, meta, pelo, nll, note):
        row = rows.get(cum)
        if row is None:
            row = _blank_row(cum, meta, "")
            row["elapsed_train_sec"] = interp_elapsed(cum)
        row["pElo"] = pelo
        row["nll"] = nll
        if note:
            row["note"] = "; ".join(x for x in [row["note"], note] if x)
        rows[cum] = row

    # Merge the in-training probe pElo/NLL curve (selfplay_probe/<key>.csv),
    # recorded by the app's periodic lichess probe during this run. Each probe raw
    # step maps to a cum_step via the same per-launch offset the [STATS] pass applied.
    #
    # A probe CSV MAY carry a `segment` column naming the launch (index into the
    # registry `logs` list) the row came from; when present that is the only exact
    # mapping. A lineage that restarted from step 1 more than once has the SAME raw
    # step number under several different bases, and the flat raw->base dict keeps
    # only the last writer — so without the column every probe row from an earlier
    # restart lands on the last restart's base and the pElo curve is scrambled onto
    # step ranges it never occupied. Rows without the column keep the old behaviour
    # (exact for cumulative lineages, nearest-prior otherwise), which is why the
    # older imported probe curves are unchanged by this.
    #
    # NOTE: this puzzleElo is on the RECORDING BUILD's scale, which differs from the
    # current-binary replay pElo — use the replay/self-play toggle to compare within
    # one consistent scale.
    probe_path = os.path.join(HERE, "selfplay_probe", f"{key}.csv")
    has_traj = os.path.exists(probe_path)
    if has_traj and base_by_raw:
        raws = sorted(base_by_raw)
        for pr in csv.DictReader(open(probe_path)):
            raw = int(pr["step"])
            seg = pr.get("segment", "")
            if seg != "" and int(seg) in seg_base:
                base = seg_base[int(seg)]
            elif raw in base_by_raw:
                base = base_by_raw[raw]
            else:
                i = bisect.bisect_right(raws, raw) - 1
                base = base_by_raw[raws[i]] if i >= 0 else 0
            put_pelo(raw + base, raw, pr["pElo"], pr.get("nll", ""), "")

    # Runs that predate the probe feature (no curve): put the single manual-probe
    # endpoint on the final row so the summary still shows a pElo/NLL value.
    if not has_traj and rows:
        last = max(rows)
        put_pelo(last, rows[last]["meta_step"], cfg.get("endpoint_pElo", ""),
                 cfg.get("endpoint_nll", ""), "trainer probe endpoint")

    # Random-init baseline: imported self-play runs whose recorded pElo begins only
    # LATE (endpoint-only runs, and late-probe runs like KbHZ/bzw3) get a nominal
    # (step 1000, pElo 450) floor anchor so the curve reads as "started at random and
    # climbed". Skipped when the run already has pElo at/below step 1000 (adding it
    # there would create a backward dip).
    pelo_cums = [c for c, r in rows.items() if r["pElo"] not in ("", None)]
    if pelo_cums and min(pelo_cums) > BASELINE_STEP:
        put_pelo(BASELINE_STEP, BASELINE_STEP, BASELINE_PELO, "", "baseline (random-init anchor)")

    # Sample every ~1000 cum-steps to match the replay tracker's cadence. The raw
    # sources are far denser — [STATS] telemetry is ~per-60s and the in-training
    # probe pElo curve is ~per-25-steps — which is what bloated the embedded
    # dashboard. Bucket rows by nearest 1000-step boundary and emit one row per
    # bucket.
    #
    # Within a bucket the two sources are complementary, not competing: a [STATS]
    # row carries loss/gNorm/pIllM/... and the real elapsed clock, a probe row
    # carries pElo/nll at an interpolated elapsed. The bucket's row is therefore the
    # [STATS] row nearest the boundary, with the nearest probe row's pElo/nll copied
    # onto it when the [STATS] row has none of its own. Simply picking one row per
    # bucket (the earlier behaviour, which preferred the pElo-bearing row) meant a
    # probe firing more often than once per 1000 steps evicted essentially every
    # [STATS] row, leaving the loss charts empty for exactly the runs with the best
    # pElo curves. Buckets with no [STATS] row keep the probe row as before.
    #
    # A [STATS]-derived row is recognised by its launch index (`segment`), which the
    # probe-only and anchor rows leave blank — not by whether `loss` parsed, because
    # a truncated [STATS] line yields a row with a timestamp and little else, and
    # such a row must still be able to stand in for a bucket that has nothing
    # better. Within a bucket a loss-bearing row beats a loss-less one outright;
    # distance to the boundary only breaks ties between equals.
    def _nearer(a, b, target):
        return a if abs(a - target) <= abs(b - target) else b

    def _better_stats(a, b, target):
        a_has = rows[a]["loss"] not in ("", None)
        b_has = rows[b]["loss"] not in ("", None)
        if a_has != b_has:
            return a if a_has else b
        return _nearer(a, b, target)

    stats_pick = {}   # bucket index -> cum_step of the chosen [STATS] row
    pelo_pick = {}    # bucket index -> cum_step of the nearest pElo-bearing row
    for s, r in rows.items():
        b = round(s / 1000.0)
        if r["segment"] != "":
            stats_pick[b] = s if b not in stats_pick else _better_stats(stats_pick[b], s, b * 1000)
        if r["pElo"] not in ("", None):
            pelo_pick[b] = s if b not in pelo_pick else _nearer(pelo_pick[b], s, b * 1000)
    out = []
    for b in sorted(set(stats_pick) | set(pelo_pick)):
        if b not in stats_pick:
            out.append(rows[pelo_pick[b]])
            continue
        row = dict(rows[stats_pick[b]])
        if row["pElo"] in ("", None) and b in pelo_pick:
            src = rows[pelo_pick[b]]
            row["pElo"] = src["pElo"]
            row["nll"] = src["nll"]
            if src["note"] and not row["note"]:
                row["note"] = src["note"]
        out.append(row)
    out.sort(key=lambda r: r["cum_step"])
    p = os.path.join(DATA, f"{key}.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out, resets, foreign


def main():
    reg = json.load(open(os.path.join(HERE, "selfplay_registry.json")))
    os.makedirs(DATA, exist_ok=True)
    print(f"{'run':6} {'rows':>5} {'step0':>7} {'stepN':>8} {'hrs':>6} {'pElo':>6} {'resets':>6} {'foreign':>7}")
    for key, cfg in reg["runs"].items():
        out, resets, foreign = build_run(key, cfg)
        if not out:
            print(f"{key:6}  (no data)")
            continue
        s0, sN = out[0]["cum_step"], out[-1]["cum_step"]
        els = [r["elapsed_train_sec"] for r in out if r["elapsed_train_sec"] != ""]
        hrs = (max(els) / 3600.0) if els else 0.0
        tag = ("  reset-reconstructed" if resets else "") + (f"  ({foreign} foreign lines dropped)" if foreign else "")
        print(f"{key:6} {len(out):>5} {s0:>7} {sN:>8} {hrs:>6.1f} {str(cfg.get('endpoint_pElo')):>6} {resets:>6} {foreign:>7}{tag}")


if __name__ == "__main__":
    main()
