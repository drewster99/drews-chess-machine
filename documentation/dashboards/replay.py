#!/usr/bin/env python3
"""
Standardized tracker / renderer for DCM corpus-replay runs.

NOTE: `render` produces the DEPRECATED static-PNG dashboard
(dcm_dashboard_DEPRECATED.html). The current dashboard is the interactive
master built by master.py -> dcm_master.html (repo root). The tracking/CSV
side of this module (data/<run>.csv, registry.json) is still the live source
of truth that master.py reads; only the HTML render here is superseded.

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
import os, re, sys, csv, json, glob, struct, math, bisect, argparse, collections, itertools, datetime
import numpy as np
from _schema import FIELDS  # single source of the CSV column order (shared with selfplay.py)

HERE = os.path.dirname(os.path.abspath(__file__))
# Config/data/output root. Defaults to the script dir, but can be pointed at a
# separate chart set (its own registry.json + data/ + dcm_dashboard.html) via
# DCM_DASH_ROOT — this is how the "elite"-corpus set lives alongside the main one
# without forking the script. The script itself (and shared assets like the v5 doc)
# still resolve against HERE.
ROOT = os.path.abspath(os.environ.get("DCM_DASH_ROOT", HERE))
REG = json.load(open(os.path.join(ROOT, "registry.json")))
MODELS = os.path.expanduser(REG["models_dir"])
LOGS = os.path.expanduser(REG["logs_dir"])
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
def _resolve_bin():
    """Locate this machine's Release DrewsChessMachine binary. The DerivedData
    hash is per-machine and changes on migration, so we GLOB for it rather than
    hardcode a path that only exists on the Mac that first wrote this file.
    Override with the DCM_BIN env var. Returns None if no build is present — only
    the probing commands (track / probe-backfill) need it; render/recompute/migrate
    work without a binary, so a missing build must not break them at import."""
    env = os.environ.get("DCM_BIN")
    if env:
        return env
    pat = os.path.expanduser(
        "~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/"
        "Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine")
    hits = [p for p in glob.glob(pat) if os.access(p, os.X_OK)]
    hits.sort(key=os.path.getmtime, reverse=True)   # newest build wins if several
    return hits[0] if hits else None

BIN = _resolve_bin()

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
    if not BIN:
        raise RuntimeError(
            "no DrewsChessMachine Release binary found under DerivedData "
            "(build the app in Xcode, or set DCM_BIN). Probing requires it.")
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
# `loss=` (combined total) is optional so older log formats without it still match.
_METRICS = re.compile(r"(?:loss=([\d.]+) )?pLoss=([\d.]+) vLoss=([\d.]+).*?pIllM=([\d.]+).*?gNorm=([\d.]+).*?ms=([\d.]+)")
_STEP = re.compile(r"\[REPLAY\] step=(\d+)\b")
_GAMES = re.compile(r"\bgames=(\d+)\b")
# The closing summary, e.g.
#   [REPLAY] done: steps=268506 positionsFed=2292646157 gamesFed=34649108 epochs=2
# It is the only EXACT games total for a segment: per-step lines are emitted every
# ~50 steps, so a run that stops between them leaves its last `games=` reading short
# of the truth (34,648,331 vs 34,649,108 for run 1). Segments' final checkpoints are
# routinely saved on that same off-cadence step, so without this the last row of each
# segment would silently under-report the compute axis.
_DONE = re.compile(r"\[REPLAY\] done: steps=(\d+)\b.*?\bgamesFed=(\d+)\b")


class LogIndex:
    """One parse of one session log, answering every question we ask of it.

    These logs are big: the v5 lineage's five run logs total 6.4 GB, the largest
    single file 3.0 GB. The earlier shape of this code re-read a log from the top
    on EVERY checkpoint lookup — fine for the handful of marks `track` takes live,
    ruinous in bulk, since importing v5's 761 probes that way would have re-read
    roughly a terabyte. Parsing once into memory and serving lookups from it turns
    that into a single streaming pass per log.

    Everything is keyed by the SEGMENT-LOCAL step (each resumed segment restarts
    its counter at 1; the cumulative axis is applied by the caller):

      order    ordered [(step, abs_sec)] over timestamped lines, midnight rollover
               resolved — the timeline `_clamped_timeline` walks.
      at_time  {step: abs_sec} for those same lines.
      metrics  {step: {loss,pLoss,vLoss,pIllM,gNorm,ms}} for lines carrying them;
               a later line for the same step wins, matching the old forward scan.
      games    {step: cumulative games fed} — raw material for the compute axis.
      m_steps / g_steps  sorted keys, for nearest-<= lookup.

    One deliberate behavior note: the old scan stopped at the first line whose step
    exceeded the target, so on a log holding two concatenated runs it would never
    read past the seam. This class indexes the whole file, so a step appearing twice
    resolves to the LAST occurrence. Session logs are one-run-per-file and strictly
    monotonic, so the two agree; the only concatenated artifact in this project
    (`run.out`) is not a session log and is never passed here.
    """
    __slots__ = ("order", "at_time", "metrics", "games", "m_steps", "g_steps")

    def __init__(self, path):
        self.order, self.at_time, self.metrics, self.games = [], {}, {}, {}
        day, prev = 0, None
        if os.path.exists(path):
            for line in open(path, errors="replace"):
                dm = _DONE.search(line)
                if dm:
                    self.games[int(dm.group(1))] = int(dm.group(2))
                    continue
                sm = _STEP.search(line)
                if not sm:
                    continue
                step = int(sm.group(1))
                t = _TS.match(line)
                if t:
                    h, mi, s, frac = int(t[1]), int(t[2]), int(t[3]), t[4]
                    sod = h * 3600 + mi * 60 + s + int(frac) / (10 ** len(frac))
                    if prev is not None and sod < prev - 1:
                        day += 1                       # crossed midnight
                    prev = sod
                    absolute = day * 86400 + sod
                    self.at_time[step] = absolute
                    self.order.append((step, absolute))
                mt = _METRICS.search(line)
                if mt:
                    self.metrics[step] = dict(loss=float(mt[1]) if mt[1] else "",
                                              pLoss=float(mt[2]), vLoss=float(mt[3]),
                                              pIllM=float(mt[4]), gNorm=float(mt[5]),
                                              ms=float(mt[6]))
                g = _GAMES.search(line)
                if g:
                    self.games[step] = int(g.group(1))
        self.m_steps = sorted(self.metrics)
        self.g_steps = sorted(self.games)


_INDEX_CACHE = {}

def log_index(logname):
    """Memoized LogIndex for a log under LOGS. A missing file yields an empty index
    rather than raising, so every caller degrades to blanks the same way."""
    if logname not in _INDEX_CACHE:
        _INDEX_CACHE[logname] = LogIndex(os.path.join(LOGS, logname))
    return _INDEX_CACHE[logname]

def _nearest_at_or_below(sorted_steps, step):
    i = bisect.bisect_right(sorted_steps, step)
    return sorted_steps[i - 1] if i else None

def _log_index(logname):
    """Back-compat shim returning (at_time, order) as the old helper did."""
    li = log_index(logname)
    return li.at_time, li.order

def _metrics_at(logname, meta_step, stale_tol=1500):
    """pLoss/vLoss/pIllM/gNorm/ms from the [REPLAY] step=<meta> line (nearest <=).

    STALENESS GUARD: if the nearest matching line is more than `stale_tol` steps behind
    the requested meta_step, return {} (blank) instead of that line's values. A gap that
    large means the log went silent across this checkpoint — logging lost during a
    disk-full / sleep / stall window — so this checkpoint's REAL training stats were
    never written. Repeating the last-known line for every checkpoint in such a window is
    exactly what produced the flat, identical-valued "notch" artifacts (e.g. nt8y's
    pLoss/vLoss step during the 2026-07-06 disk-full). Blank is honest: no data, so the
    charts skip it rather than drawing a stale plateau. Normal logging is every ~60 s
    (~170 steps here), so 1500 only ever trips on a genuine multi-interval gap."""
    li = log_index(logname)
    st = _nearest_at_or_below(li.m_steps, meta_step)
    if st is None or meta_step - st > stale_tol:
        return {}                                    # log gap -> no real stats for this checkpoint
    return li.metrics[st]

def _games_at(logname, meta_step, stale_tol=1500):
    """Cumulative games fed at (nearest <=) meta_step — this segment's compute reading.

    Carries the same staleness guard as `_metrics_at`, and for a sharper reason: a
    stale `games=` does not merely repeat a value, it UNDERSTATES the compute axis by
    however much the corpus advanced while the log was silent, and an understated
    compute axis is worse than an absent one because it still looks plottable."""
    li = log_index(logname)
    st = _nearest_at_or_below(li.g_steps, meta_step)
    if st is None or meta_step - st > stale_tol:
        return None
    return li.games[st]

def _clamped_timeline(order, grace_sec=120.0):
    """Given ordered [(step, abs_sec)] log lines, return
    ({step: cumulative TRAINING seconds}, clamped_total,
     {step: cumulative RAW WALL seconds}, raw_total).

    The raw pair is the same walk with the cap removed. It exists so the clamp is
    auditable instead of invisible: `wall_sec - elapsed_train_sec` is exactly how
    much this function decided was not training. Across the v5 lineage that is 0.00 h
    on four of five segments and 20.54 h (5.7%) on run 2's 15-day span — plausible
    machine sleep, but a number you should be able to SEE rather than infer.

    Each inter-line interval contributes
    its real wall gap, HARD-CAPPED at the training time its step delta can justify
    (Δsteps x median-sec-per-step) plus a small logging-jitter grace — so an interval
    can NEVER bank more seconds than its own step count earns. Machine sleep / app-nap /
    thermal stall / disk stall is excluded at the source, whether it lands in a tiny
    interval (10-hour sleep across 50 steps -> ~33 s, not 10 h) OR is amortized across a
    huge log gap.

    The amortized case is why this replaced an earlier `dt > cap_factor*med*ds` test:
    that threshold scaled with Δsteps, so a gap merging tens of thousands of steps (log
    goes silent during a disk-full/sleep window, then resumes) got an enormous threshold
    and ~1.6 h of real sleep slid under it — producing a flat "notch" in the by-time
    charts. The step-earned cap does not scale away: for a 59k-step gap the cap is the
    step-earned ~4.2 h, and the extra ~1.6 h of wall time is dropped as idle.

    grace_sec absorbs normal logging jitter on SMALL intervals (where med*ds is tiny);
    on large intervals med*ds dominates and the grace is negligible, so the cap stays
    tight. Runs with no stalls are essentially unaffected (every gap ~ its step-earned
    time), so this reduces to plain wall-clock training time."""
    if not order:
        return {}, 0.0, {}, 0.0
    sps = [ (t1 - t0) / (s1 - s0)
            for (s0, t0), (s1, t1) in zip(order, order[1:])
            if s1 - s0 > 0 and t1 - t0 >= 0 ]
    med = float(np.median(sps)) if sps else 0.0
    train = {order[0][0]: 0.0}
    wall = {order[0][0]: 0.0}
    cum = raw = 0.0
    for (s0, t0), (s1, t1) in zip(order, order[1:]):
        ds, dt = s1 - s0, t1 - t0
        if ds <= 0:
            dur = raw_dur = 0.0              # same-step re-log / out-of-order: no new work
        else:
            raw_dur = max(dt, 0.0)
            dur = raw_dur
            if med > 0:
                dur = min(dur, med * ds + grace_sec)   # never bank more than step-earned (+jitter)
        cum += dur
        raw += raw_dur
        train[s1] = cum
        wall[s1] = raw
    return train, cum, wall, raw


class SegTime:
    """Pause-aware, sleep-clamped elapsed-training-time across a run's segments."""
    def __init__(self, segments, run=None):
        self.segs = segments
        self.run = run                      # enables CSV fallback when a prior log is gone
        self._warned = set()
        self.idx, self.first, self.train, self.dur = [], [], [], []
        self.wall, self.wall_dur, self.wall_keys = [], [], []
        prior = raw_prior = 0.0
        for sg in segments:
            idx, order = _log_index(sg["log"])
            first = order[0][1] if order else 0.0
            # sleep-immune cumulative train-secs, plus the same walk uncapped
            train, total, wall, raw_total = _clamped_timeline(order)
            self.idx.append(idx); self.first.append(first); self.train.append(train)
            self.wall.append(wall)
            self.wall_keys.append(sorted(wall))   # sorted once, not per lookup
            self.dur.append(prior)          # cumulative TRAINING seconds before this segment
            self.wall_dur.append(raw_prior) # cumulative RAW WALL seconds before this segment
            prior += total
            raw_prior += raw_total

    def _base(self, si):
        """Seconds of elapsed training before segment si begins. Resolution order:
        (1) explicit elapsed_base_sec in the registry (precise, pinned);
        (2) log-summed prior durations (dur[si]) when the prior logs exist;
        (3) durable CSV fallback — the max elapsed already recorded for any earlier
            cumstep — used when a prior segment's log was deleted (dur[si] would be a
            wrong 0 and the by-time axis would restart at 0). Warns once so the silent
            restart-to-0 that bit us on nt8y can never recur unnoticed."""
        sg = self.segs[si]
        if "elapsed_base_sec" in sg:
            return sg["elapsed_base_sec"]
        if si == 0 or self.dur[si] > 0:      # seg 0, or prior logs present -> trust dur
            return self.dur[si]
        # prior-segment log(s) missing: recover the base from the durable CSV
        base = 0.0
        if self.run:
            for r in read_csv(self.run):
                try:
                    if int(r["cum_step"]) < sg["cumstep_base"] and r.get("elapsed_train_sec") not in ("", None):
                        base = max(base, float(r["elapsed_train_sec"]))
                except (ValueError, KeyError):
                    pass
        if si not in self._warned:
            self._warned.add(si)
            sys.stderr.write(f"[warn] SegTime {self.run}: segment {si} ('{sg.get('label','')}') has a "
                             f"missing prior-segment log; anchored elapsed base to CSV = {base/3600:.2f}h. "
                             f"Pin it by setting \"elapsed_base_sec\" on this segment in registry.json.\n")
        return base

    def _wall_base(self, si):
        """Raw wall seconds before segment si begins. Same resolution ladder as
        `_base`, with one extra rung on top: an explicit `wall_base_sec`. Where a
        segment pins only `elapsed_base_sec`, that pinned value is reused as the wall
        base too — for the v5 lineage the pinned prefix comes from a machine whose
        logs are not on this host, so there is no separate raw measurement to pin and
        pretending otherwise would invent one. The consequence is bounded and worth
        stating: `wall_sec - elapsed_train_sec` measures clamping only WITHIN the
        logged segments, never inside a pinned prefix."""
        sg = self.segs[si]
        if "wall_base_sec" in sg:
            return sg["wall_base_sec"]
        if "elapsed_base_sec" in sg:
            return sg["elapsed_base_sec"]
        if si == 0 or self.wall_dur[si] > 0:
            return self.wall_dur[si]
        base = 0.0
        if self.run:
            for r in read_csv(self.run):
                try:
                    if int(r["cum_step"]) < sg["cumstep_base"] and r.get("wall_sec") not in ("", None):
                        base = max(base, float(r["wall_sec"]))
                except (ValueError, KeyError):
                    pass
        return base

    def wall_at(self, cumstep, segment=None):
        """Raw (unclamped) cumulative wall-training seconds at cumstep, or "" when the
        segment's log is absent. Mirrors `elapsed_and_clock`'s nearest-<= lookup so the
        two columns are always read at the same point and stay directly comparable —
        including the `segment` override, which must be passed in tandem with it."""
        si = self.seg_for(cumstep) if segment is None else segment
        meta = cumstep - self.segs[si]["cumstep_base"]
        wall = self.wall[si]
        if not wall:
            return ""
        st = _nearest_at_or_below(self.wall_keys[si], meta)
        return round(self._wall_base(si) + (wall[st] if st is not None else 0.0), 1)

    def seg_for(self, cumstep):
        """Last segment whose cumstep_base <= cumstep.

        ⚠️ This is a GUESS, correct only while segments tile the axis end-to-end. A
        segment that resumes from PART-WAY THROUGH its predecessor breaks it: v5's
        segment 7 forked from segment 6 at step 46,000, so its base (857,769) lands
        inside segment 6's span (811,769–861,143), and segment 6's own final mark at
        cum 857,769 resolves here to 7 with meta=0. Callers that already KNOW the
        segment must pass it explicitly rather than rely on this."""
        best = 0
        for i, sg in enumerate(self.segs):
            if sg["cumstep_base"] <= cumstep:
                best = i
        return best

    def elapsed_and_clock(self, cumstep, segment=None):
        """`segment` overrides `seg_for`; pass it whenever the caller knows which
        segment a mark belongs to (see seg_for's warning about forked segments)."""
        si = self.seg_for(cumstep) if segment is None else segment
        meta = cumstep - self.segs[si]["cumstep_base"]
        idx = self.idx[si]
        if not idx:                       # segment log missing -> no time axis
            return "", "", meta, si
        # elapsed = sleep-clamped cumulative training seconds at (nearest <=) this step
        train = self.train[si]
        if meta in train:
            sec = train[meta]
        else:
            le = [k for k in train if k <= meta]
            sec = train[max(le)] if le else 0.0
        # wallclock_iso is a real clock reading (NOT a duration), so it uses the raw
        # timestamp — clamping only affects elapsed, never the wall-clock stamp.
        if meta in idx:
            abs_sec = idx[meta]
        else:  # nearest <= meta (e.g. abort save between log lines)
            le = [k for k in idx if k <= meta]
            abs_sec = idx[max(le)] if le else self.first[si]
        elapsed = self._base(si) + sec
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

# ---------- enumerated checkpoints (app-side --enumerate-checkpoints) ----------
# The corpus-replay runner, when launched with --enumerate-checkpoints, writes a
# step-numbered copy of the weights next to the rolling out-model on every save:
#   <stem>-replay-step<N>.safetensors   (N = the segment-local training step).
# That makes the tracker's old habit of minting its OWN cum-named "-frozen" copies
# redundant — the app already preserves every step. So for enumerated runs we probe
# the app's files in place and map their local step N to the run-cumulative axis via
# the latest segment's cumstep_base (cum = base + N). Legacy (pre-enumeration) runs
# still fall back to the "-frozen" snapshots, so mini2b/coxw/v5/… are unaffected.
def enum_glob(cfg):
    # `or ""` not `.get(k, "")`: legacy runs (v5, t97x) carry an explicit
    # "out_model": null, so .get returns None and the membership test raises.
    om = cfg.get("out_model") or ""
    return om.replace("-replay-latest", "-replay-step*") if "-replay-latest" in om else None

def enum_specs(cfg):
    """[(segment_index, glob)] naming each segment's enumerated checkpoints.

    A run's enumerated files are named from its OUT-MODEL stem, and every resumed
    segment normally gets its own stem (`…-Qeu8-replay-step*`, `…-Qeu8-resume2-replay-step*`,
    …). The run-level `out_model` names only the LATEST segment, so a single glob built
    from it finds that segment and is blind to every earlier one — nt8y hides 4 segments
    that way, mini2b and qeu8 3 each, ykkk 2, coxw 1. Their marks were never wrong, just
    absent.

    A segment may therefore declare its own `enum_stem`. Resolution per segment:
      1. explicit `enum_stem`  -> `<stem>-replay-step*.safetensors`
      2. otherwise, the LAST segment falls back to the run-level `enum_glob`

    With no `enum_stem` anywhere this returns exactly what the old single-glob code
    used, so runs that do not opt in are bit-for-bit unaffected."""
    segs = cfg.get("segments", [])
    run_glob = enum_glob(cfg)
    specs = []
    for si, sg in enumerate(segs):
        stem = sg.get("enum_stem")
        if stem:
            specs.append((si, f"{stem}-replay-step*.safetensors"))
        elif run_glob and si == len(segs) - 1:
            specs.append((si, run_glob))
    return specs

_RESUME_SUFFIX = re.compile(r"^(?:run|original|resume\d*)$")

def discover_enum_stems(cfg, run=None, verbose=True):
    """Propose an `enum_stem` for each segment by reading checkpoints on disk.

    Filling `enum_stem` by hand is error-prone and the cost of getting it wrong is
    silent (a segment's marks simply never fill). This derives it instead, and
    refuses rather than guesses when the evidence disagrees.

    Method, in order of trust:
      1. The family root comes from `out_model` minus `-replay-latest` and any
         trailing `-resume<N>`. Candidate files are globbed from that root.
      2. A candidate is accepted only if its stem is the root or the root plus a
         RESUME-SHAPED suffix that matches one of this run's segment labels. That
         is what keeps a sibling run out: `20260702-Qeu8e5-…` yields the suffix
         `e5`, which is not resume-shaped, so it is rejected rather than silently
         folded into Qeu8's marks.
      3. Every file under one stem must agree on `model_id` — the header, not the
         filename, is the authority. A stem spanning two model_ids means the
         files were overwritten in place (exactly what happened inside the v5
         bundle) and is refused.
      4. Stems must appear in the same order by earliest `created_at_unix` as
         their segments appear in the registry. A disagreement means the
         label→segment mapping is wrong, and is reported rather than written.

    Returns {segment_index: stem}. Segments with no surviving files are simply
    absent — that is not an error, it is the normal state for a run whose early
    checkpoints were cleaned up."""
    segs = cfg.get("segments", [])
    om = (cfg.get("out_model") or "")
    if "-replay-latest" not in om:
        return {}
    root = om.split("-replay-latest")[0]
    root = re.sub(r"-resume\d*$", "", root)

    # label -> segment index, using the leading token of each segment's label
    lab = {}
    for i, s in enumerate(segs):
        m = re.match(r"(run|original|resume\d*)", (s.get("label") or "").strip())
        if m:
            lab.setdefault(m.group(1), i)

    by_stem = collections.defaultdict(list)
    for p in glob.glob(os.path.join(MODELS, root + "*-replay-step*.safetensors")):
        stem = os.path.basename(p).split("-replay-step")[0]
        suffix = stem[len(root):].lstrip("-") or "run"
        if not _RESUME_SUFFIX.match(suffix):
            continue                                  # sibling run, not this one
        by_stem[stem].append(p)

    # Which segment a stem belongs to is decided by EVIDENCE, not by its name.
    # The obvious rule -- unsuffixed stem means the first segment -- is wrong:
    # qeu8's segment 0 predates --enumerate-checkpoints (its marks are the legacy
    # cum-named -frozen files), so the first segment to write enumerated files was
    # segment 1, and it wrote them under the UNSUFFIXED stem. Trusting the name
    # there maps 68 files onto segment 0's base and invents 68 rows at cum steps
    # that never existed. So: score every segment by how many of the stem's files
    # land on a cum_step the CSV already knows, and take the unambiguous winner.
    csv_cums = {int(r["cum_step"]) for r in read_csv(run)} if run else set()
    out, refused, order, pool = {}, [], [], []
    for stem, paths in sorted(by_stem.items()):
        ids, times, steps = set(), [], []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    m = json.loads(f.read(n)).get("__metadata__", {})
                ids.add(m.get("model_id")); steps.append(int(m["training_step"]))
                if m.get("created_at_unix"):
                    times.append(int(m["created_at_unix"]))
            except (OSError, ValueError, KeyError, struct.error):
                pass
        if len(ids) > 1:
            refused.append((stem, "spans %d model_ids %s" % (len(ids), sorted(ids))))
            continue
        if not steps:
            continue
        # Score by how many of the stem's files land on a cum_step the CSV already
        # knows. Deliberately NO hard cutoff on a segment's nominal length: a run
        # routinely writes checkpoints past the point its successor resumed from
        # (nt8y's resume3 reaches step 140,000 against a 139,000 span; v5's run 4
        # ran to 49,374 while run 5 forked at 46,000). Those files are real but
        # off-lineage, and excluding a segment because of them threw away the right
        # answer. A genuinely wrong segment shows up as a low hit-rate instead.
        # Two signals, because either alone is too weak. Hit-rate says "these files
        # explain marks this run actually has" -- but with marks every 1000 steps and
        # bases 1000 apart, several segments score 100% by coincidence. Span-closeness
        # says "this segment is about as long as this stem reaches", which is sharp,
        # but is NOT a hard cutoff: a run routinely writes checkpoints past the point
        # its successor resumed from (nt8y's resume3 hits step 140,000 against a
        # 139,000 span; v5's run 4 reached 49,374 while run 5 forked at 46,000). So
        # hit-rate gates, and span-closeness ranks.
        last_cum = max(csv_cums) if csv_cums else 0
        cand = {}
        for i, sg in enumerate(segs):
            base = sg["cumstep_base"]
            r = sum(1 for st in steps if base + st in csv_cums) / len(steps)
            if r >= 0.5:
                span = (segs[i + 1]["cumstep_base"] - base) if i + 1 < len(segs) else max(last_cum - base, 0)
                cand[i] = (r, -abs(span - max(steps)))
        if not cand:
            refused.append((stem, "no segment's marks account for these files"))
            continue
        pool.append((min(times) if times else 0, stem, len(paths), cand))

    # Segments run one after another, so stems ordered by creation time must map to
    # segments in STRICTLY INCREASING order. Hit-rate alone is too weak on its own --
    # a two-file stem lands on known marks under several bases by coincidence. Taking
    # the best monotonic assignment over all stems resolves those together, and is
    # what distinguishes v5's seg 6 from seg 7 and nt8y's resume2/3/4.
    pool.sort()
    best, ties = None, 0
    for combo in itertools.combinations(range(len(segs)), len(pool)):
        if any(si not in c[3] for si, c in zip(combo, pool)):
            continue
        tot = (sum(c[3][si][0] for si, c in zip(combo, pool)),
               sum(c[3][si][1] for si, c in zip(combo, pool)))
        if best is None or tot > best[0]:
            best, ties = (tot, combo), 1
        elif tot == best[0]:
            ties += 1
    if best is None:
        refused.append((None, "no monotonic assignment of %d stem(s) onto %d segments"
                        % (len(pool), len(segs))))
    elif ties > 1:
        refused.append((None, "%d monotonic assignments score equally -- refusing to guess" % ties))
    else:
        for si, (t, stem, n, cand) in zip(best[1], pool):
            out[si] = stem
            order.append((t, si, stem, n, cand[si][0]))
        order.sort()
    if verbose:
        for _, si, stem, n, sc in order:
            print("  seg %d  %-46s %4d files  %.0f%% land on known marks" % (si, stem, n, 100 * sc))
        for stem, why in refused:
            print("  REFUSED %s: %s" % (stem or "(run)", why))
    return out


def enum_path(cfg, n):
    om = cfg.get("out_model", "")
    return (os.path.join(MODELS, om.replace("-replay-latest", f"-replay-step{n}"))
            if "-replay-latest" in om else None)

# ---------- track ----------
def freeze(run, cfg, meta):
    """Return (cum, checkpoint_path) for this mark. Prefer the app's enumerated
    checkpoint for the exact step (no copy — it's already preserved on disk);
    otherwise fall back to copying the rolling out-model into a cum-named -frozen
    snapshot (legacy runs, or a mark taken between enumerated saves)."""
    base = cfg["segments"][-1]["cumstep_base"]
    cum = base + meta
    ep = enum_path(cfg, meta)
    if ep and os.path.exists(ep):
        return cum, ep
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
    st = SegTime(cfg["segments"], run); elapsed, clock, m2, si = st.elapsed_and_clock(cum)
    met = _metrics_at(cfg["segments"][si]["log"], meta)
    lm = round(1 - met["pIllM"], 4) if "pIllM" in met else ""
    row = dict(cum_step=cum, meta_step=meta, segment=si, elapsed_train_sec=elapsed,
               wallclock_iso=clock, ms_per_step=met.get("ms", ""),
               pElo=round(pr.get("pElo"), 2) if pr.get("pElo") else "",
               nll=round(pr.get("nll"), 4) if pr.get("nll") else "",
               loss=met.get("loss", ""), pLoss=met.get("pLoss", ""), vLoss=met.get("vLoss", ""),
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


def _backfill_one(cfg, st, rows, by, cum, path, name, verbose, segment=None):
    """Probe `path` and fill/create the CSV row for `cum` if it lacks pElo.
    Returns 1 if a row was filled/created, else 0. Shared by both the enumerated
    -replay-step scan and the legacy -frozen scan so they stay bit-identical.

    `segment` is passed by the enumerated scan, which already knows which segment a
    file came from; without it `seg_for` has to guess from cum and mis-attributes any
    mark sitting exactly on a segment boundary."""
    r = by.get(cum)
    if r and r.get("pElo") not in ("", None):
        return 0                                     # already has pElo
    pr = probe(path)
    if not pr.get("pElo"):
        return 0
    bn1, sae2, effs = internals(path, cfg["rezero_cap"])
    elapsed, clock, meta, si = st.elapsed_and_clock(cum, segment)
    met = _metrics_at(cfg["segments"][si]["log"], meta)
    pf = dict(pElo=round(pr["pElo"], 2),
              nll=round(pr.get("nll"), 4) if pr.get("nll") else "",
              bn1Mean=round(bn1, 4), sae2=round(sae2, 4),
              eff_alpha=";".join(f"{e:.4f}" for e in effs),
              pLogit_mean=round(pr.get("pLogit_mean"), 3) if pr.get("pLogit_mean") else "",
              pLogit_peak=pr.get("pLogit_peak", ""), frozen_file=name)
    if r:
        r.update(pf)
        if r.get("elapsed_train_sec") in ("", None):
            r["elapsed_train_sec"] = elapsed
    else:
        lm = round(1 - met["pIllM"], 4) if "pIllM" in met else ""
        r = dict(cum_step=cum, meta_step=meta, segment=si, elapsed_train_sec=elapsed,
                 wallclock_iso=clock, ms_per_step=met.get("ms", ""),
                 loss=met.get("loss", ""), pLoss=met.get("pLoss", ""), vLoss=met.get("vLoss", ""),
                 legalMass=lm, pIllM=met.get("pIllM", ""), gNorm=met.get("gNorm", ""),
                 note="probe-backfill", **pf)
        rows.append(r); by[cum] = r
    if verbose:
        print(f"  probe-backfill cum {cum}: pElo {pr['pElo']:.0f}")
    return 1


def probe_backfill(run, verbose=True):
    """Recover pElo/nll/internals for any preserved checkpoint whose CSV row lacks
    pElo — marks that were only log-backfilled, or went by while probing lagged.
    Idempotent: skips rows already probed. As long as a checkpoint exists on disk
    the pElo comes back, so a monitoring gap self-heals instead of losing history.

    Two checkpoint sources:
      • enumerated  <stem>-replay-step<N>.safetensors  — app-side (--enumerate-
        checkpoints), local step N, cum = latest-segment base + N;
      • legacy      <...>-step<cum>-frozen.safetensors  — cum-named tracker snapshots
        (pre-enumeration runs, and this run's earlier segments)."""
    cfg = REG["runs"][run]
    rows = read_csv(run)
    by = {int(r["cum_step"]): r for r in rows}
    st = SegTime(cfg["segments"], run)
    filled = 0

    # (a) enumerated app-side checkpoints. One glob PER SEGMENT: the step in an
    # enumerated filename is segment-local, so it only becomes a cumulative step
    # against its own segment's base. Mapping every match onto one base would be
    # wrong the moment two segments' files sit in the same directory.
    for si, eg in enum_specs(cfg):
        eprefix, esuffix = eg.split("*")
        ebase = cfg["segments"][si]["cumstep_base"]
        for f in sorted(glob.glob(os.path.join(MODELS, eg))):
            name = os.path.basename(f)
            try:
                n = int(name[len(eprefix):len(name) - len(esuffix)])
            except ValueError:
                continue
            filled += _backfill_one(cfg, st, rows, by, ebase + n, f, name, verbose, segment=si)

    # (b) legacy cum-named -frozen snapshots
    prefix, suffix = cfg["frozen_glob"].split("*")   # "...-step" , "-frozen.safetensors"
    for f in sorted(glob.glob(os.path.join(MODELS, cfg["frozen_glob"]))):
        name = os.path.basename(f)
        try:
            cum = int(name[len(prefix):len(name) - len(suffix)])
        except ValueError:
            continue
        filled += _backfill_one(cfg, st, rows, by, cum, f, name, verbose)

    if filled:
        rows.sort(key=lambda r: int(r["cum_step"]))
        write_csv(run, rows)
    if verbose:
        print(f"probe-backfilled {filled}")
    return filled

def recompute_elapsed(run, verbose=True):
    """Rewrite elapsed_train_sec for every existing row using the current SegTime
    (sleep-clamped). One-time migration so historical marks match newly-tracked ones
    after the clamp lands — otherwise old rows keep their wall-clock (sleep-inflated)
    elapsed while new rows are clamped, and the by-time axis mixes the two."""
    cfg = REG["runs"][run]
    rows = read_csv(run)
    if not rows:
        if verbose:
            print(f"{run}: no rows")
        return 0
    st = SegTime(cfg["segments"], run)
    changed = 0
    for r in rows:
        cum = int(r["cum_step"])
        # Trust the row's recorded segment over re-deriving it from cum, which
        # `seg_for` cannot do correctly across a forked segment.
        seg = int(r["segment"]) if str(r.get("segment", "")).isdigit() else None
        el, _clk, _meta, _si = st.elapsed_and_clock(cum, seg)
        if el != "" and str(r.get("elapsed_train_sec", "")) != str(el):
            r["elapsed_train_sec"] = el
            changed += 1
        # wall_sec must move WITH elapsed_train_sec. They are the clamped and unclamped
        # readings of one walk, and `wall_sec - elapsed_train_sec` is only meaningful
        # while both are computed from the same timeline — recomputing one alone would
        # leave a stale difference that still looks like a clamp measurement.
        wl = st.wall_at(cum, seg)
        if wl != "" and str(r.get("wall_sec", "")) != str(wl):
            r["wall_sec"] = wl
            changed += 1
    if changed:
        write_csv(run, rows)
    if verbose:
        print(f"{run}: recomputed elapsed on {changed} row(s)")
    return changed


# ---------- migrate from legacy loop_state.txt ----------
LOOP_STATE = os.path.join(HERE, "loop_state.txt")  # symlinked/copied from scratchpad if present

def _kv(line, key):
    m = re.search(rf"{re.escape(key)}=([0-9.]+)", line); return m.group(1) if m else ""

V5_DOC = os.path.join(HERE, "..", "v5-layernorm-output.md")

def migrate_v5():
    """v5's FIRST THREE segments live in the markdown table (per-subrun step), not
    loop_state. cum_step = subrun_step + offset. Elapsed left blank unless segment
    logs are present.

    ⚠️ Those three segments are no longer the whole run. v5 continued for five more
    segments whose probes were imported from JSONL (see `import_probes` and
    documentation/v5-lineage.md), and the markdown doc knows nothing about them. So
    this rebuilds ONLY the doc-sourced rows and carries every row from a later segment
    through untouched — otherwise re-running a migration would silently delete 761
    imported rows that cannot be re-probed, since most of their weight files no longer
    exist."""
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
    doc_segs = set(seg_for_name.values())
    existing = read_csv("v5")
    kept = [r for r in existing
            if str(r.get("segment", "")).isdigit() and int(r["segment"]) not in doc_segs]

    # Never let a rebuild REPLACE a real reading with a blank one. These three segments
    # ran on a different machine and their session logs are not on this host, so SegTime
    # above resolves their elapsed/wall to "" — while the CSV still carries the values
    # measured back when the logs were present. Blank is honest for something never
    # measured; it is data loss for something measured elsewhere.
    prior = {int(r["cum_step"]): r for r in existing if str(r.get("cum_step", "")).isdigit()}
    restored = 0
    for r in rows:
        old = prior.get(int(r["cum_step"]))
        if not old:
            continue
        for col in ("elapsed_train_sec", "wallclock_iso", "wall_sec", "games_fed"):
            if r.get(col) in ("", None) and old.get(col) not in ("", None):
                r[col] = old[col]
                restored += 1

    write_csv("v5", rows + kept)
    print(f"migrate v5: {len(rows)} doc rows rebuilt ({restored} field(s) preserved from the "
          f"existing CSV), {len(kept)} imported rows preserved -> {csv_path('v5')}")

# ---------- import probes recorded outside the tracker ----------
def _ckpt_index(dirs):
    """Map (model_id, training_step) -> path over every .safetensors in `dirs`.

    Keyed by METADATA, never by filename. The corpus-replay runner numbers its
    enumerated checkpoints per segment, restarting at 1 on every resume, so across
    the v5 lineage five segments competed for the same names and later runs
    overwrote earlier ones in place — four distinct files have been called
    `v5-cont-replay-step1000.safetensors`. Only the header's `model_id` (minted per
    segment) plus `training_step` names a checkpoint uniquely."""
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(os.path.expanduser(d), "*.safetensors"))):
            try:
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    m = json.loads(f.read(n)).get("__metadata__", {})
                mid, ts = m.get("model_id"), m.get("training_step")
                if mid and ts is not None:
                    out.setdefault((mid, int(ts)), p)
            except (OSError, ValueError, KeyError, struct.error):
                continue                     # unreadable file: absent, not fatal
    return out


def import_probes(run, jsonl, segment, ckpt_dirs=(), verbose=True):
    """Fold probe results recorded OUTSIDE this tracker into a run's CSV.

    The v5 lineage was monitored for five continuation segments by a separate
    bundle-side loop that appended one probe JSON per line to `new_ckpts*.jsonl`.
    Those probes are the only surviving record of most of those checkpoints — the
    weight files they describe were largely overwritten by later segments — so they
    are imported rather than re-derived. `pElo`/`nll` come straight from the JSON;
    everything else is joined from the segment's session log at the same step.

    Integrity gate: every probe's `modelID` must equal the segment's registry
    `model_id`. A monitor that trusted filenames alone once re-probed nine
    month-old files and published a fabricated curve from them; requiring the
    minted-per-segment ID makes that class of error impossible to import silently.
    Mismatches are reported and skipped, never written.

    Idempotent on cum_step, same as `track`."""
    cfg = REG["runs"][run]
    sg = cfg["segments"][segment]
    base = sg["cumstep_base"]
    want_id = sg.get("model_id")
    games_base = sg.get("games_base")
    st = SegTime(cfg["segments"], run)
    ck = _ckpt_index(ckpt_dirs) if ckpt_dirs else {}

    rows = read_csv(run)
    by = {int(r["cum_step"]): r for r in rows}
    added = skipped = rejected = 0

    for line in open(os.path.expanduser(jsonl)):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = re.search(r"step(\d+)", os.path.basename(d.get("model", "")))
        if not m:
            continue
        meta = int(m.group(1))
        # Provenance gate. Three outcomes, deliberately distinct:
        #   modelID matches            -> measured, import clean
        #   modelID absent + recovered -> the probe itself declares it was reconstructed
        #                                 (weights overwritten before archiving, values
        #                                 read back off a published chart). Import, but
        #                                 stamp the row so no reader mistakes it for a
        #                                 measurement.
        #   anything else              -> unverifiable or wrong segment; refuse.
        recovered = bool(d.get("recovered"))
        got_id = d.get("modelID")
        if want_id and got_id != want_id and not (got_id is None and recovered):
            rejected += 1
            if verbose:
                print(f"  REJECT step{meta}: modelID {got_id} != segment {segment} {want_id}")
            continue
        cum = base + meta
        if cum in by:
            skipped += 1
            continue

        met = _metrics_at(sg["log"], meta)
        # Pass `segment` explicitly: a forked segment's base can fall inside its
        # predecessor's span, so cum alone would misattribute this segment's own final
        # mark (v5 seg 6's last probe sits exactly on seg 7's base).
        elapsed, clock, _meta, si = st.elapsed_and_clock(cum, segment)
        g = _games_at(sg["log"], meta)
        lm = round(1 - met["pIllM"], 4) if "pIllM" in met else ""
        row = dict(cum_step=cum, meta_step=meta, segment=segment,
                   elapsed_train_sec=elapsed, wall_sec=st.wall_at(cum, segment),
                   wallclock_iso=clock, ms_per_step=met.get("ms", ""),
                   pElo=round(d["pElo"], 2) if d.get("pElo") else "",
                   nll=round(d["nll"], 4) if d.get("nll") else "",
                   loss=met.get("loss", ""), pLoss=met.get("pLoss", ""),
                   vLoss=met.get("vLoss", ""), legalMass=lm, pIllM=met.get("pIllM", ""),
                   gNorm=met.get("gNorm", ""),
                   pLogit_mean=d.get("policy_logit_abs_max", ""),
                   pLogit_peak=d.get("policy_logit_abs_max_peak", ""),
                   games_fed=(games_base + g) if (games_base is not None and g is not None) else "",
                   frozen_file=os.path.basename(d.get("model", "")),
                   note=("recovered:" + d.get("recovered_note", "reconstructed, not measured")
                         if recovered else f"import:{os.path.basename(jsonl)}"))

        # bn1Mean / sae2 / eff_alpha need the actual weights. Most of these
        # checkpoints no longer exist; fill them where the file survives and leave
        # them blank where it does not, rather than carrying a neighbour's value.
        p = ck.get((d.get("modelID"), meta))
        if p:
            try:
                bn1, sae2, effs = internals(p, cfg["rezero_cap"])
                row.update(bn1Mean=round(bn1, 4), sae2=round(sae2, 4),
                           eff_alpha=";".join(f"{e:.4f}" for e in effs))
            except (OSError, KeyError, ValueError):
                pass

        rows.append(row); by[cum] = row; added += 1

    if added:
        rows.sort(key=lambda r: int(r["cum_step"]))
        write_csv(run, rows)
    if verbose:
        print(f"{run} seg {segment} ({sg.get('label','')}): +{added} rows, "
              f"{skipped} already present, {rejected} rejected -> {csv_path(run)}")
    return added


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
            loss=_kv(line, "loss") or met.get("loss", ""),
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
    metrics = [("pElo", "pElo"), ("nll", "nll"), ("loss", "loss (combined)"), ("pLoss", "pLoss"), ("vLoss", "vLoss"),
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
        # Collect each run's series first, then draw in LAYERED passes: all runs'
        # raw points as a single background layer, then all runs' EMA lines on top —
        # so no run's raw scatter ever buries another run's EMA line.
        series = []
        for run, rows in runs.items():
            xs, ys = [], []
            for r in rows:
                y = _f(r[key]); x = _f(r[xkey])
                if y is None or x is None:
                    continue
                xs.append(x / 3600 if xkey == "elapsed_train_sec" else x); ys.append(y)
            if xs:
                series.append((run, xs, ys))
        if overlay_ema:
            for run, xs, ys in series:        # pass 1: all raw points (background)
                ax.plot(xs, ys, "-o", ms=2.0, lw=0.8, color=COLOR[run], alpha=0.2)
            for run, xs, ys in series:        # pass 2: all EMA lines (foreground)
                ax.plot(xs, _ema(ys, overlay_ema), "-", lw=1.9, color=COLOR[run], label=LAB[run])
        elif ema:
            for run, xs, ys in series:
                ax.plot(xs, _ema(ys, ema), "-", lw=1.8, color=COLOR[run], label=LAB[run])
        else:
            for run, xs, ys in series:
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
        runcell = (f"<td><span style='display:inline-block;width:11px;height:11px;"
                   f"background:{cfg['color']};border-radius:2px;margin-right:6px;"
                   f"vertical-align:middle'></span>{run}</td>")
        stem = cfg.get("arch_stem", cfg.get("arch_summary", ""))
        blocks = cfg.get("arch_blocks", "")
        heads = cfg.get("arch_heads", "")
        acells = (f"<td class=arch>{stem}</td><td class=arch>{blocks}</td>"
                  f"<td class=arch>{heads}</td>")
        if not rows:
            srows += (f"<tr>{runcell}{acells}<td>{cfg['params']:,}</td>"
                      "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>")
            continue
        steps = max(int(r["cum_step"]) for r in rows)
        hrs = max((_f(r["elapsed_train_sec"]) or 0) for r in rows) / 3600
        srows += (f"<tr>{runcell}{acells}<td>{cfg['params']:,}</td>"
                  f"<td>{steps:,}</td><td>{hrs:.1f}</td>"
                  f"<td>{_last(rows,'pElo')}</td><td>{_last(rows,'nll')}</td>"
                  f"<td>{_last(rows,'pLogit_mean')}</td></tr>")
    summ = ("<h2>Summary — architecture &amp; final metrics</h2>"
            "<table class=summary><tr>" + "".join(f"<th>{c}</th>" for c in scols) +
            "</tr>" + srows + "</table>")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Click-to-fullscreen lightbox: click any chart to enlarge, click (or Esc) to close.
    # Plain string (real braces) so the f-string template below doesn't parse its JS.
    lightbox = ("<div id='lb'><img alt=''></div>"
                "<script>(function(){"
                "var lb=document.getElementById('lb'),g=lb.firstElementChild;"
                "document.querySelectorAll('figure img').forEach(function(im){"
                "im.addEventListener('click',function(){g.src=im.src;lb.classList.add('on');});});"
                "lb.addEventListener('click',function(){lb.classList.remove('on');g.src='';});"
                "document.addEventListener('keydown',function(e){if(e.key==='Escape'){lb.classList.remove('on');g.src='';}});"
                "})();</script>")
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>DCM replay runs</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;max-width:min(1900px,96vw);color:#1a1a1a}}
h1{{font-size:21px}}h2{{font-size:14px;color:#444;margin-top:26px}}img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
table{{border-collapse:collapse;font-size:11px;margin:8px 0}}td,th{{border:1px solid #ccc;padding:2px 7px;text-align:right}}
summary{{cursor:pointer;font-size:13px;margin-top:10px}}.cap{{color:#666;font-size:12px}}
.pair{{display:flex;gap:14px;align-items:flex-start;margin-top:6px}}.pair figure{{flex:1;width:50%;margin:0}}
.pair img{{width:100%}}figcaption{{font-size:11px;color:#666;text-align:center;margin-bottom:2px}}
table.summary td,table.summary th{{text-align:left;font-size:11.5px;vertical-align:top}}
table.summary td.arch{{max-width:230px;white-space:normal;color:#333}}
figure img{{cursor:zoom-in}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.93);z-index:9999;cursor:zoom-out;align-items:center;justify-content:center}}
#lb.on{{display:flex}}#lb img{{max-width:98vw;max-height:96vh;width:auto;border:0;border-radius:0}}</style></head><body>
<h1>DCM — replay-run dashboard</h1>
<p class=cap>Auto-generated {now} from per-run CSVs in <code>data/</code> via <code>replay.py render</code>.
Each metric: 10-point EMA (foreground) over raw marks at 30% opacity (background), shown vs cumulative SGD step and vs elapsed training time (pause-aware). The pElo panel has a collapsible early-window zoom. Reload to refresh.</p>
{summ}
{ch}<h2>Data tables</h2>{tb}{lightbox}</body></html>"""
    out = os.path.join(ROOT, "dcm_dashboard_DEPRECATED.html")
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
    p = sub.add_parser("recompute"); p.add_argument("run", nargs="?", default="__all__")
    p = sub.add_parser("discover-stems")
    p.add_argument("run", nargs="?", default="__all__")
    p.add_argument("--write", action="store_true",
                   help="write the proposed enum_stem values into registry.json "
                        "(default: propose only, change nothing)")
    p = sub.add_parser("import-probes")
    p.add_argument("run"); p.add_argument("--jsonl", required=True)
    p.add_argument("--segment", type=int, required=True)
    p.add_argument("--ckpt-dir", action="append", default=[],
                   help="directory of .safetensors to source bn1Mean/sae2/eff_alpha from "
                        "(matched by model_id+training_step, never by filename); repeatable")
    sub.add_parser("render")
    a = ap.parse_args()
    if a.cmd == "track":
        track(a.run)
    elif a.cmd == "migrate":
        migrate(a.run, a.loop_state)
    elif a.cmd == "recompute":
        targets = list(REG["runs"]) if a.run == "__all__" else [a.run]
        for r in targets:
            recompute_elapsed(r)
    elif a.cmd == "discover-stems":
        targets = list(REG["runs"]) if a.run == "__all__" else [a.run]
        proposed = {}
        for r in targets:
            cfg = REG["runs"][r]
            if not (cfg.get("out_model") or ""):
                continue
            print(f"{r}:")
            found = discover_enum_stems(cfg, r)
            have = {i: s.get("enum_stem") for i, s in enumerate(cfg["segments"]) if s.get("enum_stem")}
            new = {i: st for i, st in found.items() if have.get(i) != st}
            if have:
                print(f"  ({len(have)} segment(s) already declare enum_stem)")
            if new:
                proposed[r] = new
                for i, st in sorted(new.items()):
                    print(f"  + seg {i}: enum_stem = {st}")
            else:
                print("  nothing to add")
        if a.write and proposed:
            reg = json.load(open(os.path.join(ROOT, "registry.json")))
            for r, new in proposed.items():
                for i, st in new.items():
                    reg["runs"][r]["segments"][i]["enum_stem"] = st
            with open(os.path.join(ROOT, "registry.json"), "w") as fh:
                json.dump(reg, fh, indent=2, ensure_ascii=False)
            print(f"\nwrote {sum(len(v) for v in proposed.values())} enum_stem value(s) to registry.json")
        elif proposed:
            print("\n(proposal only — re-run with --write to apply)")
    elif a.cmd == "import-probes":
        import_probes(a.run, a.jsonl, a.segment, a.ckpt_dir)
    elif a.cmd == "render":
        render()
